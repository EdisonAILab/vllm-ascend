// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

#ifndef GDN_SCAN_BATCH_INVARIANT_H_
#define GDN_SCAN_BATCH_INVARIANT_H_

#include "kernel_operator.h"
#include <cstdint>

namespace gdnscanbi_ops {
using namespace AscendC;

// Spec: DK = DV = 128 (single head, Qwen3.5 linear_key/value_head_dim).
constexpr int32_t DK = 128;
constexpr int32_t DV = 128;
constexpr int32_t VEC_FP32_ELEMS = 64;   // fp32 elements per WholeReduce iteration

__aicore__ inline int32_t Align8(int32_t n) { return ((n + 7) / 8) * 8; }

__aicore__ inline int32_t FloorPow2(int32_t n) {
    n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16;
    return (n + 1) >> 1;
}

// Destructive fp32 reduce of src[0..count) -> returns sum (left in src[0]).
// Fixed binary-fold order -> chunk-independent reduction order (BI requirement).
// count <= 128.
__aicore__ inline float BinaryFoldReduceSum(const LocalTensor<float>& src, int32_t count) {
    if (count > VEC_FP32_ELEMS) {
        int32_t body = FloorPow2(count);
        int32_t tail = count - body;
        if (tail > 0) {
            int32_t tailAligned = Align8(tail);
            Add(src, src, src[body], tailAligned);
            PipeBarrier<PIPE_V>();
        }
        while (body > VEC_FP32_ELEMS) {
            body = body / 2;
            Add(src, src, src[body], body);
            PipeBarrier<PIPE_V>();
        }
        AscendCUtils::SetMask<float>(VEC_FP32_ELEMS);
    } else {
        AscendCUtils::SetMask<float>(count);
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        WholeReduceSum<float, false>(src, src, MASK_PLACEHOLDER, 1, 0, 1, 0);
    } else {
        WholeReduceSum<float, false>(src, src, MASK_PLACEHOLDER, 1, 1, 1, 8);
    }
#else
    WholeReduceSum<float, false>(src, src, MASK_PLACEHOLDER, 1, 1, 1, 8);
#endif
    event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(ev);
    WaitFlag<HardEvent::V_S>(ev);
    float result = src.GetValue(0);
    AscendCUtils::ResetMask();
    return result;
}

// Stateful GDN scan kernel.
// Per-step canonical fixed order (matches pb_gdn_design_oracle.recurrent_canonical):
//   S    *= a        (a = exp(glog_t), supplied as alpha)
//   kS_e  = sum_k(S[e,k] * kt[k])        (over DECAYED S)
//   vt_e  = vt_e - kS_e
//   vt_e *= beta
//   S[e]  += vt_e * kt                   (rank-1 outer)
//   o_e    = sum_k(S[e,k] * qt[k])       (qt already L2-normed * scale)
// State S [V,K] fp32. initState in, finalState out -> chunk-carry support.
class GdnScanBIKernel {
public:
    __aicore__ inline GdnScanBIKernel() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                GM_ADDR alpha, GM_ADDR beta,
                                GM_ADDR initState, GM_ADDR out, GM_ADDR finalState,
                                int64_t B, int64_t T, int64_t chunkCount,
                                int64_t blockIdx) {
        B_ = static_cast<int32_t>(B);
        T_ = static_cast<int32_t>(T);
        chunkCount_ = static_cast<int32_t>(chunkCount);
        width_ = DV / chunkCount_;                 // columns owned by this core
        b_ = static_cast<int32_t>(blockIdx) / chunkCount_;
        int32_t chunkIdx = static_cast<int32_t>(blockIdx) % chunkCount_;
        e0_ = chunkIdx * width_;
        active_ = (b_ < B_);

        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(q));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(v));
        alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(alpha));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta));
        initStateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(initState));
        oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out));
        finalStateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(finalState));

        // Persistent state + scratch (TBuf - pure VEC pipeline, ordered automatically).
        pipe_.InitBuffer(sBuf_, static_cast<uint32_t>(width_) * DK * sizeof(float));
        pipe_.InitBuffer(alphaBuf_, static_cast<uint32_t>(T_) * sizeof(float));
        pipe_.InitBuffer(betaBuf_, static_cast<uint32_t>(T_) * sizeof(float));
        pipe_.InitBuffer(qtBuf_, DK * sizeof(float));
        pipe_.InitBuffer(ktBuf_, DK * sizeof(float));
        pipe_.InitBuffer(vtBuf_, DK * sizeof(float));
        pipe_.InitBuffer(prodBuf_, DK * sizeof(float));
        pipe_.InitBuffer(a1Buf_, DK * sizeof(float));
        pipe_.InitBuffer(oFBuf_, static_cast<uint32_t>(width_) * sizeof(float));
        // Per-timestep IO queues (depth 2 for MTE2/VEC/MTE3 overlap).
        pipe_.InitBuffer(inQueue_, 2, 3 * DK * sizeof(bfloat16_t));
        pipe_.InitBuffer(outQueue_, 2, static_cast<uint32_t>(width_) * sizeof(bfloat16_t));
        // State IO queues (fp32 width_*DK).
        pipe_.InitBuffer(stInQueue_, 1, static_cast<uint32_t>(width_) * DK * sizeof(float));
        pipe_.InitBuffer(stOutQueue_, 1, static_cast<uint32_t>(width_) * DK * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (!active_) return;

        LocalTensor<float> S = sBuf_.Get<float>();
        LocalTensor<float> alphaL = alphaBuf_.Get<float>();
        LocalTensor<float> betaL = betaBuf_.Get<float>();
        LocalTensor<float> qtF = qtBuf_.Get<float>();
        LocalTensor<float> ktF = ktBuf_.Get<float>();
        LocalTensor<float> vtF = vtBuf_.Get<float>();
        LocalTensor<float> prod = prodBuf_.Get<float>();
        LocalTensor<float> a1 = a1Buf_.Get<float>();
        LocalTensor<float> oF = oFBuf_.Get<float>();

        // Load alpha[b,:], beta[b,:] once (T fp32, arbitrary T -> DataCopyPad).
        DataCopyExtParams cpAB{1u, static_cast<uint32_t>(T_) * 4u, 0u, 0u, 0u};
        DataCopyPadExtParams<float> padAB{false, 0u, 0u, 0.0f};
        DataCopyPad(alphaL, alphaGm_[static_cast<int64_t>(b_) * T_], cpAB, padAB);
        DataCopyPad(betaL, betaGm_[static_cast<int64_t>(b_) * T_], cpAB, padAB);
        event_t evAB = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(evAB);
        WaitFlag<HardEvent::MTE2_S>(evAB);

        // --- Load initial state S[e0:e0+width, :] from GM (fp32, row-contiguous) ---
        // initState layout: [B, V, K] row-major -> base = (b*V + e0)*K, width_*DK contiguous.
        LocalTensor<float> stIn = stInQueue_.AllocTensor<float>();
        int64_t stBase = (static_cast<int64_t>(b_) * DV + e0_) * DK;
        DataCopy(stIn, initStateGm_[stBase], width_ * DK);
        stInQueue_.EnQue(stIn);
        stIn = stInQueue_.DeQue<float>();
        Adds(S, stIn, 0.0f, width_ * DK);   // VEC move into persistent S TBuf
        stInQueue_.FreeTensor(stIn);
        PipeBarrier<PIPE_V>();

        for (int32_t t = 0; t < T_; ++t) {
            float alpha_t = alphaL.GetValue(t);
            float beta_t = betaL.GetValue(t);

            // --- load q/k/v[b,t,:] (128 bf16 each) ---
            LocalTensor<bfloat16_t> inL = inQueue_.AllocTensor<bfloat16_t>();
            int64_t base = (static_cast<int64_t>(b_) * T_ + t) * DK;
            DataCopy(inL[0], qGm_[base], DK);
            DataCopy(inL[DK], kGm_[base], DK);
            DataCopy(inL[2 * DK], vGm_[base], DK);
            inQueue_.EnQue(inL);
            inL = inQueue_.DeQue<bfloat16_t>();

            // bf16 -> fp32 (widening, exact)
            Cast(qtF, inL[0], RoundMode::CAST_NONE, DK);
            Cast(ktF, inL[DK], RoundMode::CAST_NONE, DK);
            Cast(vtF, inL[2 * DK], RoundMode::CAST_NONE, DK);
            inQueue_.FreeTensor(inL);
            PipeBarrier<PIPE_V>();

            // --- L2 normalize qt, kt in fp32; qt additionally *= scale (K**-0.5) ---
            NormalizeInPlace(qtF, prod);
            NormalizeInPlace(ktF, prod);
            Muls(qtF, qtF, SCALE, DK);   // q scaled by K**-0.5 (oracle: l2norm(q)*scale)
            PipeBarrier<PIPE_V>();

            // vtF must be readable by scalar GetValue inside the column loop.
            event_t evV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(evV);
            WaitFlag<HardEvent::V_S>(evV);

            // --- per-column recurrence body (oracle canonical order) ---
            for (int32_t el = 0; el < width_; ++el) {
                int32_t e = e0_ + el;
                LocalTensor<float> Scol = S[el * DK];

                // S *= a   (decay first)
                Muls(Scol, Scol, alpha_t, DK);
                PipeBarrier<PIPE_V>();

                // kS_e = sum_k(Scol[k] * kt[k])   over DECAYED S
                Mul(prod, ktF, Scol, DK);
                float kS_e = BinaryFoldReduceSum(prod, DK);

                // vt_e = (v_e - kS_e) * beta
                float vt_e = vtF.GetValue(e);
                vt_e = vt_e - kS_e;
                vt_e = vt_e * beta_t;

                // Scol += vt_e * kt   (rank-1 outer)
                Muls(a1, ktF, vt_e, DK);
                Add(Scol, Scol, a1, DK);
                PipeBarrier<PIPE_V>();

                // o_e = sum_k(Scol[k] * qt[k])   (qt already scaled)
                Mul(prod, qtF, Scol, DK);
                float o_e = BinaryFoldReduceSum(prod, DK);

                oF.SetValue(el, o_e);
            }

            // oF (scalar-written, S pipe) -> Cast (VEC) needs S->V sync
            event_t evSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(evSV);
            WaitFlag<HardEvent::S_V>(evSV);

            // --- store o[b,t,e0:e0+width] (bf16, contiguous over e) ---
            LocalTensor<bfloat16_t> outL = outQueue_.AllocTensor<bfloat16_t>();
            Cast(outL, oF, RoundMode::CAST_RINT, width_);   // OL-81: RINT
            outQueue_.EnQue(outL);
            outL = outQueue_.DeQue<bfloat16_t>();
            int64_t oBase = (static_cast<int64_t>(b_) * T_ + t) * DV + e0_;
            DataCopy(oGm_[oBase], outL, width_);
            outQueue_.FreeTensor(outL);
        }

        // --- store final state S[e0:e0+width, :] back to GM (fp32) ---
        LocalTensor<float> stOut = stOutQueue_.AllocTensor<float>();
        Adds(stOut, S, 0.0f, width_ * DK);
        stOutQueue_.EnQue(stOut);
        stOut = stOutQueue_.DeQue<float>();
        int64_t fsBase = (static_cast<int64_t>(b_) * DV + e0_) * DK;
        DataCopy(finalStateGm_[fsBase], stOut, width_ * DK);
        stOutQueue_.FreeTensor(stOut);
    }

private:
    static constexpr float SCALE = 0.08838834764831845f;  // 128 ** -0.5

    // scalar rsqrt via vector Rsqrt on an 8-element scratch (proven 11_GroupNorm pattern).
    __aicore__ inline float ComputeRsqrtScalar(const LocalTensor<float>& work, float val) {
        event_t evSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        work.SetValue(0, val);
        SetFlag<HardEvent::S_V>(evSV);
        WaitFlag<HardEvent::S_V>(evSV);
        AscendCUtils::SetMask<float>(8);
        Rsqrt(work, work, 8);
        PipeBarrier<PIPE_V>();
        event_t evVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(evVS);
        WaitFlag<HardEvent::V_S>(evVS);
        float r = work.GetValue(0);
        AscendCUtils::ResetMask();
        return r;
    }

    // x[d] *= 1 / sqrt(sum_d x^2 + eps)   (== F.normalize fp32 eps=1e-6 additive)
    __aicore__ inline void NormalizeInPlace(const LocalTensor<float>& x,
                                            const LocalTensor<float>& scratch) {
        Mul(scratch, x, x, DK);
        float ss = BinaryFoldReduceSum(scratch, DK);
        float denom = ss + 1e-6f;                           // oracle eps=1e-6 (additive)
        float inv = ComputeRsqrtScalar(scratch, denom);     // 1/sqrt(denom)
        Muls(x, x, inv, DK);
        PipeBarrier<PIPE_V>();
    }

    TPipe pipe_;
    GlobalTensor<bfloat16_t> qGm_, kGm_, vGm_, oGm_;
    GlobalTensor<float> alphaGm_, betaGm_, initStateGm_, finalStateGm_;
    TBuf<TPosition::VECCALC> sBuf_, alphaBuf_, betaBuf_, qtBuf_, ktBuf_, vtBuf_;
    TBuf<TPosition::VECCALC> prodBuf_, a1Buf_, oFBuf_;
    TQue<QuePosition::VECIN, 2> inQueue_;
    TQue<QuePosition::VECOUT, 2> outQueue_;
    TQue<QuePosition::VECIN, 1> stInQueue_;
    TQue<QuePosition::VECOUT, 1> stOutQueue_;

    int32_t B_, T_, chunkCount_, width_, b_, e0_;
    bool active_;
};

// Graph-safe persistent-state update. One block owns one source row and skips
// NULL_BLOCK_ID rows instead of allowing -1 to alias the final cache row.
class GdnScatterStateBIKernel {
public:
    static constexpr int32_t COPY_TILE_ELEMS = 8192;

    __aicore__ inline void Init(
        GM_ADDR stateCache,
        GM_ADDR updates,
        GM_ADDR stateIndices,
        int64_t cacheRows,
        int64_t rows,
        int64_t rowElements,
        int64_t blockIdx) {
        row_ = static_cast<int32_t>(blockIdx);
        cacheRows_ = static_cast<int32_t>(cacheRows);
        rows_ = static_cast<int32_t>(rows);
        rowElements_ = static_cast<int32_t>(rowElements);
        active_ = row_ < rows_;
        stateCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(stateCache));
        updatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(updates));
        stateIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(stateIndices));
        pipe_.InitBuffer(copyBuf_, COPY_TILE_ELEMS * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (!active_) return;
        const int32_t targetRow = stateIndicesGm_.GetValue(row_);
        if (targetRow < 0 || targetRow >= cacheRows_) return;

        LocalTensor<float> copy = copyBuf_.Get<float>();
        constexpr event_t event = EVENT_ID0;
        SetFlag<HardEvent::MTE3_MTE2>(event);
        for (int32_t offset = 0; offset < rowElements_; offset += COPY_TILE_ELEMS) {
            const int32_t count =
                (rowElements_ - offset < COPY_TILE_ELEMS)
                    ? rowElements_ - offset
                    : COPY_TILE_ELEMS;
            WaitFlag<HardEvent::MTE3_MTE2>(event);
            DataCopy(
                copy,
                updatesGm_[static_cast<int64_t>(row_) * rowElements_ + offset],
                count);
            SetFlag<HardEvent::MTE2_MTE3>(event);
            WaitFlag<HardEvent::MTE2_MTE3>(event);
            DataCopy(
                stateCacheGm_[static_cast<int64_t>(targetRow) * rowElements_ + offset],
                copy,
                count);
            SetFlag<HardEvent::MTE3_MTE2>(event);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(event);
    }

private:
    TPipe pipe_;
    GlobalTensor<float> stateCacheGm_, updatesGm_;
    GlobalTensor<int32_t> stateIndicesGm_;
    TBuf<TPosition::VECCALC> copyBuf_;
    int32_t row_, cacheRows_, rows_, rowElements_;
    bool active_;
};

}  // namespace gdnscanbi_ops

#endif  // GDN_SCAN_BATCH_INVARIANT_H_
