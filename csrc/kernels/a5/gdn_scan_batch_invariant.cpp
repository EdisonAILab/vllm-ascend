// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "kernel_operator.h"
#include "gdn_scan_batch_invariant.h"

// GdnScanBI v2 - stateful batch-invariant Gated DeltaNet scan.
// inputs:  q,k,v bf16 [B,T,128]; alpha,beta fp32 [B,T]; initState fp32 [B,128,128].
// outputs: out bf16 [B,T,128]; finalState fp32 [B,128,128].
// grid: blockDim = B * chunkCount; each block owns (batch b, column-chunk).
extern "C" __global__ __aicore__ void gdn_scan_batch_invariant(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR alpha, GM_ADDR beta,
    GM_ADDR initState, GM_ADDR out, GM_ADDR finalState,
    int64_t B, int64_t T, int64_t chunkCount)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ >= 3510
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    gdnscanbi_ops::GdnScanBIKernel op;
    op.Init(q, k, v, alpha, beta, initState, out, finalState, B, T, chunkCount,
            static_cast<int64_t>(AscendC::GetBlockIdx()));
    op.Process();
}

extern "C" __global__ __aicore__ void gdn_scatter_state_batch_invariant(
    GM_ADDR stateCache, GM_ADDR updates, GM_ADDR stateIndices,
    int64_t cacheRows, int64_t rows, int64_t rowElements)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ >= 3510
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    gdnscanbi_ops::GdnScatterStateBIKernel op;
    op.Init(stateCache, updates, stateIndices, cacheRows, rows, rowElements,
            static_cast<int64_t>(AscendC::GetBlockIdx()));
    op.Process();
}

namespace vllm_ascend {

void gdn_scan_batch_invariant_impl(
    void* stream,
    void* q,
    void* k,
    void* v,
    void* alpha,
    void* beta,
    void* initial_state,
    void* output,
    void* final_state,
    int64_t batch,
    int64_t tokens,
    int64_t chunk_count)
{
    const uint32_t block_dim = static_cast<uint32_t>(batch * chunk_count);
    gdn_scan_batch_invariant<<<block_dim, nullptr, stream>>>(
        q,
        k,
        v,
        alpha,
        beta,
        initial_state,
        output,
        final_state,
        batch,
        tokens,
        chunk_count);
}

void gdn_scatter_state_batch_invariant_impl(
    void* stream,
    void* state_cache,
    void* updates,
    void* state_indices,
    int64_t cache_rows,
    int64_t rows,
    int64_t row_elements)
{
    const uint32_t block_dim = static_cast<uint32_t>(rows);
    gdn_scatter_state_batch_invariant<<<block_dim, nullptr, stream>>>(
        state_cache,
        updates,
        state_indices,
        cache_rows,
        rows,
        row_elements);
}

} // namespace vllm_ascend
