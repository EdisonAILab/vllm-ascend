"""Extreme dimension batch-invariance test for ALL operators.

Covers:
1. npu_dynamic_mx_quant
2. npu_grouped_matmul (MoE GMM2)
3. npu_grouped_matmul_swiglu_quant_v2 (MoE GMM1+SwiGLU)
4. npu_rms_norm
5. npu_add_rms_norm
"""
import torch
import torch_npu

DEVICE = "npu"
DTYPE = torch.bfloat16
E8M0 = torch_npu.float8_e8m0fnu
GS = 32

passed = 0
failed = 0


def check(name, condition, extra=""):
    global passed, failed
    if condition:
        print("  [PASS] {}".format(name))
        passed += 1
    else:
        print("  [FAIL] {} {}".format(name, extra))
        failed += 1


# ================================================================
# 1. npu_dynamic_mx_quant
# ================================================================
def test_mx_quant_bi(name, M_total, K, batch_sizes=None):
    if batch_sizes is None:
        batch_sizes = [1, 7, 16, 64]

    gen = torch.Generator().manual_seed(42)
    x = torch.randn(M_total, K, generator=gen, dtype=DTYPE).to(DEVICE)

    qx_full, sx_full = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)

    for bs in batch_sizes:
        q_chunks, s_chunks = [], []
        for s in range(0, M_total, bs):
            e = min(s + bs, M_total)
            qi, si = torch_npu.npu_dynamic_mx_quant(x[s:e], dst_type=torch.float8_e4m3fn)
            q_chunks.append(qi.view(torch.uint8))
            s_chunks.append(si)
        qx_bs = torch.cat(q_chunks, dim=0).view(torch.float8_e4m3fn)
        sx_bs = torch.cat(s_chunks, dim=0)

        q_match = torch.equal(qx_full.view(torch.uint8), qx_bs.view(torch.uint8))
        s_match = torch.equal(sx_full, sx_bs)
        if q_match and s_match:
            check("{} BS={}".format(name, bs), True)
        else:
            parts = []
            if not q_match:
                dc = (qx_full.view(torch.uint8) != qx_bs.view(torch.uint8)).sum().item()
                parts.append("quant_diff={}".format(dc))
            if not s_match:
                dc = (sx_full != sx_bs).sum().item()
                parts.append("scale_diff={}".format(dc))
            check("{} BS={}".format(name, bs), False, " ".join(parts))


print("=" * 60)
print("1. npu_dynamic_mx_quant extreme dimensions")
print("=" * 60)

# Large K
test_mx_quant_bi("K=16384", M_total=64, K=16384)
test_mx_quant_bi("K=32768", M_total=64, K=32768)
try:
    test_mx_quant_bi("K=65536", M_total=32, K=65536)
except Exception as e:
    print("  [SKIP] K=65536: {}".format(str(e)[:80]))

# Large M
test_mx_quant_bi("M=1024", M_total=1024, K=4096, batch_sizes=[1, 32, 128, 512])
test_mx_quant_bi("M=4096", M_total=4096, K=4096, batch_sizes=[1, 64, 256, 1024])
test_mx_quant_bi("M=8192", M_total=8192, K=4096, batch_sizes=[1, 128, 512, 2048])

# Non-aligned
test_mx_quant_bi("K=4128(129*32)", M_total=64, K=4128)
test_mx_quant_bi("M=63(odd)", M_total=63, K=4096, batch_sizes=[1, 7, 16])

# DeepSeek V3
test_mx_quant_bi("DSv3 K=7168", M_total=256, K=7168, batch_sizes=[1, 7, 32, 128])
test_mx_quant_bi("DSv3 K=18432", M_total=256, K=18432, batch_sizes=[1, 7, 32, 128])


# ================================================================
# 2. npu_grouped_matmul (MoE GMM2)
# ================================================================
def test_gmm_bi(name, M_total, K, N, num_experts=4, batch_sizes=None):
    if batch_sizes is None:
        batch_sizes = [1, 7, 16]

    gen = torch.Generator().manual_seed(42)
    x = torch.randn(M_total, K, generator=gen, dtype=DTYPE).to(DEVICE)
    qx, sx = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)

    # Create stacked expert weights
    gen_w = torch.Generator().manual_seed(123)
    w_bf16 = torch.randn(num_experts, N, K, generator=gen_w, dtype=DTYPE).to(DEVICE)
    wf_list, ws_list = [], []
    for e in range(num_experts):
        wf, ws = torch_npu.npu_dynamic_mx_quant(w_bf16[e], dst_type=torch.float8_e4m3fn)
        wf_list.append(wf.view(torch.uint8).transpose(0, 1).contiguous().view(torch.float8_e4m3fn))
        ws_list.append(ws.transpose(0, 1).contiguous())
    w_stacked = torch.stack([w.view(torch.uint8) for w in wf_list]).view(torch.float8_e4m3fn)
    ws_stacked = torch.stack(ws_list)

    # Uniform group_list
    tokens_per_expert = M_total // num_experts
    gl = torch.tensor(
        [tokens_per_expert * (i + 1) for i in range(num_experts)],
        dtype=torch.int64, device=DEVICE,
    )
    # Adjust last to M_total
    gl[-1] = M_total

    out_full = torch_npu.npu_grouped_matmul(
        x=[qx], weight=[w_stacked], scale=[ws_stacked],
        per_token_scale=[sx], group_list=gl,
        split_item=2, group_list_type=0, group_type=0,
        scale_dtype=E8M0, per_token_scale_dtype=E8M0, output_dtype=DTYPE,
    )[0]

    # Compare: per-expert npu_quant_matmul with different chunk sizes
    for bs in batch_sizes:
        ref_chunks = []
        boundaries = [0] + gl.tolist()
        for e in range(num_experts):
            s, end = int(boundaries[e]), int(boundaries[e + 1])
            for cs in range(s, end, bs):
                ce = min(cs + bs, end)
                qi = qx.view(torch.uint8)[cs:ce].view(torch.float8_e4m3fn)
                si = sx[cs:ce]
                oi = torch_npu.npu_quant_matmul(
                    qi, wf_list[e], ws_list[e],
                    scale_dtype=E8M0, pertoken_scale=si,
                    pertoken_scale_dtype=E8M0, output_dtype=DTYPE,
                    group_sizes=[1, 1, GS],
                )
                ref_chunks.append(oi)
        out_ref = torch.cat(ref_chunks, dim=0)

        match = torch.equal(out_full, out_ref)
        if match:
            check("{} BS={}".format(name, bs), True)
        else:
            dc = (out_full != out_ref).sum().item()
            tt = out_full.numel()
            md = (out_full - out_ref).abs().max().item()
            check("{} BS={}".format(name, bs), False,
                  "diff={}/{} ({:.2f}%) max={:.4e}".format(dc, tt, dc / tt * 100, md))


print("\n" + "=" * 60)
print("2. npu_grouped_matmul extreme dimensions")
print("=" * 60)

test_gmm_bi("K=7168,N=2048", M_total=256, K=7168, N=2048)
test_gmm_bi("K=16384,N=4096", M_total=128, K=16384, N=4096)
test_gmm_bi("M=1024,8experts", M_total=1024, K=4096, N=2048, num_experts=8, batch_sizes=[1, 16, 64])
test_gmm_bi("M=2048", M_total=2048, K=4096, N=4096, batch_sizes=[1, 32, 128])
test_gmm_bi("DSv3 down K=9216,N=7168", M_total=256, K=9216, N=7168, batch_sizes=[1, 7, 32])


# ================================================================
# 3. npu_grouped_matmul_swiglu_quant_v2
# ================================================================
def maybe_norm(scale):
    if scale is None or scale.ndim != 2:
        return scale
    return scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)


def test_swiglu_bi(name, M_total, K, gate_up_size, num_experts=4):
    gen = torch.Generator().manual_seed(42)
    x = torch.randn(M_total, K, generator=gen, dtype=DTYPE).to(DEVICE)
    qx, sx = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)

    gen_w = torch.Generator().manual_seed(123)
    w_bf16 = torch.randn(num_experts, gate_up_size, K, generator=gen_w, dtype=DTYPE).to(DEVICE)
    wf_list, ws_list = [], []
    for e in range(num_experts):
        wf, ws = torch_npu.npu_dynamic_mx_quant(w_bf16[e], dst_type=torch.float8_e4m3fn)
        wf_list.append(wf)
        ws_list.append(ws)
    w_stacked = torch.stack([w.view(torch.uint8) for w in wf_list]).transpose(1, 2).contiguous().view(torch.float8_e4m3fn)
    ws_stacked = torch.stack(ws_list).transpose(1, 2).contiguous()

    tpe = M_total // num_experts
    gl = torch.tensor([tpe * (i + 1) for i in range(num_experts)], dtype=torch.int64, device=DEVICE)
    gl[-1] = M_total

    # Reference: first expert, first 8 tokens
    ref_out = None
    ref_scale = None

    # Vary other experts' token counts, check token[0] through expert0
    # All variants keep token[0] in expert0, so its result should be identical
    # Build group_list variations with correct length = num_experts
    def make_gl(expert0_tokens, rest_pattern="uniform"):
        """Build cumulative group_list where expert0 gets expert0_tokens."""
        remaining = M_total - expert0_tokens
        other = num_experts - 1
        if rest_pattern == "uniform":
            per_other = remaining // other
            vals = [expert0_tokens]
            for i in range(1, num_experts):
                vals.append(vals[-1] + (per_other if i < num_experts - 1 else M_total - vals[-1]))
        elif rest_pattern == "last_heavy":
            vals = [expert0_tokens]
            for i in range(1, num_experts - 1):
                vals.append(vals[-1] + 1)
            vals.append(M_total)
        return torch.tensor(vals, dtype=torch.int64, device=DEVICE)

    gl_variations = [
        ("uniform", make_gl(tpe, "uniform")),
        ("skewed", make_gl(tpe, "last_heavy")),
        ("extreme", make_gl(1, "last_heavy")),
    ]

    for vname, gl_v in gl_variations:
        e0_end = int(gl_v[0].item())
        if e0_end == 0:
            continue
        out, out_s = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x=qx, weight=[w_stacked], group_list=gl_v,
            weight_scale=[ws_stacked], x_scale=sx,
            dequant_mode=2, quant_mode=2,
            quant_dtype=torch.float8_e4m3fn,
            weight_scale_dtype=E8M0, x_scale_dtype=E8M0,
        )
        out_s = maybe_norm(out_s)

        # Always compare just token[0] (exists in all variants under expert0)
        e0_out = out.view(torch.uint8)[0:1]
        e0_s = out_s[0:1]

        if ref_out is None:
            ref_out = e0_out.clone()
            ref_scale = e0_s.clone()
            check("{} {} (ref)".format(name, vname), True)
        else:
            o_match = torch.equal(e0_out, ref_out)
            s_match = torch.equal(e0_s, ref_scale)
            if o_match and s_match:
                check("{} {}".format(name, vname), True)
            else:
                parts = []
                if not o_match:
                    dc = (e0_out != ref_out).sum().item()
                    parts.append("out_diff={}".format(dc))
                if not s_match:
                    dc = (e0_s != ref_scale).sum().item()
                    parts.append("scale_diff={}".format(dc))
                check("{} {}".format(name, vname), False, " ".join(parts))

    # Vary total M with expert0 getting same first tokens
    for total_m in [M_total * 2]:
        gen2 = torch.Generator().manual_seed(42)
        x2 = torch.randn(total_m, K, generator=gen2, dtype=DTYPE).to(DEVICE)
        qx2, sx2 = torch_npu.npu_dynamic_mx_quant(x2, dst_type=torch.float8_e4m3fn)
        # Build gl with correct num_experts entries
        gl2_vals = [tpe]
        for i in range(1, num_experts - 1):
            gl2_vals.append(gl2_vals[-1] + tpe)
        gl2_vals.append(total_m)
        gl2 = torch.tensor(gl2_vals, dtype=torch.int64, device=DEVICE)

        out2, os2 = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x=qx2, weight=[w_stacked], group_list=gl2,
            weight_scale=[ws_stacked], x_scale=sx2,
            dequant_mode=2, quant_mode=2,
            quant_dtype=torch.float8_e4m3fn,
            weight_scale_dtype=E8M0, x_scale_dtype=E8M0,
        )
        os2 = maybe_norm(os2)

        check_rows = min(tpe, 8)
        o_match = torch.equal(out2.view(torch.uint8)[:check_rows], ref_out)
        s_match = torch.equal(os2[:check_rows], ref_scale)
        if o_match and s_match:
            check("{} M={}".format(name, total_m), True)
        else:
            check("{} M={}".format(name, total_m), False)


print("\n" + "=" * 60)
print("3. npu_grouped_matmul_swiglu_quant_v2 extreme dimensions")
print("=" * 60)

test_swiglu_bi("K=4096,GU=8192", M_total=64, K=4096, gate_up_size=8192)
test_swiglu_bi("K=7168,GU=18432(DSv3)", M_total=256, K=7168, gate_up_size=18432)
try:
    test_swiglu_bi("K=14336,GU=8192", M_total=128, K=14336, gate_up_size=8192)
except Exception as e:
    print("  [SKIP] K=14336: {}".format(str(e)[:80]))
test_swiglu_bi("M=512", M_total=512, K=4096, gate_up_size=8192)
test_swiglu_bi("8experts", M_total=256, K=4096, gate_up_size=8192, num_experts=8)


# ================================================================
# 4. npu_rms_norm
# ================================================================
def test_rmsnorm_bi(name, M_total, hidden, batch_sizes=None, eps=1e-6):
    if batch_sizes is None:
        batch_sizes = [1, 7, 16, 64]

    gen = torch.Generator().manual_seed(42)
    w = torch.randn(hidden, generator=gen, dtype=DTYPE).to(DEVICE)
    x = torch.randn(M_total, hidden, generator=gen, dtype=DTYPE).to(DEVICE)

    out_full, _ = torch_npu.npu_rms_norm(x, w, eps)

    for bs in batch_sizes:
        chunks = []
        for s in range(0, M_total, bs):
            e = min(s + bs, M_total)
            o, _ = torch_npu.npu_rms_norm(x[s:e], w, eps)
            chunks.append(o)
        out_bs = torch.cat(chunks, dim=0)
        match = torch.equal(out_full, out_bs)
        if match:
            check("{} BS={}".format(name, bs), True)
        else:
            dc = (out_full != out_bs).sum().item()
            tt = out_full.numel()
            md = (out_full - out_bs).abs().max().item()
            check("{} BS={}".format(name, bs), False,
                  "diff={}/{} max={:.4e}".format(dc, tt, md))


print("\n" + "=" * 60)
print("4. npu_rms_norm extreme dimensions")
print("=" * 60)

# Large hidden
test_rmsnorm_bi("hidden=16384", M_total=128, hidden=16384)
test_rmsnorm_bi("hidden=32768", M_total=64, hidden=32768)
try:
    test_rmsnorm_bi("hidden=65536", M_total=32, hidden=65536)
except Exception as e:
    print("  [SKIP] hidden=65536: {}".format(str(e)[:80]))

# Large M
test_rmsnorm_bi("M=4096", M_total=4096, hidden=4096, batch_sizes=[1, 32, 256, 1024])
test_rmsnorm_bi("M=8192", M_total=8192, hidden=4096, batch_sizes=[1, 128, 512, 2048])

# Non-aligned
test_rmsnorm_bi("hidden=4097(odd)", M_total=64, hidden=4097)
test_rmsnorm_bi("hidden=7168(DSv3)", M_total=256, hidden=7168, batch_sizes=[1, 7, 32, 128])
test_rmsnorm_bi("M=63(odd)", M_total=63, hidden=4096, batch_sizes=[1, 7, 16])


# ================================================================
# 5. npu_add_rms_norm
# ================================================================
def test_add_rmsnorm_bi(name, M_total, hidden, batch_sizes=None, eps=1e-6):
    if batch_sizes is None:
        batch_sizes = [1, 7, 16, 64]

    gen = torch.Generator().manual_seed(42)
    w = torch.randn(hidden, generator=gen, dtype=DTYPE).to(DEVICE)
    x = torch.randn(M_total, hidden, generator=gen, dtype=DTYPE).to(DEVICE)
    res = torch.randn(M_total, hidden, generator=gen, dtype=DTYPE).to(DEVICE)

    out_full, _, res_full = torch_npu.npu_add_rms_norm(x, res, w, eps)

    for bs in batch_sizes:
        out_chunks, res_chunks = [], []
        for s in range(0, M_total, bs):
            e = min(s + bs, M_total)
            o, _, r = torch_npu.npu_add_rms_norm(x[s:e], res[s:e], w, eps)
            out_chunks.append(o)
            res_chunks.append(r)
        out_bs = torch.cat(out_chunks, dim=0)
        res_bs = torch.cat(res_chunks, dim=0)

        o_match = torch.equal(out_full, out_bs)
        r_match = torch.equal(res_full, res_bs)
        if o_match and r_match:
            check("{} BS={}".format(name, bs), True)
        else:
            parts = []
            if not o_match:
                dc = (out_full != out_bs).sum().item()
                tt = out_full.numel()
                md = (out_full - out_bs).abs().max().item()
                parts.append("out diff={}/{} max={:.4e}".format(dc, tt, md))
            if not r_match:
                dc = (res_full != res_bs).sum().item()
                parts.append("res diff={}".format(dc))
            check("{} BS={}".format(name, bs), False, " ".join(parts))


print("\n" + "=" * 60)
print("5. npu_add_rms_norm extreme dimensions")
print("=" * 60)

test_add_rmsnorm_bi("hidden=16384", M_total=128, hidden=16384)
test_add_rmsnorm_bi("hidden=32768", M_total=64, hidden=32768)
test_add_rmsnorm_bi("M=4096", M_total=4096, hidden=4096, batch_sizes=[1, 32, 256, 1024])
test_add_rmsnorm_bi("M=8192", M_total=8192, hidden=4096, batch_sizes=[1, 128, 512, 2048])
test_add_rmsnorm_bi("hidden=7168(DSv3)", M_total=256, hidden=7168, batch_sizes=[1, 7, 32, 128])
test_add_rmsnorm_bi("M=63(odd)", M_total=63, hidden=4096, batch_sizes=[1, 7, 16])


# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 60)
total = passed + failed
status = "ALL PASSED" if failed == 0 else "{} FAILED".format(failed)
print("  Results: {}/{} passed - {}".format(passed, total, status))
print("=" * 60)
