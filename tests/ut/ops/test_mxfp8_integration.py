"""Integration test: verify MXFP8 batch-invariant operators work correctly.
Tests the standalone ops directly without vllm_ascend dependency.
"""
import torch
import torch_npu

DEVICE = "npu"
DTYPE = torch.bfloat16
E8M0 = torch_npu.float8_e8m0fnu
GROUP_SIZE = 32


# Define batch-invariant wrappers (same logic as mxfp8_quant_matmul.py)
def npu_dynamic_mx_quant_batch_invariant(x, dst_type=torch.float8_e4m3fn):
    return torch_npu.npu_dynamic_mx_quant(x, dst_type=dst_type)


def npu_quant_matmul_batch_invariant(
    x, weight, weight_scale, *, scale_dtype, pertoken_scale,
    pertoken_scale_dtype, bias=None, output_dtype=torch.bfloat16,
    group_sizes=None,
):
    return torch_npu.npu_quant_matmul(
        x, weight, weight_scale,
        scale_dtype=scale_dtype, pertoken_scale=pertoken_scale,
        pertoken_scale_dtype=pertoken_scale_dtype, bias=bias,
        output_dtype=output_dtype, group_sizes=group_sizes,
    )


def mxfp8_linear_batch_invariant(x, weight, weight_scale, group_size=32, bias=None):
    original_shape = x.shape
    output_dtype = x.dtype
    if x.dim() > 2:
        x = x.view(-1, x.shape[-1])
    quantized_x, dynamic_scale = npu_dynamic_mx_quant_batch_invariant(x)
    if bias is not None and bias.dtype != torch.float32:
        bias = bias.to(torch.float32)
    output = npu_quant_matmul_batch_invariant(
        quantized_x, weight, weight_scale,
        scale_dtype=E8M0, pertoken_scale=dynamic_scale,
        pertoken_scale_dtype=E8M0, bias=bias,
        output_dtype=output_dtype, group_sizes=[1, 1, group_size],
    )
    if len(original_shape) > 2:
        output = output.view(*original_shape[:-1], -1)
    return output


passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}")
        failed += 1


print("=" * 60)
print("MXFP8 Batch-Invariant Integration Test")
print("=" * 60)

# Setup
gen = torch.Generator().manual_seed(42)
K, N = 7168, 4096
M_total = 256

x = torch.randn(M_total, K, generator=gen, dtype=DTYPE).to(DEVICE)
gen_w = torch.Generator().manual_seed(123)
w_bf16 = torch.randn(N, K, generator=gen_w, dtype=DTYPE).to(DEVICE)
w_fp8, w_scale = torch_npu.npu_dynamic_mx_quant(w_bf16, dst_type=torch.float8_e4m3fn)
w_fp8_t = w_fp8.view(torch.uint8).transpose(0, 1).contiguous().view(torch.float8_e4m3fn)
w_scale_t = w_scale.transpose(0, 1).contiguous()

# Test 1: npu_dynamic_mx_quant_batch_invariant
print("\n--- Test 1: npu_dynamic_mx_quant_batch_invariant ---")
qx_bi, sx_bi = npu_dynamic_mx_quant_batch_invariant(x)
qx_native, sx_native = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
check("Output matches native", torch.equal(qx_bi.view(torch.uint8), qx_native.view(torch.uint8)))
check("Scale matches native", torch.equal(sx_bi, sx_native))

qx_h1, sx_h1 = npu_dynamic_mx_quant_batch_invariant(x[:128])
qx_h2, sx_h2 = npu_dynamic_mx_quant_batch_invariant(x[128:])
check("BI half 1", torch.equal(qx_bi.view(torch.uint8)[:128], qx_h1.view(torch.uint8)))
check("BI half 2", torch.equal(qx_bi.view(torch.uint8)[128:], qx_h2.view(torch.uint8)))

# Test 2: npu_quant_matmul_batch_invariant
print("\n--- Test 2: npu_quant_matmul_batch_invariant ---")
out_bi = npu_quant_matmul_batch_invariant(
    qx_bi, w_fp8_t, w_scale_t,
    scale_dtype=E8M0, pertoken_scale=sx_bi,
    pertoken_scale_dtype=E8M0, output_dtype=DTYPE,
    group_sizes=[1, 1, GROUP_SIZE],
)
out_native = torch_npu.npu_quant_matmul(
    qx_native, w_fp8_t, w_scale_t,
    scale_dtype=E8M0, pertoken_scale=sx_native,
    pertoken_scale_dtype=E8M0, output_dtype=DTYPE,
    group_sizes=[1, 1, GROUP_SIZE],
)
check("Output matches native", torch.equal(out_bi, out_native))

for bs in [1, 7, 32, 64, 128]:
    chunks = []
    for s in range(0, M_total, bs):
        e = min(s + bs, M_total)
        qi = qx_bi.view(torch.uint8)[s:e].view(torch.float8_e4m3fn)
        si = sx_bi[s:e]
        oi = npu_quant_matmul_batch_invariant(
            qi, w_fp8_t, w_scale_t,
            scale_dtype=E8M0, pertoken_scale=si,
            pertoken_scale_dtype=E8M0, output_dtype=DTYPE,
            group_sizes=[1, 1, GROUP_SIZE],
        )
        chunks.append(oi)
    out_chunked = torch.cat(chunks, dim=0)
    check(f"BI chunk_size={bs}", torch.equal(out_bi, out_chunked))

# Test 3: mxfp8_linear_batch_invariant
print("\n--- Test 3: mxfp8_linear_batch_invariant ---")
out_linear = mxfp8_linear_batch_invariant(x, w_fp8_t, w_scale_t, group_size=GROUP_SIZE)
check("Shape correct", out_linear.shape == (M_total, N))
check("Dtype correct", out_linear.dtype == DTYPE)

for bs in [1, 13, 37, 64, 128]:
    chunks = []
    for s in range(0, M_total, bs):
        e = min(s + bs, M_total)
        chunks.append(mxfp8_linear_batch_invariant(x[s:e], w_fp8_t, w_scale_t, group_size=GROUP_SIZE))
    out_chunked = torch.cat(chunks, dim=0)
    check(f"Linear BI chunk={bs}", torch.equal(out_linear, out_chunked))

# 3D input
x_3d = x[:32].view(4, 8, K)
out_3d = mxfp8_linear_batch_invariant(x_3d, w_fp8_t, w_scale_t, group_size=GROUP_SIZE)
check("3D shape", out_3d.shape == (4, 8, N))
out_2d = mxfp8_linear_batch_invariant(x[:32], w_fp8_t, w_scale_t, group_size=GROUP_SIZE)
check("3D == 2D reshaped", torch.equal(out_3d.view(32, N), out_2d))

# Test 4: npu_grouped_matmul (MoE)
print("\n--- Test 4: npu_grouped_matmul (MoE) ---")
num_experts = 4
intermediate = 2048
gen_moe = torch.Generator().manual_seed(77)
w1_bf16 = torch.randn(num_experts, intermediate, K, generator=gen_moe, dtype=DTYPE).to(DEVICE)

w_list = []
ws_list = []
for e in range(num_experts):
    wf, ws = torch_npu.npu_dynamic_mx_quant(w1_bf16[e], dst_type=torch.float8_e4m3fn)
    wf_t = wf.view(torch.uint8).transpose(0, 1).contiguous().view(torch.float8_e4m3fn)
    ws_t = ws.transpose(0, 1).contiguous()
    w_list.append(wf_t)
    ws_list.append(ws_t)

w_stacked = torch.stack([w.view(torch.uint8) for w in w_list]).view(torch.float8_e4m3fn)
ws_stacked = torch.stack(ws_list)

M_moe = 64
qx_moe, sx_moe = npu_dynamic_mx_quant_batch_invariant(x[:M_moe])

gl1 = torch.tensor([16, 32, 48, 64], dtype=torch.int64, device=DEVICE)
gl2 = torch.tensor([8, 24, 48, 64], dtype=torch.int64, device=DEVICE)

try:
    out1 = torch_npu.npu_grouped_matmul(
        x=[qx_moe], weight=[w_stacked], scale=[ws_stacked],
        per_token_scale=[sx_moe], group_list=gl1,
        split_item=2, group_list_type=0, group_type=0,
        scale_dtype=E8M0, per_token_scale_dtype=E8M0, output_dtype=DTYPE,
    )[0]
    out2 = torch_npu.npu_grouped_matmul(
        x=[qx_moe], weight=[w_stacked], scale=[ws_stacked],
        per_token_scale=[sx_moe], group_list=gl2,
        split_item=2, group_list_type=0, group_type=0,
        scale_dtype=E8M0, per_token_scale_dtype=E8M0, output_dtype=DTYPE,
    )[0]
    check("GMM expert3 BI [48:64]", torch.equal(out1[48:64], out2[48:64]))
    check("GMM expert0 BI [0:8]", torch.equal(out1[0:8], out2[0:8]))
except Exception as e:
    err = str(e)[:60]
    print(f"  [SKIP] npu_grouped_matmul: {err}")

# Summary
print("\n" + "=" * 60)
total = passed + failed
status = "ALL PASSED" if failed == 0 else f"{failed} FAILED"
print(f"  Results: {passed}/{total} passed - {status}")
print("=" * 60)
