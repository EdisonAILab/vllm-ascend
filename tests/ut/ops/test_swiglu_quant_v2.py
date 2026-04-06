"""Test npu_grouped_matmul_swiglu_quant_v2 batch-invariance on A5 device.

This operator is the fused GMM1 + SwiGLU + re-quantize for MoE MXFP8 path.
It's the most challenging operator for batch-invariance because:
1. group_list changes with batch size (different token-to-expert routing)
2. The fused operator may use different internal parallelization strategies
3. SwiGLU activation + re-quantization are fused with the matmul

Reference: docs/vllm-ascend-batch-invariant-analysis.md Section 7.2(4)
"""
import torch
import torch_npu

DEVICE = "npu"
DTYPE = torch.bfloat16
E8M0 = torch_npu.float8_e8m0fnu
GROUP_SIZE = 32

print("=" * 60)
print("Test: npu_grouped_matmul_swiglu_quant_v2 (A5)")
print("=" * 60)

# ── Setup: simulate MoE gate_up_proj (w13) ──
num_experts = 4
hidden = 4096
intermediate = 2048
gate_up_size = 2 * intermediate  # gate + up projection

gen = torch.Generator().manual_seed(42)
# Original weight shape: [num_experts, gate_up_size, hidden]
w13_bf16 = torch.randn(num_experts, gate_up_size, hidden,
                        generator=gen, dtype=DTYPE).to(DEVICE)

# Quantize each expert and apply process_weights_after_loading transforms
w13_fp8_list = []
w13_scale_list = []
for e in range(num_experts):
    wf, ws = torch_npu.npu_dynamic_mx_quant(w13_bf16[e], dst_type=torch.float8_e4m3fn)
    # wf: [gate_up_size, hidden], ws: [gate_up_size, hidden//GS//2, 2]
    w13_fp8_list.append(wf)
    w13_scale_list.append(ws)

# Stack and apply process_weights_after_loading transforms:
# weight: transpose (g, N, K) -> (g, K, N)
# scale: transpose (g, N, K//GS//2, 2) -> (g, K//GS//2, N, 2)
w13_fp8_raw = torch.stack([w.view(torch.uint8) for w in w13_fp8_list])
w13_fp8 = w13_fp8_raw.transpose(1, 2).contiguous().view(torch.float8_e4m3fn)
w13_scale_raw = torch.stack(w13_scale_list)
w13_scale = w13_scale_raw.transpose(1, 2).contiguous()

print(f"w13_fp8:   {w13_fp8.shape} {w13_fp8.dtype}")
print(f"w13_scale: {w13_scale.shape} {w13_scale.dtype}")
# Expected: w13_fp8 [4, hidden, gate_up_size], w13_scale [4, hidden//GS//2, gate_up_size, 2]

# ── Create input tokens and quantize ──
M = 64
gen_x = torch.Generator().manual_seed(99)
x_bf16 = torch.randn(M, hidden, generator=gen_x, dtype=DTYPE).to(DEVICE)
qx, sx = torch_npu.npu_dynamic_mx_quant(x_bf16, dst_type=torch.float8_e4m3fn)
# sx shape: [M, hidden//GS//2, 2]

print(f"qx: {qx.shape} {qx.dtype}")
print(f"sx: {sx.shape} {sx.dtype}")


def maybe_normalize_mxfp_scale_layout(scale):
    """A5DeviceAdaptor.maybe_normalize_mxfp_scale_layout"""
    if scale is None or scale.ndim != 2:
        return scale
    if scale.shape[-1] % 2 != 0:
        raise ValueError(f"Invalid MXFP8 scale shape: {tuple(scale.shape)}")
    return scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)


# ── Test 1: Basic call ──
print("\n--- Test 1: Basic npu_grouped_matmul_swiglu_quant_v2 call ---")
gl = torch.tensor([16, 32, 48, 64], dtype=torch.int64, device=DEVICE)

try:
    out, out_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
        x=qx,
        weight=[w13_fp8],
        group_list=gl,
        weight_scale=[w13_scale],
        x_scale=sx,
        dequant_mode=2,
        quant_mode=2,
        dequant_dtype=torch.float32,
        quant_dtype=torch.float8_e4m3fn,
        weight_scale_dtype=E8M0,
        x_scale_dtype=E8M0,
    )
    out_scale = maybe_normalize_mxfp_scale_layout(out_scale)
    print(f"  SUCCESS: out={out.shape} {out.dtype}, "
          f"out_scale={out_scale.shape} {out_scale.dtype}")
except Exception as e:
    print(f"  Error with dequant_dtype=torch.float32: {e}")
    print("\n  Trying alternative parameters...")

    # Try without dequant_dtype / with different values
    for dd in [None, 0, 1, 2]:
        for qd in [None, torch.float8_e4m3fn, 0, 1, 2]:
            try:
                kwargs = dict(
                    x=qx, weight=[w13_fp8], group_list=gl,
                    weight_scale=[w13_scale], x_scale=sx,
                    dequant_mode=2, quant_mode=2,
                    weight_scale_dtype=E8M0, x_scale_dtype=E8M0,
                )
                if dd is not None:
                    kwargs["dequant_dtype"] = dd
                if qd is not None:
                    kwargs["quant_dtype"] = qd
                out, out_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(**kwargs)
                out_scale = maybe_normalize_mxfp_scale_layout(out_scale)
                print(f"  SUCCESS with dequant_dtype={dd}, quant_dtype={qd}")
                print(f"  out={out.shape} {out.dtype}, "
                      f"out_scale={out_scale.shape} {out_scale.dtype}")
                break
            except Exception:
                pass
        else:
            continue
        break
    else:
        # Try with different dequant_mode/quant_mode combinations
        print("  Trying different mode combinations...")
        for dm in [0, 1, 2]:
            for qm in [0, 1, 2]:
                try:
                    out, out_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
                        x=qx, weight=[w13_fp8], group_list=gl,
                        weight_scale=[w13_scale], x_scale=sx,
                        dequant_mode=dm, quant_mode=qm,
                        weight_scale_dtype=E8M0, x_scale_dtype=E8M0,
                    )
                    out_scale = maybe_normalize_mxfp_scale_layout(out_scale)
                    print(f"  SUCCESS with dm={dm}, qm={qm}, no dtype params")
                    print(f"  out={out.shape} {out.dtype}, "
                          f"out_scale={out_scale.shape} {out_scale.dtype}")
                    break
                except Exception:
                    pass
            else:
                continue
            break
        else:
            print("  All combinations failed.")
            import sys
            sys.exit(1)

# ── Test 2: Batch-invariance with different group_lists ──
print("\n--- Test 2: Batch-invariance across different group_lists ---")


def call_swiglu_quant(qx_in, sx_in, group_list_in):
    """Call with working parameters for A5 internal test device.
    Note: dequant_dtype must be omitted (None) on this firmware,
    unlike production code which uses torch.float32.
    """
    out, out_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
        x=qx_in,
        weight=[w13_fp8],
        group_list=group_list_in,
        weight_scale=[w13_scale],
        x_scale=sx_in,
        dequant_mode=2,
        quant_mode=2,
        quant_dtype=torch.float8_e4m3fn,
        weight_scale_dtype=E8M0,
        x_scale_dtype=E8M0,
    )
    return out, maybe_normalize_mxfp_scale_layout(out_scale)


try:
    # gl1: uniform [16, 32, 48, 64]
    # gl2: non-uniform [8, 24, 48, 64]
    gl1 = torch.tensor([16, 32, 48, 64], dtype=torch.int64, device=DEVICE)
    gl2 = torch.tensor([8, 24, 48, 64], dtype=torch.int64, device=DEVICE)

    out1, os1 = call_swiglu_quant(qx, sx, gl1)
    out2, os2 = call_swiglu_quant(qx, sx, gl2)

    # Expert3 [48:64] is same in both group_lists
    e3_out_match = torch.equal(
        out1.view(torch.uint8)[48:64],
        out2.view(torch.uint8)[48:64]
    )
    e3_scale_match = torch.equal(os1[48:64], os2[48:64])

    if e3_out_match and e3_scale_match:
        print("  Expert3 [48:64]: EXACT MATCH (batch-invariant)")
    else:
        if not e3_out_match:
            d1 = out1.view(torch.uint8)[48:64].float()
            d2 = out2.view(torch.uint8)[48:64].float()
            dc = (d1 != d2).sum().item()
            tt = d1.numel()
            print(f"  Expert3 [48:64] output: MISMATCH dc={dc}/{tt} "
                  f"({dc/tt*100:.2f}%)")
        if not e3_scale_match:
            dc = (os1[48:64] != os2[48:64]).sum().item()
            tt = os1[48:64].numel()
            print(f"  Expert3 [48:64] scale: MISMATCH dc={dc}/{tt} "
                  f"({dc/tt*100:.2f}%)")

    # Expert0: [0:8] is in expert0 for both gl1 and gl2
    e0_out_match = torch.equal(
        out1.view(torch.uint8)[0:8],
        out2.view(torch.uint8)[0:8]
    )
    e0_scale_match = torch.equal(os1[0:8], os2[0:8])

    if e0_out_match and e0_scale_match:
        print("  Expert0 [0:8]:  EXACT MATCH (batch-invariant)")
    else:
        if not e0_out_match:
            d1 = out1.view(torch.uint8)[0:8].float()
            d2 = out2.view(torch.uint8)[0:8].float()
            dc = (d1 != d2).sum().item()
            tt = d1.numel()
            print(f"  Expert0 [0:8] output: MISMATCH dc={dc}/{tt} "
                  f"({dc/tt*100:.2f}%)")
        if not e0_scale_match:
            dc = (os1[0:8] != os2[0:8]).sum().item()
            tt = os1[0:8].numel()
            print(f"  Expert0 [0:8] scale: MISMATCH dc={dc}/{tt} "
                  f"({dc/tt*100:.2f}%)")

    # ── Test 3: More varied group_lists ──
    print("\n--- Test 3: More group_list variations ---")
    gl_variations = [
        ("uniform_16",   [16, 32, 48, 64]),
        ("front_heavy",  [32, 48, 56, 64]),
        ("back_heavy",   [4, 8, 16, 64]),
        ("single_token", [1, 2, 3, 64]),
    ]

    # Use the last 16 tokens (expert3: [48:64]) as the invariant check
    reference_e3_out = None
    reference_e3_scale = None

    for name, gl_vals in gl_variations:
        gl_t = torch.tensor(gl_vals, dtype=torch.int64, device=DEVICE)
        o, s = call_swiglu_quant(qx, sx, gl_t)

        if reference_e3_out is None:
            reference_e3_out = o.view(torch.uint8)[48:64].clone()
            reference_e3_scale = s[48:64].clone()
            print(f"  {name:15s}: reference set")
        else:
            out_match = torch.equal(o.view(torch.uint8)[48:64], reference_e3_out)
            scale_match = torch.equal(s[48:64], reference_e3_scale)
            status = "MATCH" if (out_match and scale_match) else "MISMATCH"
            if not out_match:
                dc = (o.view(torch.uint8)[48:64] != reference_e3_out).sum().item()
                tt = reference_e3_out.numel()
                print(f"  {name:15s}: {status} (out_diff={dc}/{tt})")
            elif not scale_match:
                dc = (s[48:64] != reference_e3_scale).sum().item()
                tt = reference_e3_scale.numel()
                print(f"  {name:15s}: {status} (scale_diff={dc}/{tt})")
            else:
                print(f"  {name:15s}: {status}")

    # ── Test 4: Repeated calls with same group_list (determinism) ──
    print("\n--- Test 4: Determinism (same inputs, 5 runs) ---")
    gl_det = torch.tensor([16, 32, 48, 64], dtype=torch.int64, device=DEVICE)
    ref_out, ref_scale = call_swiglu_quant(qx, sx, gl_det)
    all_deterministic = True
    for i in range(5):
        o, s = call_swiglu_quant(qx, sx, gl_det)
        if not torch.equal(o.view(torch.uint8), ref_out.view(torch.uint8)):
            all_deterministic = False
            print(f"  Run {i}: output MISMATCH")
        if not torch.equal(s, ref_scale):
            all_deterministic = False
            print(f"  Run {i}: scale MISMATCH")
    if all_deterministic:
        print("  All 5 runs: DETERMINISTIC")

except Exception as e:
    import traceback
    traceback.print_exc()

print("\nDone.")
