"""
MXFP8 Batch-Invariant Operator Test Suite
==========================================
Tests batch-invariance properties of MXFP8 operators using pure PyTorch.
No Triton or NPU hardware required — runs on CPU.

Usage:
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 test_mxfp8_batch_invariant.py
"""

import torch
import math
import sys

# ============================================================
# MXFP8 Utility Functions (Software Emulation)
# ============================================================

# FP8 E4M3FN range: [-448, 448]
FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN = -448.0

def quantize_to_fp8_e4m3fn(x: torch.Tensor) -> torch.Tensor:
    """Simulate FP8 E4M3FN quantization via cast round-trip."""
    return x.to(torch.float8_e4m3fn)

def compute_mx_scale(x: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """
    Compute microscaling scale factors (E8M0FNU format = power of 2).

    For each group of `group_size` elements along the last dim,
    compute scale = 2^floor(log2(amax(abs(group)))).
    """
    orig_shape = x.shape
    assert orig_shape[-1] % group_size == 0, (
        f"Last dim {orig_shape[-1]} not divisible by group_size {group_size}"
    )

    # Reshape to expose groups: (..., num_groups, group_size)
    x_grouped = x.reshape(*orig_shape[:-1], -1, group_size)

    # Per-group absmax
    amax = x_grouped.abs().amax(dim=-1)  # (..., num_groups)

    # E8M0FNU: power-of-2 scales
    # Clamp to avoid log2(0)
    amax = amax.clamp(min=1e-12)
    scale = torch.pow(2.0, torch.floor(torch.log2(amax)))

    return scale


def dynamic_mx_quant(x_bf16: torch.Tensor, group_size: int = 32):
    """
    Simulate npu_dynamic_mx_quant: BF16 -> (FP8, scale).

    Returns:
        quantized: torch.float8_e4m3fn tensor
        scale: per-group power-of-2 scales
    """
    orig_shape = x_bf16.shape
    x_flat = x_bf16.reshape(-1, x_bf16.shape[-1]).float()

    # Compute per-group scales
    scale = compute_mx_scale(x_flat, group_size)  # (rows, num_groups)

    # Quantize: divide by scale, clamp to FP8 range, cast
    num_groups = x_flat.shape[-1] // group_size
    x_grouped = x_flat.reshape(x_flat.shape[0], num_groups, group_size)
    scale_expanded = scale.unsqueeze(-1)  # (rows, num_groups, 1)

    x_scaled = x_grouped / scale_expanded
    x_scaled = x_scaled.clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)

    # Cast to FP8 (round-trip through dtype)
    x_fp8 = x_scaled.reshape(x_flat.shape).to(torch.float8_e4m3fn)

    return x_fp8.reshape(orig_shape), scale.reshape(*orig_shape[:-1], -1)


def dequantize_mx(x_fp8: torch.Tensor, scale: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Dequantize: FP8 * scale -> FP32."""
    orig_shape = x_fp8.shape
    x_flat = x_fp8.reshape(-1, x_fp8.shape[-1]).to(torch.float32)
    scale_flat = scale.reshape(x_flat.shape[0], -1)

    num_groups = x_flat.shape[-1] // group_size
    x_grouped = x_flat.reshape(x_flat.shape[0], num_groups, group_size)
    scale_expanded = scale_flat.unsqueeze(-1)

    x_dequant = x_grouped * scale_expanded
    return x_dequant.reshape(orig_shape)


# ============================================================
# Batch-Invariant MXFP8 MatMul (Reference Implementation)
# ============================================================

def mxfp8_matmul_batch_invariant(
    x_bf16: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    group_size: int = 32,
) -> torch.Tensor:
    """
    Batch-invariant MXFP8 matmul reference implementation.

    Key design: each row is processed identically regardless of batch size.
    All reductions happen in FP32 with fixed order.

    Args:
        x_bf16: activation [M, K] in BF16
        w_fp8: weight [K, N] in FP8
        w_scale: weight scales [K // group_size, N]
        group_size: microscaling group size

    Returns:
        output [M, N] in BF16
    """
    M, K = x_bf16.shape
    K2, N = w_fp8.shape
    assert K == K2

    # Step 1: Quantize activation (per-row, independent of batch)
    x_fp8, x_scale = dynamic_mx_quant(x_bf16, group_size)

    # Step 2: Dequantize both operands to FP32
    x_f32 = dequantize_mx(x_fp8, x_scale, group_size)
    w_f32 = dequantize_mx(w_fp8, w_scale, group_size)

    # Step 3: FP32 matmul (accumulation in FP32, order fixed)
    output = torch.matmul(x_f32, w_f32)

    # Step 4: Cast back to BF16
    return output.to(torch.bfloat16)


# ============================================================
# Test Cases
# ============================================================

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_result(name, passed):
    status = f"{Colors.GREEN}PASS{Colors.END}" if passed else f"{Colors.RED}FAIL{Colors.END}"
    print(f"  [{status}] {name}")
    return passed


def test_fp8_dtype_support():
    """Test 1: Verify FP8 dtype is available and works on CPU."""
    name = "FP8 dtype support on CPU"
    try:
        t = torch.randn(4, 4).to(torch.float8_e4m3fn)
        t_back = t.to(torch.float32)
        ok = t.dtype == torch.float8_e4m3fn and t_back.shape == (4, 4)
        return print_result(name, ok)
    except Exception as e:
        print(f"  [{Colors.RED}FAIL{Colors.END}] {name}: {e}")
        return False


def test_mx_scale_deterministic():
    """Test 2: Verify scale computation is deterministic."""
    name = "MX scale computation deterministic"
    torch.manual_seed(42)
    x = torch.randn(8, 256, dtype=torch.float32)

    s1 = compute_mx_scale(x, group_size=32)
    s2 = compute_mx_scale(x, group_size=32)

    passed = torch.equal(s1, s2)
    return print_result(name, passed)


def test_mx_scale_power_of_2():
    """Test 3: Verify scales are powers of 2 (E8M0FNU format)."""
    name = "MX scales are power-of-2"
    torch.manual_seed(42)
    x = torch.randn(4, 128, dtype=torch.float32)
    scale = compute_mx_scale(x, group_size=32)

    # log2(scale) should be integer
    log2_scale = torch.log2(scale)
    is_integer = torch.all(log2_scale == log2_scale.floor())

    return print_result(name, is_integer.item())


def test_quant_dequant_roundtrip():
    """Test 4: Verify quantize -> dequantize preserves approximate values."""
    name = "Quant-dequant roundtrip (approx)"
    torch.manual_seed(42)
    x = torch.randn(4, 128, dtype=torch.bfloat16)

    x_fp8, scale = dynamic_mx_quant(x, group_size=32)
    x_recon = dequantize_mx(x_fp8, scale, group_size=32)

    # Should be close but not exact due to FP8 quantization
    rel_error = (x.float() - x_recon).abs().mean() / x.float().abs().mean()
    passed = rel_error < 0.1  # <10% relative error

    if not passed:
        print(f"    relative error: {rel_error:.4f}")
    return print_result(name + f" (rel_err={rel_error:.4f})", passed)


def test_batch_invariance_matmul():
    """Test 5 (CORE): Verify matmul output is identical across different batch sizes."""
    name = "MXFP8 matmul batch-invariance (bitwise equal)"
    torch.manual_seed(42)

    K, N = 256, 128
    group_size = 32

    # Fixed weight (FP8 + scale)
    w_bf16 = torch.randn(K, N, dtype=torch.bfloat16)
    w_fp8 = w_bf16.to(torch.float8_e4m3fn)
    w_scale = compute_mx_scale(w_bf16.float().T, group_size).T  # [K//gs, N]
    # Reshape scale to match weight layout
    w_scale = compute_mx_scale(w_bf16.float(), group_size)  # [K, N//gs] -> not right
    # Actually for weight [K, N], scale should be along K dim grouped
    w_grouped = w_bf16.float().reshape(K // group_size, group_size, N)
    w_scale = w_grouped.abs().amax(dim=1)  # [K//gs, N]
    w_scale = torch.pow(2.0, torch.floor(torch.log2(w_scale.clamp(min=1e-12))))

    # Create a single token
    token = torch.randn(1, K, dtype=torch.bfloat16)

    # Test with different batch sizes containing the SAME token
    batch_sizes = [1, 2, 4, 8, 16, 32]
    outputs = []

    for bs in batch_sizes:
        batch = token.expand(bs, -1).contiguous()
        out = mxfp8_matmul_batch_invariant(batch, w_fp8, w_scale, group_size)
        outputs.append(out[0])  # Take first row

    # All outputs must be BITWISE identical
    all_equal = all(torch.equal(outputs[0], o) for o in outputs[1:])

    if not all_equal:
        for i, o in enumerate(outputs):
            diff = (outputs[0].float() - o.float()).abs().max()
            if diff > 0:
                print(f"    batch_size={batch_sizes[i]}: max_diff={diff:.6e}")

    return print_result(name, all_equal)


def test_batch_invariance_scale_computation():
    """Test 6: Verify scale computation is batch-invariant."""
    name = "MX scale computation batch-invariance"
    torch.manual_seed(42)

    K = 256
    group_size = 32
    token = torch.randn(1, K, dtype=torch.bfloat16)

    scales = []
    for bs in [1, 4, 16, 64]:
        batch = token.expand(bs, -1).contiguous()
        _, scale = dynamic_mx_quant(batch, group_size)
        scales.append(scale[0])  # First row's scale

    all_equal = all(torch.equal(scales[0], s) for s in scales[1:])
    return print_result(name, all_equal)


def test_batch_invariance_quantization():
    """Test 7: Verify quantization output is batch-invariant."""
    name = "MX quantization batch-invariance"
    torch.manual_seed(42)

    K = 256
    group_size = 32
    token = torch.randn(1, K, dtype=torch.bfloat16)

    quantized_tokens = []
    for bs in [1, 4, 16, 64]:
        batch = token.expand(bs, -1).contiguous()
        x_fp8, _ = dynamic_mx_quant(batch, group_size)
        quantized_tokens.append(x_fp8[0])  # First row

    all_equal = all(
        torch.equal(quantized_tokens[0].to(torch.float32), q.to(torch.float32))
        for q in quantized_tokens[1:]
    )
    return print_result(name, all_equal)


def test_different_tokens_different_output():
    """Test 8: Sanity check - different tokens should produce different outputs."""
    name = "Different tokens produce different outputs (sanity)"
    torch.manual_seed(42)

    K, N = 256, 128
    group_size = 32

    w_bf16 = torch.randn(K, N, dtype=torch.bfloat16)
    w_fp8 = w_bf16.to(torch.float8_e4m3fn)
    w_grouped = w_bf16.float().reshape(K // group_size, group_size, N)
    w_scale = w_grouped.abs().amax(dim=1)
    w_scale = torch.pow(2.0, torch.floor(torch.log2(w_scale.clamp(min=1e-12))))

    t1 = torch.randn(1, K, dtype=torch.bfloat16)
    t2 = torch.randn(1, K, dtype=torch.bfloat16)

    out1 = mxfp8_matmul_batch_invariant(t1, w_fp8, w_scale, group_size)
    out2 = mxfp8_matmul_batch_invariant(t2, w_fp8, w_scale, group_size)

    passed = not torch.equal(out1, out2)
    return print_result(name, passed)


def test_mixed_batch_invariance():
    """Test 9: In a mixed batch, each token's output matches its standalone result."""
    name = "Mixed-batch invariance (each token matches standalone)"
    torch.manual_seed(42)

    K, N = 256, 128
    group_size = 32

    w_bf16 = torch.randn(K, N, dtype=torch.bfloat16)
    w_fp8 = w_bf16.to(torch.float8_e4m3fn)
    w_grouped = w_bf16.float().reshape(K // group_size, group_size, N)
    w_scale = w_grouped.abs().amax(dim=1)
    w_scale = torch.pow(2.0, torch.floor(torch.log2(w_scale.clamp(min=1e-12))))

    # Create 4 different tokens
    tokens = [torch.randn(1, K, dtype=torch.bfloat16) for _ in range(4)]

    # Standalone results
    standalone = [mxfp8_matmul_batch_invariant(t, w_fp8, w_scale, group_size) for t in tokens]

    # Batched result
    batch = torch.cat(tokens, dim=0)  # [4, K]
    batched = mxfp8_matmul_batch_invariant(batch, w_fp8, w_scale, group_size)

    all_equal = all(
        torch.equal(standalone[i], batched[i:i+1])
        for i in range(4)
    )

    if not all_equal:
        for i in range(4):
            diff = (standalone[i].float() - batched[i:i+1].float()).abs().max()
            if diff > 0:
                print(f"    token {i}: max_diff={diff:.6e}")

    return print_result(name, all_equal)


def test_large_scale():
    """Test 10: Batch-invariance at realistic LLM dimensions."""
    name = "Large-scale batch-invariance (hidden=4096, out=11008)"
    torch.manual_seed(42)

    K, N = 4096, 11008  # Typical LLaMA FFN dimensions
    group_size = 32

    w_bf16 = torch.randn(K, N, dtype=torch.bfloat16)
    w_fp8 = w_bf16.to(torch.float8_e4m3fn)
    w_grouped = w_bf16.float().reshape(K // group_size, group_size, N)
    w_scale = w_grouped.abs().amax(dim=1)
    w_scale = torch.pow(2.0, torch.floor(torch.log2(w_scale.clamp(min=1e-12))))

    token = torch.randn(1, K, dtype=torch.bfloat16)

    out_bs1 = mxfp8_matmul_batch_invariant(
        token, w_fp8, w_scale, group_size
    )
    out_bs8 = mxfp8_matmul_batch_invariant(
        token.expand(8, -1).contiguous(), w_fp8, w_scale, group_size
    )

    passed = torch.equal(out_bs1[0], out_bs8[0])

    if not passed:
        diff = (out_bs1[0].float() - out_bs8[0].float()).abs().max()
        print(f"    max_diff={diff:.6e}")

    return print_result(name, passed)


# ============================================================
# Main
# ============================================================

def main():
    print(f"\n{Colors.BOLD}{'='*60}")
    print("MXFP8 Batch-Invariant Operator Test Suite")
    print(f"{'='*60}{Colors.END}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  Device: CPU (software emulation)")
    print(f"  FP8 support: {hasattr(torch, 'float8_e4m3fn')}")
    print()

    if not hasattr(torch, 'float8_e4m3fn'):
        print(f"{Colors.RED}ERROR: PyTorch >= 2.1 required for FP8 dtype support{Colors.END}")
        sys.exit(1)

    tests = [
        ("Basic FP8 Support", [
            test_fp8_dtype_support,
        ]),
        ("Scale & Quantization Correctness", [
            test_mx_scale_deterministic,
            test_mx_scale_power_of_2,
            test_quant_dequant_roundtrip,
        ]),
        ("Batch-Invariance Properties", [
            test_batch_invariance_scale_computation,
            test_batch_invariance_quantization,
            test_batch_invariance_matmul,
            test_mixed_batch_invariance,
        ]),
        ("Sanity Checks", [
            test_different_tokens_different_output,
        ]),
        ("Large Scale", [
            test_large_scale,
        ]),
    ]

    total = 0
    passed = 0

    for group_name, test_fns in tests:
        print(f"\n{Colors.BOLD}--- {group_name} ---{Colors.END}")
        for fn in test_fns:
            total += 1
            if fn():
                passed += 1

    print(f"\n{Colors.BOLD}{'='*60}")
    color = Colors.GREEN if passed == total else Colors.RED
    print(f"Results: {color}{passed}/{total} passed{Colors.END}")
    print(f"{'='*60}{Colors.END}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
