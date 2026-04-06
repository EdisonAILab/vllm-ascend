"""
Test MXFP8 batch-invariance on Ascend NPU.

This script verifies whether npu_dynamic_mx_quant and npu_quant_matmul
produce identical results for shared rows regardless of batch size.

Usage:
    python test_mxfp8_batch_invariant.py
"""

import torch
import torch_npu

DEVICE = "npu"
DTYPE = torch.bfloat16
GROUP_SIZE = 32
FLOAT8_E8M0FNU_DTYPE = torch_npu.float8_e8m0fnu


def cat_fp8(tensors, dim=0):
    """Concatenate FP8 tensors by viewing as uint8."""
    dtype = tensors[0].dtype
    uint8_tensors = [t.view(torch.uint8) for t in tensors]
    result = torch.cat(uint8_tensors, dim=dim)
    return result.view(dtype)


def cat_any(tensors, dim=0):
    """Concatenate tensors, handling FP8 and other special dtypes."""
    try:
        return torch.cat(tensors, dim=dim)
    except RuntimeError:
        return cat_fp8(tensors, dim=dim)


def equal_any(a, b):
    """Compare tensors, handling FP8 dtypes via uint8 view."""
    try:
        return torch.equal(a, b)
    except RuntimeError:
        return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def slice_fp8(tensor, start, end):
    """Slice FP8 tensor along dim 0 via uint8 view."""
    return tensor.view(torch.uint8)[start:end].view(tensor.dtype)


def make_bf16_input(M, K, seed=42):
    """Create reproducible BF16 input."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(M, K, generator=gen, dtype=DTYPE).to(DEVICE)


def make_fp8_weight(K, N, seed=123):
    """Create fake FP8 weight and scale for testing.

    npu_dynamic_mx_quant returns:
        w_fp8: [N, K] float8_e4m3fn
        w_scale: [N, K//GROUP_SIZE//2, 2] uint8 (E8M0 microscaling format)

    npu_quant_matmul expects (after process_weights_after_loading):
        weight: [K, N] float8_e4m3fn (transposed)
        weight_scale: [K//GROUP_SIZE//2, N, 2] uint8 (transposed)
    """
    gen = torch.Generator().manual_seed(seed)
    w_bf16 = torch.randn(N, K, generator=gen, dtype=DTYPE).to(DEVICE)
    w_fp8, w_scale = torch_npu.npu_dynamic_mx_quant(w_bf16, dst_type=torch.float8_e4m3fn)
    # Transpose: (N, K) -> (K, N) for weight
    w_fp8_t = w_fp8.view(torch.uint8).transpose(0, 1).contiguous().view(torch.float8_e4m3fn)
    # Transpose scale: (N, K//GS//2, 2) -> (K//GS//2, N, 2)
    w_scale_t = w_scale.transpose(0, 1).contiguous()
    return w_fp8_t, w_scale_t


# ============================================================
# Test 1: npu_dynamic_mx_quant batch-invariance
# ============================================================
def test_dynamic_mx_quant_batch_invariance():
    """
    Check if npu_dynamic_mx_quant gives identical results for the same row
    regardless of how many other rows are in the batch.
    """
    print("=" * 60)
    print("Test 1: npu_dynamic_mx_quant batch-invariance")
    print("=" * 60)

    K = 4096

    # Create a large input with known seed
    x_large = make_bf16_input(64, K, seed=42)

    # Process as full batch
    quant_full, scale_full = torch_npu.npu_dynamic_mx_quant(
        x_large, dst_type=torch.float8_e4m3fn
    )

    # Process row-by-row
    quant_rows = []
    scale_rows = []
    for i in range(x_large.shape[0]):
        q, s = torch_npu.npu_dynamic_mx_quant(
            x_large[i:i+1], dst_type=torch.float8_e4m3fn
        )
        quant_rows.append(q)
        scale_rows.append(s)
    quant_single = cat_any(quant_rows, dim=0)
    scale_single = cat_any(scale_rows, dim=0)

    # Process in various batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    all_match = True

    for bs in batch_sizes:
        quant_chunks = []
        scale_chunks = []
        for start in range(0, 64, bs):
            end = min(start + bs, 64)
            q, s = torch_npu.npu_dynamic_mx_quant(
                x_large[start:end], dst_type=torch.float8_e4m3fn
            )
            quant_chunks.append(q)
            scale_chunks.append(s)
        quant_bs = cat_any(quant_chunks, dim=0)
        scale_bs = cat_any(scale_chunks, dim=0)

        quant_match = equal_any(quant_full, quant_bs)
        scale_match = equal_any(scale_full, scale_bs)

        if not quant_match or not scale_match:
            all_match = False
            # Count differences
            quant_diff = (quant_full.view(torch.uint8) != quant_bs.view(torch.uint8)).sum().item()
            scale_diff = (scale_full.view(torch.uint8) != scale_bs.view(torch.uint8)).sum().item()
            print(f"  BS={bs:3d}: quant_match={quant_match}, scale_match={scale_match}, "
                  f"quant_diff_count={quant_diff}, scale_diff_count={scale_diff}")
        else:
            print(f"  BS={bs:3d}: MATCH ✓")

    # Also check single-row vs full-batch
    quant_match_single = equal_any(quant_full, quant_single)
    scale_match_single = equal_any(scale_full, scale_single)
    print(f"  Single-row vs full: quant={quant_match_single}, scale={scale_match_single}")

    result = "PASS (batch-invariant)" if all_match else "FAIL (NOT batch-invariant)"
    print(f"\n  Result: {result}\n")
    return all_match


# ============================================================
# Test 2: npu_quant_matmul batch-invariance
# ============================================================
def test_quant_matmul_batch_invariance():
    """
    Check if npu_quant_matmul gives identical results for the same row
    regardless of batch size.
    """
    print("=" * 60)
    print("Test 2: npu_quant_matmul batch-invariance")
    print("=" * 60)

    K, N = 4096, 2048

    # Create input and quantize
    x_large = make_bf16_input(64, K, seed=42)
    quant_x, scale_x = torch_npu.npu_dynamic_mx_quant(
        x_large, dst_type=torch.float8_e4m3fn
    )

    # Create weight
    w_fp8, w_scale = make_fp8_weight(K, N, seed=123)

    # Process full batch
    output_full = torch_npu.npu_quant_matmul(
        quant_x, w_fp8, w_scale,
        scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        pertoken_scale=scale_x,
        pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        output_dtype=DTYPE,
        group_sizes=[1, 1, GROUP_SIZE],
    )

    # Process in various batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32]
    all_match = True

    for bs in batch_sizes:
        output_chunks = []
        for start in range(0, 64, bs):
            end = min(start + bs, 64)
            out = torch_npu.npu_quant_matmul(
                slice_fp8(quant_x, start, end), w_fp8, w_scale,
                scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                pertoken_scale=scale_x[start:end],
                pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                output_dtype=DTYPE,
                group_sizes=[1, 1, GROUP_SIZE],
            )
            output_chunks.append(out)
        output_bs = torch.cat(output_chunks, dim=0)

        match = torch.equal(output_full, output_bs)
        if not match:
            all_match = False
            diff = (output_full != output_bs)
            max_diff = (output_full - output_bs).abs().max().item()
            diff_count = diff.sum().item()
            total = diff.numel()
            print(f"  BS={bs:3d}: MISMATCH  diff_count={diff_count}/{total} "
                  f"({diff_count/total*100:.2f}%), max_diff={max_diff:.6e}")
        else:
            print(f"  BS={bs:3d}: MATCH ✓")

    result = "PASS (batch-invariant)" if all_match else "FAIL (NOT batch-invariant)"
    print(f"\n  Result: {result}\n")
    return all_match


# ============================================================
# Test 3: Full MXFP8 linear (quant + matmul) batch-invariance
# ============================================================
def test_mxfp8_linear_batch_invariance():
    """
    Check batch-invariance of the full MXFP8 linear pipeline:
    npu_dynamic_mx_quant + npu_quant_matmul
    """
    print("=" * 60)
    print("Test 3: Full MXFP8 linear (quant+matmul) batch-invariance")
    print("=" * 60)

    K, N = 4096, 2048
    x_large = make_bf16_input(64, K, seed=42)
    w_fp8, w_scale = make_fp8_weight(K, N, seed=123)

    def mxfp8_linear(x):
        qx, sx = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        return torch_npu.npu_quant_matmul(
            qx, w_fp8, w_scale,
            scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            pertoken_scale=sx,
            pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            output_dtype=DTYPE,
            group_sizes=[1, 1, GROUP_SIZE],
        )

    # Full batch
    output_full = mxfp8_linear(x_large)

    batch_sizes = [1, 2, 4, 8, 16, 32]
    all_match = True

    for bs in batch_sizes:
        output_chunks = []
        for start in range(0, 64, bs):
            end = min(start + bs, 64)
            output_chunks.append(mxfp8_linear(x_large[start:end]))
        output_bs = torch.cat(output_chunks, dim=0)

        match = torch.equal(output_full, output_bs)
        if not match:
            all_match = False
            diff = (output_full != output_bs)
            max_diff = (output_full - output_bs).abs().max().item()
            diff_count = diff.sum().item()
            total = diff.numel()
            print(f"  BS={bs:3d}: MISMATCH  diff_count={diff_count}/{total} "
                  f"({diff_count/total*100:.2f}%), max_diff={max_diff:.6e}")
        else:
            print(f"  BS={bs:3d}: MATCH ✓")

    result = "PASS (batch-invariant)" if all_match else "FAIL (NOT batch-invariant)"
    print(f"\n  Result: {result}\n")
    return all_match


# ============================================================
# Test 4: Batch-invariant MXFP8 matmul (our implementation)
# ============================================================
def test_batch_invariant_mxfp8_matmul():
    """
    Test our batch-invariant MXFP8 matmul implementation.
    Uses fixed-chunk processing to guarantee batch-invariance.
    """
    print("=" * 60)
    print("Test 4: Batch-invariant MXFP8 quant_matmul (fixed-chunk)")
    print("=" * 60)

    K, N = 4096, 2048
    x_large = make_bf16_input(64, K, seed=42)
    w_fp8, w_scale = make_fp8_weight(K, N, seed=123)

    # Quantize all at once (quant is per-token, should be BI)
    quant_x, scale_x = torch_npu.npu_dynamic_mx_quant(
        x_large, dst_type=torch.float8_e4m3fn
    )

    from mxfp8_batch_invariant_ops import npu_quant_matmul_batch_invariant

    # Full batch through our BI implementation
    output_full = npu_quant_matmul_batch_invariant(
        quant_x, w_fp8, w_scale,
        scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        pertoken_scale=scale_x,
        pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        output_dtype=DTYPE,
        group_sizes=[1, 1, GROUP_SIZE],
    )

    batch_sizes = [1, 2, 4, 8, 16, 32]
    all_match = True

    for bs in batch_sizes:
        output_chunks = []
        for start in range(0, 64, bs):
            end = min(start + bs, 64)
            out = npu_quant_matmul_batch_invariant(
                slice_fp8(quant_x, start, end), w_fp8, w_scale,
                scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                pertoken_scale=scale_x[start:end],
                pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                output_dtype=DTYPE,
                group_sizes=[1, 1, GROUP_SIZE],
            )
            output_chunks.append(out)
        output_bs = torch.cat(output_chunks, dim=0)

        match = torch.equal(output_full, output_bs)
        if not match:
            all_match = False
            diff_count = (output_full != output_bs).sum().item()
            max_diff = (output_full - output_bs).abs().max().item()
            total = output_full.numel()
            print(f"  BS={bs:3d}: MISMATCH  diff_count={diff_count}/{total} "
                  f"({diff_count/total*100:.2f}%), max_diff={max_diff:.6e}")
        else:
            print(f"  BS={bs:3d}: MATCH ✓")

    result = "PASS (batch-invariant)" if all_match else "FAIL (NOT batch-invariant)"
    print(f"\n  Result: {result}\n")
    return all_match


# ============================================================
# Test 5: Compare BI implementation accuracy vs native
# ============================================================
def test_batch_invariant_accuracy():
    """
    Compare output of our batch-invariant implementation vs native npu_quant_matmul.
    """
    print("=" * 60)
    print("Test 5: Batch-invariant accuracy vs native npu_quant_matmul")
    print("=" * 60)

    K, N = 4096, 2048
    x = make_bf16_input(32, K, seed=42)
    w_fp8, w_scale = make_fp8_weight(K, N, seed=123)

    quant_x, scale_x = torch_npu.npu_dynamic_mx_quant(
        x, dst_type=torch.float8_e4m3fn
    )

    # Native
    output_native = torch_npu.npu_quant_matmul(
        quant_x, w_fp8, w_scale,
        scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        pertoken_scale=scale_x,
        pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        output_dtype=DTYPE,
        group_sizes=[1, 1, GROUP_SIZE],
    )

    # Our BI implementation
    from mxfp8_batch_invariant_ops import npu_quant_matmul_batch_invariant
    output_bi = npu_quant_matmul_batch_invariant(
        quant_x, w_fp8, w_scale,
        scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        pertoken_scale=scale_x,
        pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        output_dtype=DTYPE,
        group_sizes=[1, 1, GROUP_SIZE],
    )

    exact_match = torch.equal(output_native, output_bi)
    max_diff = (output_native.float() - output_bi.float()).abs().max().item()
    mean_diff = (output_native.float() - output_bi.float()).abs().mean().item()
    rel_diff = (
        (output_native.float() - output_bi.float()).abs()
        / (output_native.float().abs() + 1e-8)
    ).mean().item()

    print(f"  Exact match: {exact_match}")
    print(f"  Max absolute diff: {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")
    print(f"  Mean relative diff: {rel_diff:.6e}")
    print()
    return exact_match or max_diff < 1e-2


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MXFP8 Batch-Invariance Test Suite")
    print("=" * 60 + "\n")

    results = {}

    # Tests 1-3: Check native operators
    results["dynamic_mx_quant"] = test_dynamic_mx_quant_batch_invariance()
    results["quant_matmul"] = test_quant_matmul_batch_invariance()
    results["mxfp8_linear"] = test_mxfp8_linear_batch_invariance()

    # Tests 4-5: Check our BI implementation (only if needed)
    try:
        results["bi_mxfp8_matmul"] = test_batch_invariant_mxfp8_matmul()
        results["bi_accuracy"] = test_batch_invariant_accuracy()
    except ImportError:
        print("\n  [SKIP] Batch-invariant MXFP8 ops not available.")
        print("  Place mxfp8_batch_invariant_ops.py in the same directory.\n")

    # Summary
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:30s}: {status}")
    print()
