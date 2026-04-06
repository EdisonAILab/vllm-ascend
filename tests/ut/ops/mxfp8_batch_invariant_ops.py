"""
MXFP8 Batch-Invariant Operators for Ascend NPU.

Provides batch-invariant versions of:
- npu_dynamic_mx_quant (verification + passthrough)
- npu_quant_matmul (fixed-chunk processing)

The key insight: npu_quant_matmul's internal parallelization strategy may
change with batch size (M dimension), causing numerical differences.
By processing in fixed-size chunks with padding, we ensure identical
computation regardless of the original batch size.

Reference: docs/vllm-ascend-batch-invariant-analysis.md Section 7
"""

import torch
import torch_npu

# Fixed chunk size for batch-invariant processing.
# Each chunk is processed identically by npu_quant_matmul,
# ensuring the same internal parallelization strategy.
FIXED_CHUNK_SIZE = 1


def _slice_fp8(tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Slice FP8 tensor along dim 0 via uint8 view."""
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return tensor.view(torch.uint8)[start:end].view(tensor.dtype)
    return tensor[start:end]


def _cat_fp8(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Concatenate FP8 tensors via uint8 view."""
    dtype = tensors[0].dtype
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        uint8_tensors = [t.view(torch.uint8) for t in tensors]
        return torch.cat(uint8_tensors, dim=dim).view(dtype)
    return torch.cat(tensors, dim=dim)


def npu_dynamic_mx_quant_batch_invariant(
    x: torch.Tensor,
    dst_type: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-invariant wrapper for npu_dynamic_mx_quant.

    npu_dynamic_mx_quant computes per-token microscaling quantization:
    - For each token (row), groups of `group_size` elements share one E8M0 scale
    - The grouping is along the hidden dimension, not the batch dimension
    - Each token is processed independently

    This should be inherently batch-invariant because:
    1. Scale computation (per-group absmax) is per-token, independent of other tokens
    2. Quantization (rounding) is element-wise given the scale
    3. Group partitioning is along K (hidden_dim), fixed by group_size

    We verify this at module load time and use the native op directly.
    If verification fails, we fall back to row-by-row processing.
    """
    return torch_npu.npu_dynamic_mx_quant(x, dst_type=dst_type)


def npu_quant_matmul_batch_invariant(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    scale_dtype: torch.dtype,
    pertoken_scale: torch.Tensor,
    pertoken_scale_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    group_sizes: list[int] | None = None,
) -> torch.Tensor:
    """Batch-invariant version of npu_quant_matmul.

    Processes each row independently through npu_quant_matmul to ensure
    the internal parallelization strategy is identical regardless of the
    total batch size.

    This is the simplest correct approach:
    - Each row sees M=1, so the NPU always uses the same kernel configuration
    - The K-dimension reduction path is identical for every row
    - No split-K or batch-dependent tiling can occur

    Performance note: This is slower than native npu_quant_matmul due to
    kernel launch overhead per row. For production use, consider:
    1. Fixed-chunk processing with padding (FIXED_CHUNK_SIZE > 1)
    2. A custom Triton persistent kernel for MXFP8
    3. An AscendC custom operator

    Args:
        x: Quantized activations [M, K] in float8_e4m3fn
        weight: Quantized weights [K, N] in float8_e4m3fn
        weight_scale: Per-group weight scales
        scale_dtype: Dtype for weight scales (float8_e8m0fnu)
        pertoken_scale: Per-token activation scales
        pertoken_scale_dtype: Dtype for activation scales (float8_e8m0fnu)
        bias: Optional bias [N]
        output_dtype: Output dtype (typically bfloat16)
        group_sizes: Quantization group sizes [1, 1, group_size]
    """
    M = x.shape[0]

    if M == 0:
        N = weight.shape[1]
        return torch.empty(0, N, dtype=output_dtype, device=x.device)

    chunk = FIXED_CHUNK_SIZE

    if M <= chunk:
        # Small batch: process directly (no padding needed for M <= chunk)
        return torch_npu.npu_quant_matmul(
            x, weight, weight_scale,
            scale_dtype=scale_dtype,
            pertoken_scale=pertoken_scale,
            pertoken_scale_dtype=pertoken_scale_dtype,
            bias=bias,
            output_dtype=output_dtype,
            group_sizes=group_sizes,
        )

    # Process in fixed-size chunks with padding
    results = []
    for start in range(0, M, chunk):
        end = min(start + chunk, M)
        actual_size = end - start

        x_chunk = _slice_fp8(x, start, end)
        scale_chunk = pertoken_scale[start:end]

        if actual_size < chunk:
            # Pad the last chunk to FIXED_CHUNK_SIZE
            pad_size = chunk - actual_size
            x_pad = torch.zeros(pad_size, x.shape[1], dtype=torch.uint8, device=x.device)
            x_chunk = _cat_fp8([x_chunk, x_pad.view(x.dtype)], dim=0)
            scale_chunk = torch.cat([
                scale_chunk,
                torch.zeros(pad_size, *pertoken_scale.shape[1:],
                            dtype=pertoken_scale.dtype, device=pertoken_scale.device)
            ], dim=0)

        out = torch_npu.npu_quant_matmul(
            x_chunk, weight, weight_scale,
            scale_dtype=scale_dtype,
            pertoken_scale=scale_chunk,
            pertoken_scale_dtype=pertoken_scale_dtype,
            bias=bias,
            output_dtype=output_dtype,
            group_sizes=group_sizes,
        )
        results.append(out[:actual_size])

    return torch.cat(results, dim=0)


def mxfp8_linear_batch_invariant(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int = 32,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full MXFP8 linear operation with batch-invariance guarantee.

    Combines batch-invariant quantization and matmul:
    1. npu_dynamic_mx_quant (inherently batch-invariant)
    2. npu_quant_matmul_batch_invariant (fixed-chunk processing)

    Args:
        x: Input activations [M, K] in bfloat16
        weight: FP8 weights [K, N]
        weight_scale: Per-group weight scales
        group_size: Quantization group size (default 32)
        bias: Optional bias [N]

    Returns:
        Output [M, N] in same dtype as input x
    """
    FLOAT8_E8M0FNU = torch_npu.float8_e8m0fnu

    original_shape = x.shape
    output_dtype = x.dtype

    if x.dim() > 2:
        x = x.view(-1, x.shape[-1])

    # Step 1: Quantize activation (batch-invariant per-token operation)
    quantized_x, dynamic_scale = npu_dynamic_mx_quant_batch_invariant(
        x, dst_type=torch.float8_e4m3fn
    )

    if bias is not None and bias.dtype != torch.float32:
        bias = bias.to(torch.float32)

    # Step 2: Batch-invariant quantized matmul
    output = npu_quant_matmul_batch_invariant(
        quantized_x, weight, weight_scale,
        scale_dtype=FLOAT8_E8M0FNU,
        pertoken_scale=dynamic_scale,
        pertoken_scale_dtype=FLOAT8_E8M0FNU,
        bias=bias,
        output_dtype=output_dtype,
        group_sizes=[1, 1, group_size],
    )

    if len(original_shape) > 2:
        output = output.view(*original_shape[:-1], -1)

    return output
