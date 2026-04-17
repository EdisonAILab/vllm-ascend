# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Batch-invariant rotary embedding (RoPE) Triton kernel for Ascend NPU.

Approach: reshape [n_tokens, n_heads * head_dim] to [n_tokens * n_heads, head_dim],
apply RoPE per-row (each row = one head of one token), reshape back.
This avoids dynamic per-head loops that may cause compiler issues.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _rope_kernel(
    x_ptr,             # [n_rows, head_dim]  (flattened: n_tokens * n_heads rows)
    cos_ptr,           # [n_rows, half_dim]  (broadcast-expanded: same cos for all heads)
    sin_ptr,           # [n_rows, half_dim]
    x_stride_row,
    cs_stride_row,
    n_rows,
    half_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply NEOX-style RoPE per row.
    Row layout: [a_0..a_{half-1}, b_0..b_{half-1}]
    cos/sin layout: [cos_a_0..cos_a_{half-1}, cos_b_0..cos_b_{half-1}]
    Output: [a*cos_a - b*sin_a, b*cos_b + a*sin_b]
    """
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    rows_per_program = (n_rows + n_programs - 1) // n_programs
    start_row = pid * rows_per_program
    end_row = tl.minimum(start_row + rows_per_program, n_rows)

    for row_idx in range(start_row, end_row):
        x_row = x_ptr + row_idx * x_stride_row
        cs_base = row_idx * cs_stride_row

        for col_offset in range(0, half_dim, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < half_dim

            # Load first half (a) and second half (b)
            a = tl.load(x_row + col_idx, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(x_row + half_dim + col_idx, mask=mask, other=0.0).to(tl.float32)

            # cos/sin for first half and second half are DIFFERENT
            cos_a = tl.load(cos_ptr + cs_base + col_idx,
                            mask=mask, other=0.0).to(tl.float32)
            sin_a = tl.load(sin_ptr + cs_base + col_idx,
                            mask=mask, other=0.0).to(tl.float32)
            cos_b = tl.load(cos_ptr + cs_base + half_dim + col_idx,
                            mask=mask, other=0.0).to(tl.float32)
            sin_b = tl.load(sin_ptr + cs_base + half_dim + col_idx,
                            mask=mask, other=0.0).to(tl.float32)

            # NEOX: out_a = a*cos_a - b*sin_a,  out_b = b*cos_b + a*sin_b
            out_a = a * cos_a - b * sin_a
            out_b = b * cos_b + a * sin_b

            tl.store(x_row + col_idx,
                     out_a.to(x_ptr.dtype.element_ty), mask=mask)
            tl.store(x_row + half_dim + col_idx,
                     out_b.to(x_ptr.dtype.element_ty), mask=mask)


def rotary_embedding(
    q: torch.Tensor,    # [n_tokens, n_q_heads * head_dim]
    k: torch.Tensor,    # [n_tokens, n_k_heads * head_dim]
    cos: torch.Tensor,  # [n_tokens, head_dim]
    sin: torch.Tensor,  # [n_tokens, head_dim]
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE in-place to Q and K."""
    n_tokens = q.shape[0]
    n_q_heads = q.shape[1] // head_dim
    n_k_heads = k.shape[1] // head_dim
    half_dim = head_dim // 2

    def apply_rope(x, n_heads):
        x = x.contiguous()
        # Reshape to [n_tokens * n_heads, head_dim]
        x_flat = x.view(n_tokens * n_heads, head_dim)
        # Expand cos/sin: [n_tokens, head_dim] -> [n_tokens, 1, head_dim] -> [n_tokens, n_heads, head_dim]
        # -> [n_tokens * n_heads, head_dim]
        cos_exp = cos.unsqueeze(1).expand(-1, n_heads, -1).contiguous().view(-1, head_dim)
        sin_exp = sin.unsqueeze(1).expand(-1, n_heads, -1).contiguous().view(-1, head_dim)

        n_rows = n_tokens * n_heads
        BLOCK_SIZE = 64
        max_grid = triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )["num_vectorcore"]
        grid = (min(n_rows, max_grid),)

        _rope_kernel[grid](
            x_flat, cos_exp, sin_exp,
            x_flat.stride(0), cos_exp.stride(0),  # cs_stride = head_dim
            n_rows, half_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        # Note: cos_exp.stride(0) = head_dim since shape is [n_rows, head_dim]
        return x_flat.view(n_tokens, n_heads * head_dim)

    q_out = apply_rope(q, n_q_heads)
    k_out = apply_rope(k, n_k_heads)
    return q_out, k_out


def rotary_embedding_batch_invariant(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-invariant RoPE wrapper."""
    return rotary_embedding(q, k, cos, sin, head_dim)
