# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Batch-invariant rotary embedding (RoPE) Triton kernel for Ascend NPU.

Standard NEOX-style RoPE:
  rotate_half([a, b]) = [-b, a]    (where a, b are halves of head_dim)
  out = x * cos + rotate_half(x) * sin

Per-token operation, no cross-batch reduction → naturally batch-invariant.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _rope_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,           # [n_tokens, head_dim] (precomputed for each position)
    sin_ptr,           # [n_tokens, head_dim]
    q_stride_token,
    k_stride_token,
    cs_stride_token,
    n_tokens,
    n_q_heads,
    n_k_heads,
    head_dim,
    half_dim,          # head_dim // 2
    BLOCK_SIZE: tl.constexpr,
):
    """Apply RoPE per token. Each program handles multiple tokens."""
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    tokens_per_program = (n_tokens + n_programs - 1) // n_programs
    start_tok = pid * tokens_per_program
    end_tok = tl.minimum(start_tok + tokens_per_program, n_tokens)

    for tok_idx in range(start_tok, end_tok):
        q_tok_ptr = q_ptr + tok_idx * q_stride_token
        k_tok_ptr = k_ptr + tok_idx * k_stride_token
        cs_tok_ptr_base = tok_idx * cs_stride_token

        # Iterate over heads and positions within head_dim
        # Process Q heads
        for h in range(n_q_heads):
            head_offset = h * head_dim
            for col_offset in range(0, half_dim, BLOCK_SIZE):
                col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
                mask = col_idx < half_dim

                # Load first half (a) and second half (b)
                a = tl.load(q_tok_ptr + head_offset + col_idx,
                            mask=mask, other=0.0).to(tl.float32)
                b = tl.load(q_tok_ptr + head_offset + half_dim + col_idx,
                            mask=mask, other=0.0).to(tl.float32)

                # Load cos/sin for first half and second half
                cos_a = tl.load(cos_ptr + cs_tok_ptr_base + col_idx,
                                mask=mask, other=0.0).to(tl.float32)
                cos_b = tl.load(cos_ptr + cs_tok_ptr_base + half_dim + col_idx,
                                mask=mask, other=0.0).to(tl.float32)
                sin_a = tl.load(sin_ptr + cs_tok_ptr_base + col_idx,
                                mask=mask, other=0.0).to(tl.float32)
                sin_b = tl.load(sin_ptr + cs_tok_ptr_base + half_dim + col_idx,
                                mask=mask, other=0.0).to(tl.float32)

                # NEOX rotate_half: [-b, a]
                # Output first half: a * cos_a - b * sin_a
                # Output second half: b * cos_b + a * sin_b
                out_a = a * cos_a - b * sin_a
                out_b = b * cos_b + a * sin_b

                tl.store(q_tok_ptr + head_offset + col_idx,
                         out_a.to(q_ptr.dtype.element_ty), mask=mask)
                tl.store(q_tok_ptr + head_offset + half_dim + col_idx,
                         out_b.to(q_ptr.dtype.element_ty), mask=mask)

        # Process K heads (same logic)
        for h in range(n_k_heads):
            head_offset = h * head_dim
            for col_offset in range(0, half_dim, BLOCK_SIZE):
                col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
                mask = col_idx < half_dim

                a = tl.load(k_tok_ptr + head_offset + col_idx,
                            mask=mask, other=0.0).to(tl.float32)
                b = tl.load(k_tok_ptr + head_offset + half_dim + col_idx,
                            mask=mask, other=0.0).to(tl.float32)

                cos_a = tl.load(cos_ptr + cs_tok_ptr_base + col_idx,
                                mask=mask, other=0.0).to(tl.float32)
                cos_b = tl.load(cos_ptr + cs_tok_ptr_base + half_dim + col_idx,
                                mask=mask, other=0.0).to(tl.float32)
                sin_a = tl.load(sin_ptr + cs_tok_ptr_base + col_idx,
                                mask=mask, other=0.0).to(tl.float32)
                sin_b = tl.load(sin_ptr + cs_tok_ptr_base + half_dim + col_idx,
                                mask=mask, other=0.0).to(tl.float32)

                out_a = a * cos_a - b * sin_a
                out_b = b * cos_b + a * sin_b

                tl.store(k_tok_ptr + head_offset + col_idx,
                         out_a.to(k_ptr.dtype.element_ty), mask=mask)
                tl.store(k_tok_ptr + head_offset + half_dim + col_idx,
                         out_b.to(k_ptr.dtype.element_ty), mask=mask)


def rotary_embedding(
    q: torch.Tensor,    # [n_tokens, n_q_heads * head_dim] (in-place modified)
    k: torch.Tensor,    # [n_tokens, n_k_heads * head_dim]
    cos: torch.Tensor,  # [n_tokens, head_dim]
    sin: torch.Tensor,  # [n_tokens, head_dim]
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE in-place to Q and K, returning the modified tensors."""
    assert q.dim() == 2 and k.dim() == 2
    n_tokens = q.shape[0]
    n_q_heads = q.shape[1] // head_dim
    n_k_heads = k.shape[1] // head_dim
    half_dim = head_dim // 2

    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    BLOCK_SIZE = 64
    max_grid = triton.runtime.driver.active.utils.get_device_properties(
        torch.npu.current_device()
    )["num_vectorcore"]
    grid = (min(n_tokens, max_grid),)

    _rope_kernel[grid](
        q, k, cos, sin,
        q.stride(0), k.stride(0), cos.stride(0),
        n_tokens, n_q_heads, n_k_heads,
        head_dim, half_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return q, k


def rotary_embedding_batch_invariant(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-invariant RoPE wrapper. Modifies q and k in place."""
    return rotary_embedding(q, k, cos, sin, head_dim)
