# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Batch-invariant topk-softmax Triton kernel for Ascend NPU.

Used in MoE gating: compute softmax(scores) and select top-k experts per token.

Per-row operation, no cross-batch reduction → batch-invariant.

Note: Top-k via repeated argmax is O(k * num_experts) per row, suitable for
small num_experts (typically 8-256 in MoE).
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _topk_softmax_kernel(
    scores_ptr,           # [n_tokens, n_experts]
    weights_ptr,          # [n_tokens, top_k]   (output)
    indices_ptr,          # [n_tokens, top_k]   (output)
    scores_row_stride,
    weights_row_stride,
    indices_row_stride,
    n_tokens,
    n_experts,
    top_k: tl.constexpr,
    BLOCK_E: tl.constexpr,  # >= n_experts (rounded up to power of 2)
):
    """Per-row softmax + topk. Each program handles multiple rows."""
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    rows_per_program = (n_tokens + n_programs - 1) // n_programs
    start_row = pid * rows_per_program
    end_row = tl.minimum(start_row + rows_per_program, n_tokens)

    expert_idx = tl.arange(0, BLOCK_E)
    expert_mask = expert_idx < n_experts

    for row_idx in range(start_row, end_row):
        row_ptr = scores_ptr + row_idx * scores_row_stride
        w_out_ptr = weights_ptr + row_idx * weights_row_stride
        i_out_ptr = indices_ptr + row_idx * indices_row_stride

        # Load entire row of scores
        scores = tl.load(row_ptr + expert_idx, mask=expert_mask,
                         other=-float("inf")).to(tl.float32)

        # Softmax
        row_max = tl.max(scores)
        exp_scores = tl.exp(scores - row_max)
        # Mask out the padded experts (set to 0 so they don't contribute to sum)
        exp_scores = tl.where(expert_mask, exp_scores, 0.0)
        sum_exp = tl.sum(exp_scores)
        probs = exp_scores / sum_exp  # softmax probabilities

        # Top-k via repeated argmax (deterministic since we always pick the
        # smallest index on ties)
        # We mask out picked experts by setting them to -inf
        remaining = tl.where(expert_mask, probs, -float("inf"))
        for k in range(top_k):
            # Argmax of remaining
            best_idx = tl.argmax(remaining, axis=0)
            best_val = tl.max(remaining)

            # Store
            tl.store(w_out_ptr + k, best_val)
            tl.store(i_out_ptr + k, best_idx.to(tl.int32))

            # Mask out the selected expert: set to -inf
            remaining = tl.where(expert_idx == best_idx,
                                 -float("inf"), remaining)


def topk_softmax(scores: torch.Tensor, top_k: int):
    """Compute softmax(scores) and select top-k per row.

    Args:
        scores: [n_tokens, n_experts]
        top_k: number of experts to select per token
    Returns:
        weights: [n_tokens, top_k] float32
        indices: [n_tokens, top_k] int32
    """
    assert scores.dim() == 2
    n_tokens, n_experts = scores.shape

    scores = scores.contiguous()
    weights = torch.empty(n_tokens, top_k, dtype=torch.float32, device=scores.device)
    indices = torch.empty(n_tokens, top_k, dtype=torch.int32, device=scores.device)

    # BLOCK_E must be a power of 2 >= n_experts
    BLOCK_E = 1
    while BLOCK_E < n_experts:
        BLOCK_E *= 2
    BLOCK_E = max(BLOCK_E, 8)

    max_grid = triton.runtime.driver.active.utils.get_device_properties(
        torch.npu.current_device()
    )["num_vectorcore"]
    grid = (min(n_tokens, max_grid),)

    _topk_softmax_kernel[grid](
        scores, weights, indices,
        scores.stride(0), weights.stride(0), indices.stride(0),
        n_tokens, n_experts,
        top_k=top_k, BLOCK_E=BLOCK_E,
    )

    return weights, indices


def topk_softmax_batch_invariant(scores: torch.Tensor, top_k: int):
    """Batch-invariant wrapper for topk_softmax."""
    return topk_softmax(scores, top_k)
