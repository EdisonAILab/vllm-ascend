# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Batch-invariant MoE gating: hidden → gate Linear → softmax → topk.

Composes BI matmul (existing) with BI topk_softmax (new Triton kernel).
"""

import torch

from vllm_ascend.ops.triton.batch_invariant.matmul import linear_persistent
from vllm_ascend.ops.triton.batch_invariant.topk_softmax import topk_softmax_batch_invariant


def moe_gating(
    hidden_states: torch.Tensor,    # [n_tokens, hidden_size]
    gate_weight: torch.Tensor,      # [n_experts, hidden_size]
    top_k: int,
):
    """MoE gating: linear (BI matmul) + topk_softmax (BI Triton).

    Returns:
        weights: [n_tokens, top_k] float32
        indices: [n_tokens, top_k] int32
    """
    # BI Linear: hidden_states @ gate_weight.T using existing persistent kernel
    logits = linear_persistent(hidden_states, gate_weight)
    # BI topk-softmax
    return topk_softmax_batch_invariant(logits, top_k)


def moe_gating_batch_invariant(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
):
    """Batch-invariant wrapper for MoE gating."""
    return moe_gating(hidden_states, gate_weight, top_k)
