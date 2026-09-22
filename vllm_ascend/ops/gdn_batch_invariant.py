# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch-invariant Qwen3.5 Gated DeltaNet scan helpers for Ascend A5."""

from collections.abc import Sequence

import torch


GDN_HEAD_DIM = 128
GDN_MAX_TOKENS_PER_LAUNCH = 4096


def is_gdn_scan_batch_invariant_available() -> bool:
    namespace = getattr(torch.ops, "_C_ascend", None)
    return (
        namespace is not None
        and hasattr(namespace, "gdn_scan_batch_invariant")
        and hasattr(namespace, "gdn_scatter_state_batch_invariant")
    )


def gdn_scatter_state_batch_invariant(
    state_cache: torch.Tensor,
    updates: torch.Tensor,
    state_indices: torch.Tensor,
) -> None:
    """Update valid recurrent-state rows and ignore graph-padding indices."""
    torch.ops._C_ascend.gdn_scatter_state_batch_invariant(
        state_cache,
        updates.float().contiguous(),
        state_indices.to(torch.int32).contiguous(),
    )


def _expand_key_heads(tensor: torch.Tensor, value_heads: int) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError("GDN query and key tensors must have shape [batch, tokens, heads, 128].")
    key_heads = tensor.shape[2]
    if key_heads <= 0:
        raise ValueError("GDN query and key tensors must contain at least one head.")
    if value_heads % key_heads != 0:
        raise ValueError(
            "The number of GDN value heads must be divisible by the number of key heads."
        )
    return tensor.repeat_interleave(value_heads // key_heads, dim=2)


def _run_flat_scan(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one dense [batch, tokens, heads, 128] scan."""
    batch, tokens, value_heads, value_dim = value.shape
    expected_prefix = (batch, tokens)
    if query.shape[:2] != expected_prefix or key.shape[:2] != expected_prefix:
        raise ValueError("GDN query, key, and value tensors must share batch and token dimensions.")
    if value_dim != GDN_HEAD_DIM or query.shape[-1] != GDN_HEAD_DIM or key.shape[-1] != GDN_HEAD_DIM:
        raise ValueError("gdn_scan_batch_invariant requires 128-wide key and value heads.")
    expected_gate_shape = (batch, tokens, value_heads)
    if log_decay.shape != expected_gate_shape or beta.shape != expected_gate_shape:
        raise ValueError("log_decay and beta must have shape [batch, tokens, value_heads].")
    if initial_state.shape != (batch, value_heads, GDN_HEAD_DIM, GDN_HEAD_DIM):
        raise ValueError(
            "initial_state must have shape [batch, value_heads, 128, 128]."
        )

    query = _expand_key_heads(query, value_heads)
    key = _expand_key_heads(key, value_heads)

    def flatten_heads(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.permute(0, 2, 1, 3).reshape(
            batch * value_heads,
            tokens,
            GDN_HEAD_DIM,
        )

    flat_query = flatten_heads(query).to(torch.bfloat16).contiguous()
    flat_key = flatten_heads(key).to(torch.bfloat16).contiguous()
    flat_value = flatten_heads(value).to(torch.bfloat16).contiguous()
    flat_alpha = (
        torch.exp(log_decay.float())
        .permute(0, 2, 1)
        .reshape(batch * value_heads, tokens)
        .contiguous()
    )
    flat_beta = (
        beta.float()
        .permute(0, 2, 1)
        .reshape(batch * value_heads, tokens)
        .contiguous()
    )
    flat_state = initial_state.reshape(
        batch * value_heads,
        GDN_HEAD_DIM,
        GDN_HEAD_DIM,
    ).float().contiguous()

    if not is_gdn_scan_batch_invariant_available():
        raise RuntimeError(
            "The A5 gdn_scan_batch_invariant custom operator is not installed."
        )
    flat_output, flat_final_state = torch.ops._C_ascend.gdn_scan_batch_invariant(
        flat_query,
        flat_key,
        flat_value,
        flat_alpha,
        flat_beta,
        flat_state,
    )
    output = flat_output.reshape(
        batch,
        value_heads,
        tokens,
        GDN_HEAD_DIM,
    ).permute(0, 2, 1, 3)
    final_state = flat_final_state.reshape(
        batch,
        value_heads,
        GDN_HEAD_DIM,
        GDN_HEAD_DIM,
    )
    return output, final_state


def gdn_scan_batch_invariant_dense(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    max_tokens_per_launch: int = GDN_MAX_TOKENS_PER_LAUNCH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a uniform dense scan with bounded sequence chunks and FP32 state carry."""
    tokens = query.shape[1]
    if tokens <= 0:
        raise ValueError("GDN scan requires at least one token.")
    if max_tokens_per_launch <= 0:
        raise ValueError("max_tokens_per_launch must be positive.")

    state = initial_state
    outputs = []
    for start in range(0, tokens, max_tokens_per_launch):
        end = min(start + max_tokens_per_launch, tokens)
        output, state = _run_flat_scan(
            query[:, start:end],
            key[:, start:end],
            value[:, start:end],
            log_decay[:, start:end],
            beta[:, start:end],
            state,
        )
        outputs.append(output)
    return torch.cat(outputs, dim=1), state


def gdn_scan_batch_invariant_packed(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens_host: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run packed TND prefill sequences without mixing their recurrent states."""
    if query.shape[0] != 1:
        raise ValueError("Packed GDN input must use the vLLM outer batch dimension of one.")
    cu_seqlens = tuple(int(offset) for offset in cu_seqlens_host)
    if len(cu_seqlens) != initial_state.shape[0] + 1:
        raise ValueError("cu_seqlens_host must describe every initial-state row.")
    if not cu_seqlens or cu_seqlens[0] != 0 or cu_seqlens[-1] != query.shape[1]:
        raise ValueError("cu_seqlens_host must span the complete packed token dimension.")

    outputs = []
    final_states = []
    for sequence_index, (start, end) in enumerate(
        zip(cu_seqlens, cu_seqlens[1:])
    ):
        if end < start:
            raise ValueError("cu_seqlens_host must be nondecreasing.")
        if end == start:
            final_states.append(initial_state[sequence_index : sequence_index + 1])
            continue
        output, final_state = gdn_scan_batch_invariant_dense(
            query[:, start:end],
            key[:, start:end],
            value[:, start:end],
            log_decay[:, start:end],
            beta[:, start:end],
            initial_state[sequence_index : sequence_index + 1],
        )
        outputs.append(output)
        final_states.append(final_state)

    if outputs:
        packed_output = torch.cat(outputs, dim=1)
    else:
        packed_output = value[:, :0]
    return packed_output, torch.cat(final_states, dim=0)
