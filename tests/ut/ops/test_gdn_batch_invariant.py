# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops import gdn_batch_invariant as gdn_bi


def _inputs(
    *,
    batch: int = 2,
    tokens: int = 3,
    key_heads: int = 1,
    value_heads: int = 2,
) -> tuple[torch.Tensor, ...]:
    query = torch.arange(
        batch * tokens * key_heads * gdn_bi.GDN_HEAD_DIM,
        dtype=torch.bfloat16,
    ).reshape(batch, tokens, key_heads, gdn_bi.GDN_HEAD_DIM)
    key = query + 1
    value = torch.arange(
        batch * tokens * value_heads * gdn_bi.GDN_HEAD_DIM,
        dtype=torch.bfloat16,
    ).reshape(batch, tokens, value_heads, gdn_bi.GDN_HEAD_DIM)
    log_decay = torch.full((batch, tokens, value_heads), -0.5, dtype=torch.float32)
    beta = torch.full((batch, tokens, value_heads), 0.25, dtype=torch.bfloat16)
    state = torch.zeros(
        batch,
        value_heads,
        gdn_bi.GDN_HEAD_DIM,
        gdn_bi.GDN_HEAD_DIM,
        dtype=torch.float32,
    )
    return query, key, value, log_decay, beta, state


def test_flat_scan_expands_heads_and_restores_layout():
    query, key, value, log_decay, beta, state = _inputs()
    captured = {}

    def fake_scan(flat_query, flat_key, flat_value, alpha, flat_beta, flat_state):
        captured.update(
            query=flat_query.clone(),
            key=flat_key.clone(),
            alpha=alpha.clone(),
            beta=flat_beta.clone(),
            state=flat_state.clone(),
        )
        return flat_value + 2, flat_state + 3

    with (
        patch.object(gdn_bi, "is_gdn_scan_batch_invariant_available", return_value=True),
        patch.object(
            torch.ops._C_ascend,
            "gdn_scan_batch_invariant",
            side_effect=fake_scan,
            create=True,
        ),
    ):
        output, final_state = gdn_bi.gdn_scan_batch_invariant_dense(
            query,
            key,
            value,
            log_decay,
            beta,
            state,
        )

    assert captured["query"].shape == (4, 3, 128)
    assert captured["key"].shape == (4, 3, 128)
    torch.testing.assert_close(captured["query"][0], captured["query"][1])
    torch.testing.assert_close(captured["query"][2], captured["query"][3])
    torch.testing.assert_close(captured["alpha"], torch.exp(log_decay).permute(0, 2, 1).reshape(4, 3))
    torch.testing.assert_close(captured["beta"], beta.float().permute(0, 2, 1).reshape(4, 3))
    torch.testing.assert_close(output, value + 2)
    torch.testing.assert_close(final_state, state + 3)


def test_dense_scan_carries_fp32_state_across_bounded_chunks():
    query, key, value, log_decay, beta, state = _inputs(batch=1, tokens=5)
    spans = []

    def fake_flat_scan(q, k, v, g, b, initial_state):
        del k, g, b
        spans.append(q.shape[1])
        return v + len(spans), initial_state + 1

    with patch.object(gdn_bi, "_run_flat_scan", side_effect=fake_flat_scan):
        output, final_state = gdn_bi.gdn_scan_batch_invariant_dense(
            query,
            key,
            value,
            log_decay,
            beta,
            state,
            max_tokens_per_launch=2,
        )

    assert spans == [2, 2, 1]
    torch.testing.assert_close(output[:, :2], value[:, :2] + 1)
    torch.testing.assert_close(output[:, 2:4], value[:, 2:4] + 2)
    torch.testing.assert_close(output[:, 4:], value[:, 4:] + 3)
    torch.testing.assert_close(final_state, state + 3)
    assert final_state.dtype == torch.float32


def test_packed_scan_keeps_sequence_states_separate_and_allows_empty_rows():
    query, key, value, log_decay, beta, state = _inputs(batch=1, tokens=5)
    state = torch.stack((state[0], state[0] + 10, state[0] + 20))
    calls = []

    def fake_dense(q, k, v, g, b, initial_state, max_tokens_per_launch=gdn_bi.GDN_MAX_TOKENS_PER_LAUNCH):
        del k, g, b, max_tokens_per_launch
        calls.append((q.shape[1], initial_state.clone()))
        return v + len(calls), initial_state + 1

    with patch.object(gdn_bi, "gdn_scan_batch_invariant_dense", side_effect=fake_dense):
        output, final_state = gdn_bi.gdn_scan_batch_invariant_packed(
            query,
            key,
            value,
            log_decay,
            beta,
            state,
            (0, 2, 2, 5),
        )

    assert [length for length, _ in calls] == [2, 3]
    torch.testing.assert_close(output[:, :2], value[:, :2] + 1)
    torch.testing.assert_close(output[:, 2:], value[:, 2:] + 2)
    torch.testing.assert_close(final_state[0], state[0] + 1)
    torch.testing.assert_close(final_state[1], state[1])
    torch.testing.assert_close(final_state[2], state[2] + 1)


@pytest.mark.parametrize(
    "cu_seqlens",
    [
        (1, 5),
        (0, 4),
        (0, 3, 2, 5),
    ],
)
def test_packed_scan_rejects_invalid_offsets(cu_seqlens):
    query, key, value, log_decay, beta, state = _inputs(batch=1, tokens=5)
    state = state[: len(cu_seqlens) - 1]
    with pytest.raises(ValueError):
        gdn_bi.gdn_scan_batch_invariant_packed(
            query,
            key,
            value,
            log_decay,
            beta,
            state,
            cu_seqlens,
        )


def test_dense_scan_rejects_incompatible_gate_shape():
    query, key, value, log_decay, beta, state = _inputs()
    with pytest.raises(ValueError, match="log_decay and beta"):
        gdn_bi._run_flat_scan(
            query,
            key,
            value,
            log_decay[:, :, :1],
            beta,
            state,
        )


def test_state_scatter_forwards_fixed_width_indices_and_contiguous_fp32_updates():
    state = torch.zeros(3, 2, 8, dtype=torch.float32)
    updates = torch.ones(4, 2, 8, dtype=torch.bfloat16)
    indices = torch.tensor([0, 1, -1, -1], dtype=torch.int64)
    captured = {}

    def fake_scatter(cache, update_rows, state_indices):
        captured["cache"] = cache
        captured["updates"] = update_rows
        captured["indices"] = state_indices

    with patch.object(
        torch.ops._C_ascend,
        "gdn_scatter_state_batch_invariant",
        side_effect=fake_scatter,
        create=True,
    ):
        gdn_bi.gdn_scatter_state_batch_invariant(state, updates, indices)

    assert captured["cache"] is state
    assert captured["updates"].dtype == torch.float32
    assert captured["updates"].is_contiguous()
    assert captured["indices"].dtype == torch.int32
    assert captured["indices"].tolist() == [0, 1, -1, -1]
