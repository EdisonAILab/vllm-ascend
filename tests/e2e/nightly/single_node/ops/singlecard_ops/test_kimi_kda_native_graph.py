# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from unittest.mock import patch

import torch
from torch import nn
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.kimi_kda import AscendKimiGatedDeltaNetAttention


@torch.inference_mode()
def test_reduced_native_kda_decode_is_exact_under_aclgraph_replay():
    """The reduced-model KDA oracle must preserve state exactly in FULL graph."""
    torch.manual_seed(20260910)
    device = torch.device("npu")
    graph_rows, active_rows, heads, dim = 2, 1, 4, 32
    state_slots = 8

    layer = object.__new__(AscendKimiGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer.head_dim = dim
    layer.gate_lower_bound = -5.0
    layer.A_log = nn.Parameter(torch.randn(1, 1, heads, 1, device=device))
    layer.dt_bias = nn.Parameter(torch.randn(heads, dim, device=device))

    graph_q = torch.zeros(1, graph_rows, heads, dim, dtype=torch.bfloat16, device=device)
    graph_k = torch.zeros_like(graph_q)
    graph_v = torch.zeros_like(graph_q)
    graph_gate = torch.zeros_like(graph_q)
    graph_beta = torch.zeros(1, graph_rows, heads, dtype=torch.float32, device=device)
    graph_cu_seqlens = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
    graph_state_indices = torch.tensor([3, PAD_SLOT_ID], dtype=torch.int32, device=device)
    initial_state = torch.randn(state_slots, heads, dim, dim, dtype=torch.float32, device=device)
    graph_state = initial_state.clone()

    with patch.dict(os.environ, {"VLLM_ASCEND_KIMI_NATIVE_STATE_OPS": "1"}):
        layer._run_native_kda_graph_decode(
            graph_q,
            graph_k,
            graph_v,
            graph_gate,
            graph_beta,
            graph_state,
            graph_cu_seqlens,
            graph_state_indices,
        )
        torch.npu.synchronize()
        graph_state.copy_(initial_state)

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            graph_output = layer._run_native_kda_graph_decode(
                graph_q,
                graph_k,
                graph_v,
                graph_gate,
                graph_beta,
                graph_state,
                graph_cu_seqlens,
                graph_state_indices,
            )
        torch.npu.synchronize()
        graph_state.copy_(initial_state)
        eager_state = initial_state.clone()

        for step in range(8):
            torch.manual_seed(20260910 + step)
            graph_q.copy_(torch.randn_like(graph_q))
            graph_k.copy_(torch.randn_like(graph_k))
            graph_v.copy_(torch.randn_like(graph_v))
            graph_gate.copy_(torch.randn_like(graph_gate))
            graph_beta.copy_(torch.rand_like(graph_beta))
            graph_state_indices.copy_(
                torch.tensor([3, PAD_SLOT_ID], dtype=torch.int32, device=device)
            )

            eager_output = layer._run_native_kda_graph_decode(
                graph_q,
                graph_k,
                graph_v,
                graph_gate,
                graph_beta,
                eager_state,
                graph_cu_seqlens,
                graph_state_indices,
            )
            graph.replay()
            torch.npu.synchronize()

            assert torch.equal(graph_output, eager_output), f"output mismatch at step {step}"
            assert torch.equal(graph_state, eager_state), f"state mismatch at step {step}"
            assert torch.equal(
                graph_output[:, active_rows:],
                torch.zeros_like(graph_output[:, active_rows:]),
            )
