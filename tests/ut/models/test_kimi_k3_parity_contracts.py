# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm_ascend.models import kimi_k3 as kimi_k3_module
from vllm_ascend.models.kimi_k3 import (
    AscendKimiK3ForCausalLM,
    _KimiPairwiseRMSNorm,
    _KimiQkvAProjLinear,
    _kimi_packed_modules_mapping,
)
from vllm_ascend.ops.kimi_kda import _kimi_use_fused_kda_qkv


def test_kimi_pairwise_rmsnorm_uses_explicit_tree():
    hidden_states = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0, 5.0]],
        dtype=torch.bfloat16,
    )
    norm = _KimiPairwiseRMSNorm(5, 1e-6)
    norm.weight.data.copy_(
        torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5], dtype=norm.weight.dtype)
    )

    values = hidden_states.float().square()
    pair_01 = values[..., 0:1] + values[..., 1:2]
    pair_23 = values[..., 2:3] + values[..., 3:4]
    pair_4_padding = values[..., 4:5] + torch.zeros_like(values[..., 4:5])
    variance = ((pair_01 + pair_23) + pair_4_padding) / 5
    expected = norm.weight * (
        hidden_states.float() * torch.rsqrt(variance + 1e-6)
    ).to(hidden_states.dtype)

    assert torch.equal(norm(hidden_states), expected)


@pytest.mark.parametrize(
    ("quantized", "separate", "expected"),
    [
        (False, False, False),
        (True, False, True),
        (True, True, False),
    ],
)
def test_kimi_kda_qkv_fusion_is_opt_in_compatible(
    quantized: bool,
    separate: bool,
    expected: bool,
):
    quant_config = object() if quantized else None
    with patch.dict(
        os.environ,
        {"VLLM_ASCEND_KIMI_SEPARATE_KDA_QKV": "1" if separate else "0"},
    ):
        assert _kimi_use_fused_kda_qkv(quant_config) is expected


def test_kimi_separate_kda_qkv_drops_only_that_packing_contract():
    base_mapping = AscendKimiK3ForCausalLM.packed_modules_mapping
    with patch.dict(os.environ, {"VLLM_ASCEND_KIMI_SEPARATE_KDA_QKV": "1"}):
        actual = _kimi_packed_modules_mapping(base_mapping, object())

    assert "fused_qkv" not in actual
    assert actual["fused_qkv_a_proj"] == ["q_a_proj", "kv_a_proj_with_mqa"]
    assert actual["gate_up_proj"] == ["gate_proj", "up_proj"]
    assert "fused_qkv" in base_mapping


def test_kimi_default_quantized_mapping_keeps_qkv_packing():
    base_mapping = AscendKimiK3ForCausalLM.packed_modules_mapping
    with patch.dict(os.environ, {"VLLM_ASCEND_KIMI_SEPARATE_KDA_QKV": "0"}):
        actual = _kimi_packed_modules_mapping(base_mapping, object())

    assert actual == base_mapping
    assert actual is not base_mapping


def test_kimi_dense_mlp_graph_boundary_calls_registered_native_path():
    class FakeDenseMLP:
        @staticmethod
        def _forward_reference(hidden_states):
            return hidden_states + 2

    layer_name = "test.layers.0.mlp"
    kimi_k3_module._KIMI_DENSE_MLP_REGISTRY[layer_name] = FakeDenseMLP()
    inputs = torch.tensor([[1.0, -3.0]], dtype=torch.bfloat16)
    try:
        actual = kimi_k3_module._kimi_reference_dense_mlp_impl(
            inputs,
            layer_name,
        )
        fake = kimi_k3_module._kimi_reference_dense_mlp_fake(
            inputs,
            layer_name,
        )
    finally:
        kimi_k3_module._KIMI_DENSE_MLP_REGISTRY.pop(layer_name, None)

    assert torch.equal(actual, torch.tensor([[3.0, -1.0]], dtype=torch.bfloat16))
    assert fake.shape == inputs.shape
    assert fake.dtype == inputs.dtype


def test_kimi_dense_mlp_reference_pads_and_slices_fixed_capacity():
    mlp = kimi_k3_module.KimiK3MLP.__new__(kimi_k3_module.KimiK3MLP)
    nn.Module.__init__(mlp)
    mlp.reference_dense_mlp_capacity = 4
    observed_shapes = []

    def fake_native(hidden_states, export_buffer=None):
        assert export_buffer is None
        observed_shapes.append(tuple(hidden_states.shape))
        return hidden_states + 1

    mlp._forward_native = fake_native
    inputs = torch.arange(10, dtype=torch.bfloat16).reshape(5, 2)

    actual = mlp._forward_reference(inputs)

    assert observed_shapes == [(4, 2), (4, 2)]
    assert actual.shape == inputs.shape
    assert torch.equal(actual, inputs + 1)


def test_kimi_separate_mla_qkv_a_matches_independent_linear_calls():
    projection = _KimiQkvAProjLinear.__new__(_KimiQkvAProjLinear)
    nn.Module.__init__(projection)
    projection.kimi_q_lora_rank = 3
    projection.weight = nn.Parameter(
        torch.tensor(
            [
                [0.5, -0.25, 1.0, 0.75],
                [1.25, 0.5, -0.75, 0.25],
                [-0.5, 0.25, 0.75, 1.0],
                [0.5, 1.0, -0.25, -0.75],
                [0.75, -0.5, 1.25, 0.25],
            ],
            dtype=torch.bfloat16,
        ),
        requires_grad=False,
    )
    hidden_states = torch.tensor(
        [[0.25, -0.5, 0.75, -1.0], [1.0, 0.5, -0.25, 0.75]],
        dtype=torch.bfloat16,
    )
    expected = torch.cat(
        (
            F.linear(hidden_states, projection.weight[:3]),
            F.linear(hidden_states, projection.weight[3:]),
        ),
        dim=-1,
    )

    with patch.dict(os.environ, {"VLLM_ASCEND_KIMI_SEPARATE_MLA_QKV_A": "1"}):
        actual, bias = projection(hidden_states)

    assert bias is None
    assert torch.equal(actual, expected)
