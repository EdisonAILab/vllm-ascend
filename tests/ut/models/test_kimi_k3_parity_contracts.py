# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm_ascend.models.kimi_k3 import (
    AscendKimiK3ForCausalLM,
    _KimiQkvAProjLinear,
    _kimi_packed_modules_mapping,
)
from vllm_ascend.ops.kimi_kda import _kimi_use_fused_kda_qkv


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
