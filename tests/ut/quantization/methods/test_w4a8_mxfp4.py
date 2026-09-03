from unittest.mock import patch

import torch

from tests.ut.quantization.conftest_quantization import create_mock_vllm_config
from vllm_ascend.quantization.methods.w4a8_mxfp4 import (
    AscendW4A8MXFPDynamicLinearMethod,
    _megatron_reference_select_experts,
)


def test_w4a8_linear_accepts_packed_checkpoint_marker():
    with (
        patch("vllm_ascend.quantization.methods.w4a8_mxfp4.ensure_mxfp4_linear_available"),
        patch(
            "vllm_ascend.quantization.methods.w4a8_mxfp4.get_current_vllm_config",
            return_value=create_mock_vllm_config(),
        ),
    ):
        method = AscendW4A8MXFPDynamicLinearMethod(use_weight_packed=True)

    assert method.group_size == 32


def test_kimi_reference_router_matches_fp32_megatron_topk_contract():
    router_logits = torch.tensor(
        [[-1.0, 0.25, 2.0, 0.5], [1.5, -0.5, 0.75, 0.0]],
        dtype=torch.bfloat16,
    )

    weights, expert_ids = _megatron_reference_select_experts(
        router_logits=router_logits,
        top_k=2,
        renormalize=True,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        expert_bias=None,
        use_grouped_topk=False,
        num_expert_group=None,
        topk_group=None,
    )

    scores = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    expected_weights, expected_ids = torch.topk(
        scores,
        k=2,
        dim=-1,
        sorted=torch.is_grad_enabled(),
    )
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True) + 1e-20

    assert torch.equal(expert_ids, expected_ids.to(torch.int32))
    assert torch.equal(weights, expected_weights.to(router_logits.dtype))
