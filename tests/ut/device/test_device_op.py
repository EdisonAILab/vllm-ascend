from unittest.mock import patch

import torch

from vllm_ascend.device.device_op import (
    _training_parity_moe_gating_top_k,
    _training_parity_softmax_input,
)
from vllm_ascend.training_parity import (
    get_training_parity_sequence_length,
    set_training_parity_sequence_length,
)


def test_device_op_placeholder():
    pass


def test_training_parity_sequence_length_context():
    set_training_parity_sequence_length(456)
    assert get_training_parity_sequence_length() == 456
    set_training_parity_sequence_length(1)


def test_training_parity_softmax_input_uses_prefix_length():
    logits = torch.arange(8, dtype=torch.bfloat16).reshape(1, 8)
    with patch(
        "vllm_ascend.device.device_op.get_training_parity_sequence_length",
        return_value=456,
    ):
        expanded = _training_parity_softmax_input(logits)

    assert expanded.shape == (456, 8)
    assert expanded.is_contiguous()
    assert torch.equal(expanded[0], logits[0])
    assert torch.equal(expanded[-1], logits[0])


def test_training_parity_router_returns_only_decode_rows():
    router_logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.bfloat16)
    with patch(
        "vllm_ascend.device.device_op.get_training_parity_sequence_length",
        return_value=32,
    ):
        weights, expert_ids, out = _training_parity_moe_gating_top_k(
            router_logits,
            k=2,
            k_group=1,
            group_count=1,
            norm_type=0,
            routed_scaling_factor=1.0,
            bias_opt=None,
        )

    assert weights.shape == (1, 2)
    assert expert_ids.tolist() == [[3, 2]]
    assert out.numel() == 0
