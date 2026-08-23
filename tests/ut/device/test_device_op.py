from unittest.mock import patch

import torch

from vllm_ascend.device.device_op import _training_parity_moe_gating_top_k


def test_device_op_placeholder():
    pass


def test_training_parity_router_uses_bi_softmax():
    router_logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.bfloat16)
    bi_weights = torch.tensor([[0.75, 0.25]], dtype=torch.float32)
    with patch(
        "vllm_ascend.device.device_op.npu_softmax_batch_invariant",
        return_value=bi_weights,
    ) as bi_softmax:
        weights, expert_ids, out = _training_parity_moe_gating_top_k(
            router_logits,
            k=2,
            k_group=1,
            group_count=1,
            norm_type=0,
            routed_scaling_factor=2.0,
            bias_opt=None,
        )

    bi_softmax.assert_called_once()
    softmax_input, dim = bi_softmax.call_args.args
    assert softmax_input.dtype == torch.float32
    assert softmax_input.shape == (1, 2)
    assert dim == -1
    assert torch.equal(softmax_input, torch.tensor([[3.0, 2.0]]))
    assert weights.shape == (1, 2)
    assert weights.dtype == torch.bfloat16
    assert torch.equal(weights, torch.tensor([[1.5, 0.5]], dtype=torch.bfloat16))
    assert expert_ids.tolist() == [[3, 2]]
    assert out.numel() == 0
