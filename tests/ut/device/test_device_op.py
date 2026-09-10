from unittest.mock import patch

import torch

from vllm_ascend.device.device_op import (
    A5DeviceAdaptor,
    _gather_paged_kv_cache,
    _training_parity_moe_gating_top_k,
)


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


def test_gather_paged_kv_cache_handles_multiple_requests_and_offsets():
    key_cache = torch.arange(6 * 4 * 2 * 3, dtype=torch.float32).view(6, 4, 2, 3)
    value_cache = key_cache + 10_000
    block_tables = torch.tensor([[3, 1, 5], [2, 4, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([5, 7], dtype=torch.int32)
    seq_offsets = torch.tensor([2, 1], dtype=torch.int32)
    key = torch.empty(12, 2, 3)
    value = torch.empty_like(key)

    _gather_paged_kv_cache(
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        seq_offsets,
        key,
        value,
    )

    expected_indices = [
        (request, position)
        for request, (length, start) in enumerate(zip(seq_lens, seq_offsets))
        for position in range(int(start), int(start + length))
    ]
    expected_key = torch.stack(
        [
            key_cache[
                block_tables[request, position // key_cache.shape[1]],
                position % key_cache.shape[1],
            ]
            for request, position in expected_indices
        ]
    )
    assert torch.equal(key, expected_key)
    assert torch.equal(value, expected_key + 10_000)


def test_a5_kv_cache_load_uses_normal_layout_gather():
    key_cache = torch.arange(2 * 4 * 2, dtype=torch.float32).view(2, 4, 1, 2)
    value_cache = key_cache + 100
    block_tables = torch.tensor([[1, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([3], dtype=torch.int32)
    seq_offsets = torch.tensor([2], dtype=torch.int32)
    key = torch.empty(3, 1, 2)
    value = torch.empty_like(key)

    A5DeviceAdaptor.kv_cache_load(
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        seq_offsets,
        key,
        value,
    )

    assert torch.equal(key, torch.stack([key_cache[1, 2], key_cache[1, 3], key_cache[0, 0]]))
    assert torch.equal(value, key + 100)
