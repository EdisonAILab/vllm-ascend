from unittest import mock

import torch

from vllm_ascend.device.device_op import (
    A5DeviceAdaptor,
    BaseDeviceAdaptor,
    _gather_paged_kv_cache,
)


def test_reshape_and_cache_makes_scatter_inputs_contiguous():
    key = torch.randn(2, 3, 4).transpose(0, 1)
    value = torch.randn(2, 3, 4).transpose(0, 1)
    slot_mapping = torch.arange(8, dtype=torch.int32)[::2]
    key_cache = object()
    value_cache = object()

    assert not key.is_contiguous()
    assert not value.is_contiguous()
    assert not slot_mapping.is_contiguous()

    with mock.patch("vllm_ascend.device.device_op.torch_npu.npu_scatter_pa_kv_cache") as mock_scatter:
        BaseDeviceAdaptor.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

    mock_scatter.assert_called_once()
    call_kwargs = mock_scatter.call_args.kwargs
    assert call_kwargs["key"] is not key
    assert call_kwargs["value"] is not value
    assert call_kwargs["slot_mapping"] is not slot_mapping
    assert call_kwargs["key"].is_contiguous()
    assert call_kwargs["value"].is_contiguous()
    assert call_kwargs["slot_mapping"].is_contiguous()
    torch.testing.assert_close(call_kwargs["key"], key)
    torch.testing.assert_close(call_kwargs["value"], value)
    torch.testing.assert_close(call_kwargs["slot_mapping"], slot_mapping)
    assert call_kwargs["key_cache"] is key_cache
    assert call_kwargs["value_cache"] is value_cache
    assert call_kwargs["cache_mode"] == "Norm"


def test_kv_cache_load_makes_seq_lens_contiguous():
    cache_kv_c = object()
    cache_k_pe = object()
    block_table = object()
    context_seq_len_npu = torch.arange(8, dtype=torch.int32)[::2]
    seq_starts = object()
    key = object()
    value = object()

    assert not context_seq_len_npu.is_contiguous()

    with mock.patch("vllm_ascend.device.device_op.torch_npu.npu_gather_pa_kv_cache") as mock_gather:
        BaseDeviceAdaptor.kv_cache_load(
            cache_kv_c,
            cache_k_pe,
            block_table,
            context_seq_len_npu,
            seq_starts,
            key,
            value,
        )

    mock_gather.assert_called_once()
    call_args = mock_gather.call_args.args
    assert call_args[0] is cache_kv_c
    assert call_args[1] is cache_k_pe
    assert call_args[2] is block_table
    assert call_args[3] is not context_seq_len_npu
    assert call_args[3].is_contiguous()
    torch.testing.assert_close(call_args[3], context_seq_len_npu)
    assert mock_gather.call_args.kwargs["seq_offset"] is seq_starts
    assert mock_gather.call_args.kwargs["key"] is key
    assert mock_gather.call_args.kwargs["value"] is value


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
