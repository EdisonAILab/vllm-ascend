# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import numpy as np
import torch

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.attention.context_parallel.attention_cp import (
    AscendAttentionDCPImpl,
    AscendAttentionDCPMetadata,
    AscendAttentionDCPMetadataBuilder,
    AscendMetadataForDecode,
    AscendMetadataForPrefill,
)
from vllm_ascend.attention.context_parallel.common_cp import (
    _update_out_and_lse,
)


def test_gqa_dcp_extends_v1_backend_without_polluting_base_metadata() -> None:
    assert issubclass(AscendAttentionDCPImpl, AscendAttentionBackendImpl)
    assert issubclass(
        AscendAttentionDCPMetadataBuilder,
        AscendAttentionMetadataBuilder,
    )
    assert AscendAttentionDCPMetadataBuilder.metadata_cls is (AscendAttentionDCPMetadata)
    assert not hasattr(AscendMetadata(), "decode_meta")
    assert not hasattr(AscendMetadata(), "prefill")


def test_dcp_chunked_request_mask_marks_nonempty_contexts() -> None:
    local_context_lens = torch.tensor(
        [
            [0, 0],
            [4, 0],
            [0, 7],
        ],
        dtype=torch.int32,
    )

    assert AscendAttentionDCPMetadataBuilder._get_chunked_req_mask(local_context_lens) == [
        False,
        True,
        True,
    ]


def test_dcp_decode_metadata_keeps_rank_local_context_lengths() -> None:
    local_context_lens = np.array([[11, 12], [21, 22]], dtype=np.int32)
    block_tables = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    metadata = AscendMetadataForDecode(
        num_computed_tokens_of_dcp=local_context_lens,
        block_tables=block_tables,
    )

    np.testing.assert_array_equal(metadata.num_computed_tokens_of_dcp[:, 1], [12, 22])
    assert metadata.block_tables is block_tables


def test_dcp_partial_attention_merge_matches_weighted_reference() -> None:
    outputs = torch.tensor(
        [
            [[[[1.0, 3.0]]]],
            [[[[5.0, 7.0]]]],
        ]
    ).reshape(2, 1, 1, 2)
    lse = torch.tensor([0.0, np.log(3.0)], dtype=torch.float32).reshape(2, 1, 1, 1)

    output, merged_lse = _update_out_and_lse(outputs, lse)

    torch.testing.assert_close(output, torch.tensor([[[4.0, 6.0]]]))
    torch.testing.assert_close(merged_lse, torch.tensor([[[np.log(4.0)]]], dtype=torch.float32))


@patch("vllm_ascend.attention.context_parallel.attention_cp.torch_npu.npu_fused_infer_attention_score")
def test_dcp_chunked_prefill_uses_batch_invariant_fia_dispatch(mock_fia) -> None:
    impl = AscendAttentionDCPImpl.__new__(AscendAttentionDCPImpl)
    impl.dcp_rank = 0
    impl.dcp_size = 1
    impl.num_heads = 2
    impl.num_kv_heads = 1
    impl.head_size = 4
    impl.scale = 0.5
    key = torch.randn(3, 1, 4)
    value = torch.randn(3, 1, 4)
    impl._load_kv_for_chunk = lambda *_args: (key, value)

    chunked_context = AscendMetadataForPrefill.ChunkedContextMetadata(
        actual_chunk_seq_lengths=torch.tensor([2]),
        actual_seq_lengths_kv=torch.tensor([3]),
        starts=torch.tensor([0]),
        chunk_seq_mask_filtered_indices=torch.tensor([0]),
        local_context_lens_allranks=torch.tensor([[3]]),
        local_total_toks=3,
    )
    metadata = AscendAttentionDCPMetadata(
        prefill=AscendMetadataForPrefill(chunked_context=chunked_context),
    )
    query = torch.randn(2, 2, 4)
    expected_output = torch.randn(2, 2, 4)
    expected_lse = torch.randn(2, 2, 1)
    mock_fia.return_value = (expected_output, expected_lse)

    output, lse = impl._compute_prefill_context(query, (key, value), metadata)

    assert output is expected_output
    assert lse is expected_lse
    mock_fia.assert_called_once()


@patch("vllm_ascend.attention.context_parallel.attention_cp.record_attention_compute_start")
@patch("vllm_ascend.attention.context_parallel.attention_cp.torch_npu.npu_fused_infer_attention_score")
def test_dcp_prefill_uses_batch_invariant_fia_dispatch(mock_fia, _mock_record) -> None:
    impl = AscendAttentionDCPImpl.__new__(AscendAttentionDCPImpl)
    impl.num_heads = 2
    impl.num_kv_heads = 1
    impl.scale = 0.5

    metadata = AscendAttentionDCPMetadata(
        num_actual_tokens=2,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        prefill=AscendMetadataForPrefill(
            chunked_context=None,
            actual_seq_lengths_q=torch.tensor([2]),
        ),
    )
    query = torch.randn(2, 2, 4)
    key = torch.randn(2, 1, 4)
    value = torch.randn(2, 1, 4)
    expected_output = torch.randn(2, 2, 4)
    expected_lse = torch.randn(2, 2, 1)
    mock_fia.return_value = (expected_output, expected_lse)
    output = torch.empty_like(expected_output)

    result = impl.forward_impl(query, key, value, (key, value), metadata, output)

    assert result is output
    torch.testing.assert_close(result, expected_output)
    mock_fia.assert_called_once()
