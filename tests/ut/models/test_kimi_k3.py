# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from vllm.model_executor.models.utils import StageMissingLayer

from vllm_ascend.models import kimi_k3
from vllm_ascend.models.kimi_k3 import (
    AscendKimiK3ForCausalLM,
    AscendKimiK3ForConditionalGeneration,
    KimiK3MLP,
    KimiK3MoE,
    KimiK3MultiModalProjector,
    KimiK3TextModel,
    KimiK3VisionEncoderLayer,
    _apply_attention_residual,
    _configure_kimi_mlapo_shape,
    _decomposed_mla_forward_decode,
    _decomposed_mla_forward_prefill,
    _KimiReferenceRMSNorm,
    _kimi_reference_rowwise_attention_residual_impl,
    _move_module_to_device,
    _reference_mla_forward_decode,
    _reference_mla_forward_prefill,
    _resolve_packed_expert_weight_name,
    _routed_latent_quant_config,
    get_spec_layer_idx_from_weight_name,
)
from vllm_ascend.ops.activation import (
    AscendSituAndMul,
    SituActivationConfig,
    situ_and_mul,
)
from vllm_ascend.transformers_utils.configs.kimi_k3 import (
    KimiK3Config,
    KimiK3VisionConfig,
)


def test_kimi_k3_model_declares_checkpoint_packing_contract():
    assert AscendKimiK3ForCausalLM.packed_modules_mapping["fused_qkv"] == [
        "q_proj",
        "k_proj",
        "v_proj",
    ]
    assert AscendKimiK3ForCausalLM.packed_modules_mapping["experts"] == [
        "experts.0.w1",
        "experts.0.w3",
        "experts.0.w2",
    ]


@pytest.mark.parametrize(
    ("enabled", "kv_lora_rank", "expected_enabled", "expected_fallback"),
    [
        (True, 128, False, True),
        (True, 512, True, False),
        (False, 128, False, False),
    ],
)
def test_kimi_mlapo_reduced_shape_fallback(
    enabled: bool,
    kv_lora_rank: int,
    expected_enabled: bool,
    expected_fallback: bool,
):
    mla_impl = SimpleNamespace(enable_mlapo=enabled)

    actual = _configure_kimi_mlapo_shape(mla_impl, kv_lora_rank)

    assert actual is expected_fallback
    assert mla_impl.enable_mlapo is expected_enabled


def test_kimi_k3_loads_qkv_checkpoint_shards_into_fused_linear():
    model = KimiK3TextModel.__new__(KimiK3TextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_experts=0)
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].self_attn = nn.Module()
    model.layers[0].self_attn.fused_qkv = nn.Module()

    fused_weight = nn.Parameter(torch.empty(1))
    fused_weight.weight_loader = MagicMock()
    model.layers[0].self_attn.fused_qkv.register_parameter("weight", fused_weight)
    weights = [(f"layers.0.self_attn.{name}.weight", torch.empty(1)) for name in ("q_proj", "k_proj", "v_proj")]

    with (
        patch("vllm_ascend.models.kimi_k3.get_spec_layer_idx_from_weight_name", return_value=None),
        patch("vllm_ascend.models.kimi_k3.fused_moe_make_expert_params_mapping", return_value=[]),
        patch("vllm_ascend.models.kimi_k3.is_pp_missing_parameter", return_value=False),
    ):
        loaded = model.load_weights(weights)

    assert [call.args[2] for call in fused_weight.weight_loader.call_args_list] == ["q", "k", "v"]
    assert loaded == {"layers.0.self_attn.fused_qkv.weight"}


@pytest.mark.parametrize(
    ("quant_name", "uses_quantized_latent_projections"),
    [
        ("ascend", True),
        ("compressed-tensors", True),
        ("other", False),
    ],
)
def test_kimi_k3_quantizes_packed_latent_projections(
    quant_name: str,
    uses_quantized_latent_projections: bool,
):
    quant_config = MagicMock()
    quant_config.get_name.return_value = quant_name

    actual = _routed_latent_quant_config(quant_config)

    if uses_quantized_latent_projections:
        assert actual is quant_config
    else:
        assert actual is None


def test_kimi_k3_unquantized_model_keeps_latent_projections_unquantized():
    assert _routed_latent_quant_config(None) is None


def test_kimi_k3_situ_min_rows_pads_singleton_without_exposing_padding():
    hidden_states = torch.tensor(
        [[0.25, -0.5, 0.75, -1.0]],
        dtype=torch.bfloat16,
    )
    tanh_shapes = []
    native_tanh = torch.tanh

    def record_tanh_shape(inputs: torch.Tensor) -> torch.Tensor:
        tanh_shapes.append(tuple(inputs.shape))
        return native_tanh(inputs)

    with (
        patch.dict(os.environ, {"VLLM_ASCEND_KIMI_SITU_MIN_ROWS": "2"}),
        patch("vllm_ascend.ops.activation.torch.tanh", side_effect=record_tanh_shape),
    ):
        output = situ_and_mul(hidden_states, beta=4.0, linear_beta=25.0)

    assert output.shape == (1, 2)
    assert tanh_shapes == [(2, 2), (2, 2)]


def test_kimi_k3_situ_min_rows_leaves_two_rows_unpadded():
    hidden_states = torch.zeros((2, 4), dtype=torch.bfloat16)
    with patch.dict(os.environ, {"VLLM_ASCEND_KIMI_SITU_MIN_ROWS": "2"}):
        output = situ_and_mul(hidden_states, beta=4.0, linear_beta=25.0)

    assert output.shape == (2, 2)


def test_kimi_k3_reference_rms_norm_casts_before_weight():
    hidden_states = torch.tensor(
        [[0.75, -1.25, 2.0, -0.5]],
        dtype=torch.bfloat16,
    )
    norm = _KimiReferenceRMSNorm(4, eps=1e-6).to(dtype=torch.bfloat16)
    norm.weight.data.copy_(torch.tensor([0.5, 1.5, -0.75, 2.0], dtype=torch.bfloat16))

    normalized = hidden_states.float()
    normalized *= torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + 1e-6)
    expected = norm.weight * normalized.to(torch.bfloat16)

    assert torch.equal(norm(hidden_states), expected)


def test_kimi_k3_decomposed_mla_prefill_matches_rowwise_reference():
    generator = torch.Generator().manual_seed(20260907)
    q_nope = torch.randn(7, 2, 4, generator=generator, dtype=torch.bfloat16)
    q_pe = torch.randn(7, 2, 2, generator=generator, dtype=torch.bfloat16)
    k_nope = torch.randn(7, 2, 4, generator=generator, dtype=torch.bfloat16)
    k_pe = torch.randn(7, 2, 2, generator=generator, dtype=torch.bfloat16)
    value = torch.randn(7, 2, 3, generator=generator, dtype=torch.bfloat16)
    impl = SimpleNamespace(num_heads=2, v_head_dim=3, scale=6**-0.5)
    metadata = SimpleNamespace(prefill=SimpleNamespace(actual_seq_lengths_q=[3, 7]))

    expected = _reference_mla_forward_prefill(
        impl,
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        value,
        (),
        metadata,
    )
    actual = _decomposed_mla_forward_prefill(
        impl,
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        value,
        (),
        metadata,
    )

    assert torch.equal(actual, expected)


class _ElementwiseKVProjection:
    def __call__(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        first = inputs[..., :1]
        second = inputs[..., 1:2]
        return (
            torch.cat(
                (
                    first,
                    second,
                    first + second,
                    first - second,
                    first * 0.5,
                    second * 0.5,
                    -first,
                    -second,
                    first * 2.0,
                    second * 2.0,
                ),
                dim=-1,
            ),
            None,
        )


def _make_reduced_mla_decode_fixture():
    generator = torch.Generator().manual_seed(20260909)
    impl = SimpleNamespace(
        num_kv_heads=1,
        num_heads=2,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=1,
        v_head_dim=3,
        scale=3**-0.5,
        kv_b_proj=_ElementwiseKVProjection(),
    )
    block_size = 4
    k_latent_cache = torch.randn(
        3,
        block_size,
        1,
        impl.kv_lora_rank,
        generator=generator,
        dtype=torch.bfloat16,
    )
    k_pe_cache = torch.randn(
        3,
        block_size,
        1,
        impl.qk_rope_head_dim,
        generator=generator,
        dtype=torch.bfloat16,
    )
    q_nope = torch.randn(
        2,
        impl.num_heads,
        impl.qk_nope_head_dim,
        generator=generator,
        dtype=torch.bfloat16,
    )
    q_pe = torch.randn(
        2,
        impl.num_heads,
        impl.qk_rope_head_dim,
        generator=generator,
        dtype=torch.bfloat16,
    )
    block_table = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32)
    return impl, q_nope, q_pe, k_latent_cache, k_pe_cache, block_size, block_table


def test_kimi_k3_decomposed_mla_chunked_prefill_reads_cached_context():
    (
        impl,
        q_nope,
        q_pe,
        k_latent_cache,
        k_pe_cache,
        block_size,
        block_table,
    ) = _make_reduced_mla_decode_fixture()
    context_length = 3
    current_k_nope = torch.tensor(
        [
            [[0.25, -0.5], [0.75, 1.0]],
            [[-1.0, 0.5], [0.125, -0.25]],
        ],
        dtype=torch.bfloat16,
    )
    current_k_pe = torch.tensor(
        [[[0.5], [-0.75]], [[1.0], [0.25]]],
        dtype=torch.bfloat16,
    )
    current_value = torch.tensor(
        [
            [[0.5, -0.25, 0.75], [1.0, 0.25, -0.5]],
            [[-0.75, 0.5, 0.25], [0.125, -1.0, 0.5]],
        ],
        dtype=torch.bfloat16,
    )
    metadata = SimpleNamespace(
        prefill=SimpleNamespace(
            actual_seq_lengths_q=[2],
            context_lens=torch.tensor([context_length + 2], dtype=torch.int32),
            block_table=block_table[:1],
        )
    )

    expected = _reference_mla_forward_prefill(
        impl,
        q_nope,
        q_pe,
        current_k_nope,
        current_k_pe,
        current_value,
        (k_latent_cache, k_pe_cache),
        metadata,
    )
    actual = _decomposed_mla_forward_prefill(
        impl,
        q_nope,
        q_pe,
        current_k_nope,
        current_k_pe,
        current_value,
        (k_latent_cache, k_pe_cache),
        metadata,
    )

    assert actual.shape == (2, impl.num_heads * impl.v_head_dim)
    assert torch.equal(actual, expected)


def test_kimi_k3_decomposed_mla_decode_matches_rowwise_reference():
    (
        impl,
        q_nope,
        q_pe,
        k_latent_cache,
        k_pe_cache,
        block_size,
        block_table,
    ) = _make_reduced_mla_decode_fixture()
    sequence_lengths = [3, 6]
    metadata = SimpleNamespace(
        decode=SimpleNamespace(
            block_table=block_table,
            seq_lens_list=sequence_lengths,
            seq_lens_device=torch.tensor(sequence_lengths, dtype=torch.int32),
        )
    )

    expected = _reference_mla_forward_decode(
        impl,
        q_nope,
        q_pe,
        k_latent_cache,
        k_pe_cache,
        block_size,
        metadata,
    )
    actual = _decomposed_mla_forward_decode(
        impl,
        q_nope,
        q_pe,
        k_latent_cache,
        k_pe_cache,
        block_size,
        metadata,
    )

    assert torch.equal(actual, expected)


def test_kimi_k3_decomposed_mla_decode_zeros_padded_graph_row():
    (
        impl,
        q_nope,
        q_pe,
        k_latent_cache,
        k_pe_cache,
        block_size,
        block_table,
    ) = _make_reduced_mla_decode_fixture()
    graph_metadata = SimpleNamespace(
        decode=SimpleNamespace(
            block_table=block_table,
            seq_lens_list=[5, 0],
            seq_lens_device=torch.tensor([5, 0], dtype=torch.int32),
        )
    )
    reference_metadata = SimpleNamespace(
        decode=SimpleNamespace(
            block_table=block_table[:1],
            seq_lens_list=[5],
        )
    )

    expected = _reference_mla_forward_decode(
        impl,
        q_nope[:1],
        q_pe[:1],
        k_latent_cache,
        k_pe_cache,
        block_size,
        reference_metadata,
    )
    actual = _decomposed_mla_forward_decode(
        impl,
        q_nope,
        q_pe,
        k_latent_cache,
        k_pe_cache,
        block_size,
        graph_metadata,
    )

    assert torch.equal(actual[:1], expected)
    assert torch.equal(actual[1:], torch.zeros_like(actual[1:]))


def test_kimi_k3_vectorized_attention_residual_uses_explicit_fp32_reductions():
    prefix_sum = torch.tensor(
        [[0.5, -0.25, 1.0, 0.75], [1.25, 0.5, -0.75, 0.25]],
        dtype=torch.bfloat16,
    )
    block_residual = torch.tensor(
        [
            [[0.25, 0.5, -0.5, 1.0], [1.0, -0.75, 0.5, 0.25]],
            [[-0.5, 0.25, 0.75, 1.0], [0.5, 1.0, -0.25, -0.75]],
        ],
        dtype=torch.bfloat16,
    )
    projection = SimpleNamespace(weight=torch.tensor([[0.75, -0.5, 1.25, 0.25]], dtype=torch.bfloat16))
    norm = SimpleNamespace(
        weight=torch.tensor([1.0, 0.5, 1.5, -0.75], dtype=torch.bfloat16),
        variance_epsilon=1e-6,
    )
    rows = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    rows_float = rows.float()
    normalized = rows_float * torch.rsqrt(rows_float.square().mean(dim=-1, keepdim=True) + norm.variance_epsilon)
    score_weight = norm.weight.float() * projection.weight.squeeze(0).float()
    probabilities = torch.softmax((normalized * score_weight).sum(dim=-1), dim=-1)
    expected = (probabilities.unsqueeze(-1) * rows_float).sum(dim=-2).to(rows.dtype)

    with patch.dict(
        os.environ,
        {
            "VLLM_ASCEND_KIMI_REFERENCE_ATTN_RES": "1",
            "VLLM_ASCEND_KIMI_VECTORIZED_ATTN_RES": "1",
        },
    ):
        actual = _apply_attention_residual(
            prefix_sum,
            block_residual,
            projection,
            norm,
        )

    assert torch.equal(actual, expected)


def test_kimi_k3_rowwise_attention_residual_preserves_per_token_schedule():
    prefix_sum = torch.tensor(
        [[0.5, -0.25, 1.0, 0.75], [1.25, 0.5, -0.75, 0.25]],
        dtype=torch.bfloat16,
    )
    block_residual = torch.tensor(
        [
            [[0.25, 0.5, -0.5, 1.0], [1.0, -0.75, 0.5, 0.25]],
            [[-0.5, 0.25, 0.75, 1.0], [0.5, 1.0, -0.25, -0.75]],
        ],
        dtype=torch.bfloat16,
    )
    projection_weight = torch.tensor(
        [[0.75, -0.5, 1.25, 0.25]], dtype=torch.bfloat16
    )
    norm_weight = torch.tensor([1.0, 0.5, 1.5, -0.75], dtype=torch.bfloat16)
    variance_epsilon = 1e-6

    expected_rows = []
    rows = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    score_weight = norm_weight.float() * projection_weight.squeeze(0).float()
    for token_idx in range(rows.shape[0]):
        rows_float = rows[token_idx : token_idx + 1].float()
        normalized = rows_float * torch.rsqrt(
            rows_float.square().mean(dim=-1, keepdim=True) + variance_epsilon
        )
        probabilities = torch.softmax((normalized * score_weight).sum(dim=-1), dim=-1)
        expected_rows.append(
            (probabilities.unsqueeze(-1) * rows_float).sum(dim=-2).to(rows.dtype)
        )

    actual = _kimi_reference_rowwise_attention_residual_impl(
        prefix_sum,
        block_residual,
        projection_weight,
        norm_weight,
        variance_epsilon,
    )

    assert torch.equal(actual, torch.cat(expected_rows, dim=0))


def test_kimi_k3_rowwise_attention_residual_handles_empty_batch():
    actual = _kimi_reference_rowwise_attention_residual_impl(
        torch.empty((0, 4), dtype=torch.bfloat16),
        torch.empty((0, 2, 4), dtype=torch.bfloat16),
        torch.ones((1, 4), dtype=torch.bfloat16),
        torch.ones((4,), dtype=torch.bfloat16),
        1e-6,
    )

    assert actual.shape == (0, 4)
    assert actual.dtype == torch.bfloat16


def test_kimi_k3_reference_router_skips_discarded_bf16_projection():
    moe = KimiK3MoE.__new__(KimiK3MoE)
    nn.Module.__init__(moe)
    moe.hidden_size = 4
    moe.parity_tap_prefix = None
    moe._reference_router_weight_fp32 = None
    moe.gate = MagicMock()
    moe.gate.weight = torch.tensor(
        [[0.5, -1.0, 0.25, 2.0], [-0.75, 0.5, 1.5, -0.25]],
        dtype=torch.bfloat16,
    )
    moe.experts = MagicMock(return_value=torch.zeros(2, 4, dtype=torch.bfloat16))
    hidden_states = torch.tensor(
        [[1.0, -0.5, 0.25, 2.0], [-1.5, 0.75, 0.5, -0.25]],
        dtype=torch.bfloat16,
    )

    with patch.dict(os.environ, {"VLLM_ASCEND_KIMI_REFERENCE_ROUTER_FP32": "1"}):
        moe(hidden_states)

    moe.gate.assert_not_called()
    expected = torch.nn.functional.linear(hidden_states.float(), moe.gate.weight.float())
    assert torch.equal(moe.experts.call_args.kwargs["router_logits"], expected)


def test_kimi_k3_reference_router_caches_bf16_rounded_fp32_weight():
    moe = KimiK3MoE.__new__(KimiK3MoE)
    nn.Module.__init__(moe)
    moe.hidden_size = 4
    moe.parity_tap_prefix = None
    moe._reference_router_weight_fp32 = None
    moe.gate = MagicMock()
    moe.gate.weight = MagicMock()
    cached_weight = torch.tensor(
        [[0.5, -1.0, 0.25, 2.0], [-0.75, 0.5, 1.5, -0.25]],
        dtype=torch.float32,
    )
    moe.gate.weight.detach.return_value.float.return_value = cached_weight
    moe.experts = MagicMock(return_value=torch.zeros(2, 4, dtype=torch.bfloat16))
    hidden_states = torch.tensor(
        [[1.0, -0.5, 0.25, 2.0], [-1.5, 0.75, 0.5, -0.25]],
        dtype=torch.bfloat16,
    )
    router_logits = torch.zeros(2, 2, dtype=torch.float32)

    with (
        patch.dict(os.environ, {"VLLM_ASCEND_KIMI_REFERENCE_ROUTER_FP32": "1"}),
        patch("torch.nn.functional.linear", return_value=router_logits) as linear,
    ):
        moe(hidden_states)
        moe(hidden_states)

    moe.gate.assert_not_called()
    moe.gate.weight.detach.assert_called_once_with()
    assert moe._reference_router_weight_fp32 is cached_weight
    assert linear.call_count == 2
    assert all(call.args[1] is cached_weight for call in linear.call_args_list)


def test_kimi_k3_reference_router_uses_ascend_internal_schedule():
    moe = KimiK3MoE.__new__(KimiK3MoE)
    nn.Module.__init__(moe)
    moe.hidden_size = 4
    moe.parity_tap_prefix = None
    moe._use_internal_router_fp32 = True
    moe.gate = MagicMock()
    moe.experts = MagicMock(return_value=torch.zeros(2, 4, dtype=torch.bfloat16))
    moe.experts.is_internal_router = True
    hidden_states = torch.tensor(
        [[1.0, -0.5, 0.25, 2.0], [-1.5, 0.75, 0.5, -0.25]],
        dtype=torch.bfloat16,
    )

    with (
        patch.dict(os.environ, {"VLLM_ASCEND_KIMI_REFERENCE_ROUTER_FP32": "1"}),
        patch("torch.nn.functional.linear") as linear,
    ):
        output = moe(hidden_states)

    linear.assert_not_called()
    moe.gate.assert_not_called()
    internal_hidden_states = moe.experts.call_args.kwargs["hidden_states"]
    internal_router_placeholder = moe.experts.call_args.kwargs["router_logits"]
    assert torch.equal(internal_hidden_states, hidden_states)
    assert internal_hidden_states.data_ptr() == hidden_states.data_ptr()
    assert internal_router_placeholder is internal_hidden_states
    assert output.shape == hidden_states.shape


def test_kimi_k3_projector_registers_rotation_for_weight_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    class StubReplicatedLinear(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, hidden_states):
            return hidden_states, None

    monkeypatch.setattr(kimi_k3, "ReplicatedLinear", StubReplicatedLinear)
    monkeypatch.setattr(kimi_k3, "RMSNorm", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(kimi_k3, "get_act_fn", lambda *args, **kwargs: nn.Identity())
    config = KimiK3VisionConfig(
        mm_hidden_size=2,
        text_hidden_size=8,
        merge_kernel_size=(2, 2),
    )
    projector = KimiK3MultiModalProjector(config)

    assert projector.rot_proj is not None


@pytest.mark.parametrize(
    ("loaded_weights", "has_rot_proj"),
    [
        ({"mm_projector.rot_proj.weight"}, True),
        ({"mm_projector.linear_1.weight"}, False),
    ],
)
def test_kimi_k3_enables_projector_rotation_only_when_weight_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
    loaded_weights: set[str],
    has_rot_proj: bool,
):
    class StubLoader:
        def __init__(self, model, *, skip_prefixes):
            assert model is wrapper
            assert skip_prefixes == []

        def load_weights(self, weights, *, mapper):
            assert list(weights) == []
            assert mapper is wrapper.hf_to_vllm_mapper
            return loaded_weights

    monkeypatch.setattr(kimi_k3, "AutoWeightsLoader", StubLoader)
    wrapper = AscendKimiK3ForConditionalGeneration.__new__(AscendKimiK3ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.mm_projector = nn.Module()
    wrapper.mm_projector.rot_proj = nn.Linear(1, 1, bias=False)

    actual = wrapper.load_weights(iter(()))

    assert actual == loaded_weights
    assert hasattr(wrapper.mm_projector, "rot_proj") is has_rot_proj
    assert ("mm_projector.rot_proj.weight" in dict(wrapper.named_parameters())) is has_rot_proj


def test_kimi_k3_deletes_unused_rot_proj_when_projector_is_placeholder(
    monkeypatch: pytest.MonkeyPatch,
):
    # Text-only serving (--language-model-only, or --limit-mm-per-prompt at 0
    # for all tower modalities) wraps tower components in StageMissingLayer.
    # Its __getattr__ delegates to the wrapped projector, but del acts on the
    # placeholder's own registries (empty by design), so deleting
    # mm_projector.rot_proj directly raises AttributeError. The deletion must
    # target the wrapped module instead.
    class StubLoader:
        def __init__(self, model, *, skip_prefixes):
            assert model is wrapper
            assert skip_prefixes == []

        def load_weights(self, weights, *, mapper):
            assert list(weights) == []
            assert mapper is wrapper.hf_to_vllm_mapper
            return {"mm_projector.linear_1.weight"}

    monkeypatch.setattr(kimi_k3, "AutoWeightsLoader", StubLoader)
    wrapper = AscendKimiK3ForConditionalGeneration.__new__(AscendKimiK3ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    projector = nn.Module()
    projector.rot_proj = nn.Linear(1, 1, bias=False)
    wrapper.mm_projector = StageMissingLayer("vision_tower", projector)

    actual = wrapper.load_weights(iter(()))

    assert actual == {"mm_projector.linear_1.weight"}
    # The unused rotation was released from the wrapped projector...
    assert hasattr(projector, "rot_proj") is False
    # ...and lookups through the placeholder no longer find it either.
    assert hasattr(wrapper.mm_projector, "rot_proj") is False


def test_kimi_k3_projector_applies_rotation_only_after_weight_load():
    class PassthroughLinear(nn.Module):
        def forward(self, hidden_states):
            return hidden_states, None

    class ScaleLinear(nn.Module):
        def forward(self, hidden_states):
            return hidden_states * 2, None

    projector = KimiK3MultiModalProjector.__new__(KimiK3MultiModalProjector)
    nn.Module.__init__(projector)
    projector.input_size = 2
    projector.linear_1 = PassthroughLinear()
    projector.linear_2 = PassthroughLinear()
    projector.act = nn.Identity()
    projector.post_norm = nn.Identity()
    image_features = torch.tensor([[1.0, 2.0]])

    projector.rot_proj = ScaleLinear()
    del projector.rot_proj
    assert not hasattr(projector, "rot_proj")
    torch.testing.assert_close(projector(image_features), image_features)

    projector.rot_proj = ScaleLinear()
    torch.testing.assert_close(projector(image_features), image_features * 2)


@pytest.mark.parametrize(
    ("name", "params", "expected"),
    [
        (
            "layers.1.experts.w13_weight",
            {"layers.1.experts.w13_weight": object()},
            "layers.1.experts.w13_weight",
        ),
        (
            "layers.1.experts.w13_weight",
            {"layers.1.experts.w13_weight_packed": object()},
            "layers.1.experts.w13_weight_packed",
        ),
        (
            "layers.1.experts.w2_weight",
            {"layers.1.experts.w2_weight_packed": object()},
            "layers.1.experts.w2_weight_packed",
        ),
        (
            "layers.1.experts.w13_weight_scale",
            {"layers.1.experts.w13_weight_packed": object()},
            "layers.1.experts.w13_weight_scale",
        ),
    ],
)
def test_kimi_k3_resolves_packed_expert_checkpoint_names(
    name: str,
    params: dict[str, object],
    expected: str,
):
    assert _resolve_packed_expert_weight_name(name, params) == expected


def test_kimi_k3_config_normalizes_checkpoint_schema_for_vllm():
    """Cover only the non-pass-through checkpoint-to-vLLM adaptations."""
    config = KimiK3Config(
        text_config={"hidden_size": 4096},
        vision_config={
            "vt_num_attention_heads": 12,
            "vt_num_hidden_layers": 7,
            "vt_hidden_size": 1024,
            "vt_intermediate_size": 3584,
            "text_hidden_size": 1024,
        },
        use_unified_vision_chunk=True,
    )

    # Old plugin checkpoints used vision_chunk, while vLLM consumes image.
    assert not hasattr(config, "use_unified_vision_chunk")
    # MoonViT consumers use canonical names instead of checkpoint vt_* names.
    assert config.vision_config.num_attention_heads == 12
    assert config.vision_config.hidden_size == 1024
    # The projector output must follow the text model, not stale vision config.
    assert config.vision_config.text_hidden_size == config.text_config.hidden_size


def test_kimi_k3_model_uses_image_placeholder_from_upstream_contract():
    assert AscendKimiK3ForConditionalGeneration.get_placeholder_str("image", 0) == "<|kimi_image_placeholder|>"
    with pytest.raises(ValueError, match="does not support modality"):
        AscendKimiK3ForConditionalGeneration.get_placeholder_str(
            "vision_chunk",
            0,
        )


def test_kimi_k3_weight_mapper_adds_inner_language_model_prefix():
    mapper = AscendKimiK3ForConditionalGeneration.hf_to_vllm_mapper

    assert (
        mapper._map_name("language_model.layers.12.self_attn.q_proj.weight")
        == "language_model.model.layers.12.self_attn.q_proj.weight"
    )
    assert (
        mapper._map_name("language_model.model.layers.12.self_attn.q_proj.weight")
        == "language_model.model.layers.12.self_attn.q_proj.weight"
    )
    assert mapper._map_name("mm_projector.proj.0.weight") == "mm_projector.linear_1.weight"


@pytest.mark.parametrize(
    ("weight_name", "expected_layer"),
    [
        ("model.layers.93.self_attn.q_proj.weight", 93),
        ("layers.94.self_attn.q_proj.weight", 94),
        ("language_model.layers.93.mlp.gate_proj.weight", 93),
        ("language_model.model.layers.94.mlp.up_proj.weight", 94),
        ("model.layers.92.self_attn.q_proj.weight", None),
        ("model.layers.95.self_attn.q_proj.weight", None),
    ],
)
def test_kimi_k3_spec_layer_detection_accepts_loader_prefixes(
    weight_name: str,
    expected_layer: int | None,
):
    config = SimpleNamespace(
        num_hidden_layers=93,
        num_nextn_predict_layers=2,
    )

    assert get_spec_layer_idx_from_weight_name(config, weight_name) == expected_layer


def test_kimi_k3_spec_layer_detection_allows_missing_nextn_config():
    config = SimpleNamespace(num_hidden_layers=93)

    assert (
        get_spec_layer_idx_from_weight_name(
            config,
            "model.layers.93.self_attn.q_proj.weight",
        )
        is None
    )


def test_kimi_k3_vision_tp16_falls_back_to_data_parallel(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.models import vision as vision_utils

    class StubModule(nn.Module):
        pass

    qkv_kwargs: dict[str, object] = {}
    output_kwargs: dict[str, object] = {}

    def fake_qkv(*args, **kwargs):
        del args
        qkv_kwargs.update(kwargs)
        return StubModule()

    def fake_output(*args, **kwargs):
        del args
        output_kwargs.update(kwargs)
        return StubModule()

    monkeypatch.setattr(
        vision_utils,
        "get_tensor_model_parallel_world_size",
        lambda: 16,
    )
    monkeypatch.setattr(
        kimi_k3,
        "get_tensor_model_parallel_world_size",
        lambda: 16,
    )
    monkeypatch.setattr(
        kimi_k3,
        "KimiK3VisionMLP",
        lambda *args, **kwargs: StubModule(),
    )
    monkeypatch.setattr(kimi_k3, "get_act_fn", lambda name: nn.Identity())
    monkeypatch.setattr(kimi_k3, "QKVParallelLinear", fake_qkv)
    monkeypatch.setattr(kimi_k3, "RowParallelLinear", fake_output)
    monkeypatch.setattr(
        kimi_k3,
        "MMEncoderAttention",
        lambda *args, **kwargs: StubModule(),
    )

    layer = KimiK3VisionEncoderLayer(
        KimiK3VisionConfig(vt_num_attention_heads=12),
        quant_config=None,
        prefix="vision_tower.encoder.blocks.0",
    )

    assert layer.use_data_parallel is True
    assert layer.tp_size == 1
    assert layer.num_local_heads == 12
    assert qkv_kwargs["disable_tp"] is True
    assert output_kwargs["disable_tp"] is True


def test_kimi_k3_vit_dp_compat_calls_release_helper_without_num_heads(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[None] = []

    def release_helper():
        calls.append(None)
        return False

    monkeypatch.setattr(kimi_k3, "vllm_version_is", lambda version: version == "0.25.1")
    monkeypatch.setattr(kimi_k3, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(kimi_k3, "is_vit_use_data_parallel", release_helper)

    assert kimi_k3._is_vit_use_data_parallel(8) is False
    assert calls == [None]


def test_kimi_k3_vit_dp_compat_recreates_release_tp_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    def unexpected_release_helper():
        pytest.fail("The release helper must not run after the TP fallback")

    monkeypatch.setattr(kimi_k3, "vllm_version_is", lambda version: version == "0.25.1")
    monkeypatch.setattr(kimi_k3, "get_tensor_model_parallel_world_size", lambda: 16)
    monkeypatch.setattr(kimi_k3, "is_vit_use_data_parallel", unexpected_release_helper)

    assert kimi_k3._is_vit_use_data_parallel(12) is True


def test_kimi_k3_vit_dp_compat_passes_num_heads_to_main_helper(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def main_helper(num_heads):
        calls.append(num_heads)
        return True

    monkeypatch.setattr(kimi_k3, "vllm_version_is", lambda version: False)
    monkeypatch.setattr(kimi_k3, "is_vit_use_data_parallel", main_helper)

    assert kimi_k3._is_vit_use_data_parallel(12) is True
    assert calls == [12]


def test_kimi_k3_skips_explicit_move_for_meta_modules():
    module = nn.Linear(4, 4, device="meta")

    actual = _move_module_to_device(
        module,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert actual is module
    assert all(parameter.is_meta for parameter in module.parameters())


def test_kimi_k3_moves_non_meta_modules():
    module = nn.Linear(4, 4)

    actual = _move_module_to_device(
        module,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert actual is module
    assert all(parameter.device.type == "cpu" for parameter in module.parameters())
    assert all(parameter.dtype == torch.bfloat16 for parameter in module.parameters())


def test_kimi_k3_passes_situ_parameters_through_activation_config(monkeypatch):
    class StubModule(nn.Module):
        pass

    fused_moe_kwargs = {}

    def fake_replicated_linear(*args, **kwargs):
        return StubModule()

    def fake_fused_moe(**kwargs):
        fused_moe_kwargs.update(kwargs)
        return StubModule()

    monkeypatch.setattr(kimi_k3, "ReplicatedLinear", fake_replicated_linear)
    monkeypatch.setattr(kimi_k3, "FusedMoE", fake_fused_moe)
    config = SimpleNamespace(
        hidden_act="situ",
        hidden_size=32,
        routed_expert_hidden_size=16,
        num_shared_experts=0,
        num_experts=8,
        rms_norm_eps=1e-6,
        latent_moe_use_norm=False,
        moe_intermediate_size=12,
        num_experts_per_token=2,
        moe_renormalize=True,
        use_grouped_topk=True,
        num_expert_group=4,
        topk_group=2,
        moe_router_activation_func="sigmoid",
        routed_scaling_factor=2.5,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )

    KimiK3MoE(config, prefix="model.layers.1.block_sparse_moe")

    activation = fused_moe_kwargs["activation"]
    assert isinstance(activation, SituActivationConfig)
    assert activation.beta == 4.0
    assert activation.linear_beta == 25.0


def test_kimi_k3_reference_router_is_precast_and_passed_to_fused_moe(monkeypatch):
    class StubModule(nn.Module):
        pass

    fused_moe_kwargs = {}

    def fake_replicated_linear(*args, **kwargs):
        return StubModule()

    def fake_fused_moe(**kwargs):
        fused_moe_kwargs.update(kwargs)
        module = StubModule()
        module.is_internal_router = True
        return module

    monkeypatch.setattr(kimi_k3, "ReplicatedLinear", fake_replicated_linear)
    monkeypatch.setattr(kimi_k3, "FusedMoE", fake_fused_moe)
    config = SimpleNamespace(
        hidden_act="situ",
        hidden_size=32,
        routed_expert_hidden_size=16,
        num_shared_experts=0,
        num_experts=8,
        rms_norm_eps=1e-6,
        latent_moe_use_norm=False,
        moe_intermediate_size=12,
        num_experts_per_token=2,
        moe_renormalize=True,
        use_grouped_topk=True,
        num_expert_group=4,
        topk_group=2,
        moe_router_activation_func="sigmoid",
        routed_scaling_factor=2.5,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )

    with patch.dict(os.environ, {"VLLM_ASCEND_KIMI_REFERENCE_ROUTER_FP32": "1"}):
        moe = KimiK3MoE(config, prefix="model.layers.1.block_sparse_moe")

    assert moe._use_internal_router_fp32 is True
    assert moe.gate.precast_fp32_weight is True
    assert fused_moe_kwargs["gate"] is moe.gate


def test_kimi_k3_dense_mlp_uses_callable_situ(monkeypatch):
    class StubLinear(nn.Module):
        def forward(self, hidden_states):
            return hidden_states, None

    monkeypatch.setattr(kimi_k3, "MergedColumnParallelLinear", lambda *args, **kwargs: StubLinear())
    monkeypatch.setattr(kimi_k3, "RowParallelLinear", lambda *args, **kwargs: StubLinear())
    config = SimpleNamespace(
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )

    mlp = KimiK3MLP(config, hidden_size=4, intermediate_size=2)
    hidden_states = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    output = mlp(hidden_states)

    assert isinstance(mlp.act_fn, AscendSituAndMul)
    assert output.shape == (1, 2)


def test_text_model_captures_materialized_dspark_aux_stream(monkeypatch: pytest.MonkeyPatch):
    residual_calls: list[tuple[int, torch.Tensor]] = []
    consumed_inputs: list[torch.Tensor] = []

    class Marker(nn.Module):
        def __init__(self, value: int) -> None:
            super().__init__()
            self.value = value

    class FakeLayer(nn.Module):
        def __init__(self, layer_idx: int) -> None:
            super().__init__()
            self.layer_idx = layer_idx
            self.self_attention_res_proj = Marker(layer_idx)
            self.self_attention_res_norm = nn.Identity()

        def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            block_residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del positions
            if block_residual.shape[1] > 0:
                hidden_states = kimi_k3._apply_attention_residual(
                    hidden_states,
                    block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
            consumed_inputs.append(hidden_states.clone())
            if block_residual.shape[1] == 0:
                block_residual = hidden_states.unsqueeze(1)
            return hidden_states + 10, block_residual

    def fake_attention_residual(
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        projection: Marker,
        norm: nn.Module,
    ) -> torch.Tensor:
        del block_residual, norm
        residual_calls.append((projection.value, hidden_states.clone()))
        return hidden_states + projection.value * 100

    monkeypatch.setattr(kimi_k3, "_apply_attention_residual", fake_attention_residual)
    monkeypatch.setattr(
        kimi_k3,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    model = KimiK3TextModel.__new__(KimiK3TextModel)
    nn.Module.__init__(model)
    model.do_not_compile = True
    model.start_layer = 0
    model.end_layer = 2
    model.layers = nn.ModuleList([FakeLayer(0), FakeLayer(1)])
    model.embed_input_ids = MagicMock(return_value=torch.tensor([[1.0]]))
    model.output_attn_res_proj = Marker(0)
    model.output_attn_res_norm = nn.Identity()
    model.norm = nn.Identity()
    model.dspark_aux_capture_materialized = True
    model._set_aux_hidden_state_layers((0, 1))

    hidden_states, aux_hidden_states = model(
        torch.tensor([1]),
        torch.tensor([0]),
        None,
    )

    torch.testing.assert_close(aux_hidden_states[0], consumed_inputs[0])
    torch.testing.assert_close(aux_hidden_states[1], consumed_inputs[1])
    torch.testing.assert_close(aux_hidden_states[0], torch.tensor([[1.0]]))
    torch.testing.assert_close(aux_hidden_states[1], torch.tensor([[111.0]]))
    torch.testing.assert_close(hidden_states, torch.tensor([[121.0]]))
    assert [layer_idx for layer_idx, _ in residual_calls] == [1, 1, 0]

    residual_calls.clear()
    consumed_inputs.clear()
    model.dspark_aux_capture_materialized = False
    model._set_aux_hidden_state_layers((1,))

    _, raw_aux_hidden_states = model(
        torch.tensor([1]),
        torch.tensor([0]),
        None,
    )

    torch.testing.assert_close(raw_aux_hidden_states[0], torch.tensor([[11.0]]))
    assert [layer_idx for layer_idx, _ in residual_calls] == [1, 0]


def test_kimi_k3_dspark_aux_capture_mode_is_forwarded():
    causal_model = AscendKimiK3ForCausalLM.__new__(AscendKimiK3ForCausalLM)
    nn.Module.__init__(causal_model)
    causal_model.model = SimpleNamespace(dspark_aux_capture_materialized=False)

    causal_model.set_dspark_aux_capture_materialized(True)

    assert causal_model.model.dspark_aux_capture_materialized is True

    wrapper = AscendKimiK3ForConditionalGeneration.__new__(AscendKimiK3ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.language_model = MagicMock()

    wrapper.set_dspark_aux_capture_materialized(True)

    wrapper.language_model.set_dspark_aux_capture_materialized.assert_called_once_with(True)
