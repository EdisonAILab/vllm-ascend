# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend.models.kimi_runtime import (
    configure_kimi_reduced_w4a8_runtime,
    kimi_reduced_w4a8_runtime_enabled,
    kimi_runtime_flag,
    kimi_runtime_int,
)


def _runtime_config(
    *,
    hidden_size: int = 1024,
    quant_format: str | None = "mxfp4-pack-quantized",
    quantization: str | None = None,
):
    text_config = SimpleNamespace(
        model_type="kimi_linear",
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=8,
        q_lora_rank=256,
        kv_lora_rank=128,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        num_experts=8,
        num_experts_per_token=2,
        linear_attn_config={"head_dim": 32, "num_heads": 8},
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=text_config,
            quantization=quantization,
            hf_config=SimpleNamespace(
                quantization_config=(
                    {"format": quant_format} if quant_format is not None else None
                ),
            ),
        )
    )


def test_reduced_w4a8_profile_supplies_defaults_without_environment_flags():
    try:
        with patch.dict("os.environ", {}, clear=True):
            assert configure_kimi_reduced_w4a8_runtime(_runtime_config())
            assert kimi_reduced_w4a8_runtime_enabled()
            assert kimi_runtime_flag("VLLM_ASCEND_KIMI_REFERENCE_ATTN_RES", reduced_default=True)
            assert not kimi_runtime_flag("VLLM_ASCEND_KIMI_REFERENCE_ROUTING", reduced_default=False)
            assert (
                kimi_runtime_int(
                    "VLLM_ASCEND_KIMI_SITU_MIN_ROWS",
                    reduced_default=2,
                )
                == 2
            )
    finally:
        configure_kimi_reduced_w4a8_runtime(_runtime_config(hidden_size=2048))


def test_explicit_environment_override_wins_over_reduced_default():
    try:
        configure_kimi_reduced_w4a8_runtime(_runtime_config())
        with patch.dict(
            "os.environ",
            {
                "VLLM_ASCEND_KIMI_REFERENCE_ATTN_RES": "0",
                "VLLM_ASCEND_KIMI_REFERENCE_ROUTING": "1",
                "VLLM_ASCEND_KIMI_SITU_MIN_ROWS": "7",
            },
            clear=False,
        ):
            assert not kimi_runtime_flag(
                "VLLM_ASCEND_KIMI_REFERENCE_ATTN_RES",
                reduced_default=True,
            )
            assert kimi_runtime_flag(
                "VLLM_ASCEND_KIMI_REFERENCE_ROUTING",
                reduced_default=False,
            )
            assert (
                kimi_runtime_int(
                    "VLLM_ASCEND_KIMI_SITU_MIN_ROWS",
                    reduced_default=2,
                )
                == 7
            )
    finally:
        configure_kimi_reduced_w4a8_runtime(_runtime_config(hidden_size=2048))


def test_vllm_normalized_compressed_tensors_config_activates_profile():
    try:
        assert configure_kimi_reduced_w4a8_runtime(
            _runtime_config(quant_format=None, quantization="compressed-tensors")
        )
    finally:
        configure_kimi_reduced_w4a8_runtime(_runtime_config(hidden_size=2048))


def test_full_size_or_non_mxfp4_model_retains_existing_defaults():
    assert not configure_kimi_reduced_w4a8_runtime(_runtime_config(hidden_size=2048))
    assert not kimi_runtime_flag(
        "VLLM_ASCEND_KIMI_REFERENCE_ATTN_RES",
        reduced_default=True,
    )
    assert (
        kimi_runtime_int(
            "VLLM_ASCEND_KIMI_SITU_MIN_ROWS",
            reduced_default=2,
        )
        == 0
    )
    assert not configure_kimi_reduced_w4a8_runtime(
        _runtime_config(quant_format="dense")
    )
