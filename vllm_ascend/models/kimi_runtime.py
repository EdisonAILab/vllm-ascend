# SPDX-License-Identifier: Apache-2.0
"""Runtime defaults for the architecture-preserving reduced Kimi K3 model."""

from __future__ import annotations

import os
from typing import Any


_reduced_w4a8_enabled = False


def _value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def configure_kimi_reduced_w4a8_runtime(vllm_config: Any) -> bool:
    """Enable safe defaults only for the sealed reduced W4A8 architecture.

    Layer count is deliberately not part of the signature, allowing the same
    reduced tensor dimensions to be exercised with more or fewer layers. Full
    Kimi K3 dimensions and non-MXFP4 checkpoints retain their existing paths.
    """
    global _reduced_w4a8_enabled

    model_config = _value(vllm_config, "model_config")
    text_config = _value(model_config, "hf_text_config")
    hf_config = _value(model_config, "hf_config")
    quantization_config = _value(hf_config, "quantization_config", {}) or {}
    linear_attn_config = _value(text_config, "linear_attn_config", {}) or {}
    quantization = _value(model_config, "quantization")
    is_mxfp4 = (
        _value(quantization_config, "format") == "mxfp4-pack-quantized"
        or quantization == "compressed-tensors"
    )
    _reduced_w4a8_enabled = bool(
        is_mxfp4
        and _value(text_config, "model_type") == "kimi_linear"
        and _value(text_config, "hidden_size") == 1024
        and _value(text_config, "num_attention_heads") == 8
        and _value(text_config, "q_lora_rank") == 256
        and _value(text_config, "kv_lora_rank") == 128
        and _value(text_config, "qk_nope_head_dim") == 64
        and _value(text_config, "qk_rope_head_dim") == 32
        and _value(text_config, "v_head_dim") == 64
        and _value(text_config, "num_experts") == 8
        and _value(text_config, "num_experts_per_token") == 2
        and _value(linear_attn_config, "head_dim") == 32
        and _value(linear_attn_config, "num_heads") == 8
    )
    return _reduced_w4a8_enabled


def kimi_reduced_w4a8_runtime_enabled() -> bool:
    return _reduced_w4a8_enabled


def kimi_runtime_flag(name: str, *, reduced_default: bool) -> bool:
    """Return an explicit environment override or the reduced-model default."""
    override = os.environ.get(name)
    if override is not None:
        return bool(int(override))
    return _reduced_w4a8_enabled and reduced_default


def kimi_runtime_int(name: str, *, reduced_default: int, default: int = 0) -> int:
    """Return an explicit integer override or a reduced-model default."""
    override = os.environ.get(name)
    if override is not None:
        return int(override)
    return reduced_default if _reduced_w4a8_enabled else default
