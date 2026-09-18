# SPDX-License-Identifier: Apache-2.0
"""Shared routed-experts wire schema for Kimi K3 full R3 replay."""

from __future__ import annotations

from vllm.model_executor.layers.fused_moe import (
    routed_experts_capturer as _routed_experts_capturer,
)

KIMI_FULL_R3_PACK_FACTOR = 3
_KIMI_MODEL_TYPES = {"kimi_k3", "kimi_linear"}
_ORIGINAL_GET_TOPK = _routed_experts_capturer._get_num_experts_per_tok
_PATCHED = False


def _wire_width(hf_config) -> int:
    topk = int(_ORIGINAL_GET_TOPK(hf_config))
    model_type = str(getattr(hf_config, "model_type", "")).lower()
    return topk * KIMI_FULL_R3_PACK_FACTOR if model_type in _KIMI_MODEL_TYPES else topk


def install_kimi_full_r3_schema_patch() -> None:
    """Size worker and scheduler buffers for IDs plus two BF16 byte lanes."""
    global _PATCHED
    if _PATCHED:
        return
    _routed_experts_capturer._get_num_experts_per_tok = _wire_width
    _PATCHED = True
