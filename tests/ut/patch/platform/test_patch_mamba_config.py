# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.patch.platform import patch_mamba_config as patch


class _KimiHybridModel:
    @staticmethod
    def get_mamba_state_shape_from_config(vllm_config):
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        local_heads = 8 // tp_size
        local_conv_dim = 3 * 8 * 32 // tp_size
        return ((local_conv_dim, 3), (local_heads, 32, 32))

    @staticmethod
    def get_mamba_state_dtype_from_config(_vllm_config):
        return (torch.bfloat16, torch.float32)


def _config(tp_size: int):
    text_config = SimpleNamespace(kv_lora_rank=128, qk_rope_head_dim=32)
    model_config = SimpleNamespace(
        architecture="KimiK3ForConditionalGeneration",
        dtype=torch.bfloat16,
        hf_text_config=text_config,
        max_model_len=320,
        use_mla=True,
        get_num_kv_heads=lambda _parallel_config: 1,
    )
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=None,
            cache_dtype="auto",
            enable_prefix_caching=False,
            mamba_block_size=None,
            mamba_cache_mode="none",
            mamba_page_size_padded=None,
        ),
        kv_transfer_config=None,
        model_config=model_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
    )


@pytest.mark.parametrize("tp_size", (1, 2, 4, 8))
def test_kimi_tp_allows_attention_page_larger_than_sharded_ssm(
    monkeypatch,
    tp_size: int,
):
    monkeypatch.setattr(
        patch.MambaModelConfig,
        "verify_and_update_config",
        classmethod(lambda _cls, _config: None),
    )
    monkeypatch.setattr(
        patch.ModelRegistry,
        "resolve_model_cls",
        lambda *_args, **_kwargs: (_KimiHybridModel, None),
    )

    config = _config(tp_size)
    patch.verify_and_update_config.__func__(None, config)

    assert config.cache_config.block_size == 128
    assert config.cache_config.mamba_page_size_padded is not None
    assert config.cache_config.mamba_block_size == config.model_config.max_model_len
