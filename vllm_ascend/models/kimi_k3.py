#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

"""Native multimodal Kimi K3 model for vLLM-Ascend."""

import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from types import MethodType, SimpleNamespace
from typing import Annotated, Any, Literal, cast

import numpy as np
import torch
import torch_npu
from torch import nn
from transformers import BatchFeature
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions, ImageDummyOptions
from vllm.distributed import divide, get_pp_group, get_tensor_model_parallel_world_size
from vllm.inputs import MultiModalDataDict
from vllm.logger import logger
from vllm.model_executor.layers.activation import SiluAndMul, get_act_fn
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.fused_moe import FusedMoE, fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import KimiGatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
)
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors import compressed_tensors
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.deepseek_v2 import DeepSeekV2FusedQkvAProjLinear
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.kimi_k25_vit import (
    Learnable2DInterpPosEmbDivided_fixed,
    Rope2DPosEmbRepeated,
    apply_rope,
    tpool_patch_merger,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    init_vllm_registered_model,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel, run_dp_sharded_mrope_vision_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import ImageProcessorItems, ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.triton_utils import HAS_TRITON
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.models.kimi_runtime import (
    configure_kimi_reduced_w4a8_runtime,
    kimi_runtime_flag,
)
from vllm_ascend.ops.activation import AscendSituAndMul, SituActivationConfig
from vllm_ascend.ops.kimi_kda import uses_kimi_k3_global_inputs_embeds
from vllm_ascend.ops.kimi_kda_state import kimi_kda_state_shape
from vllm_ascend.transformers_utils.configs.kimi_k3 import KimiK3Config, KimiK3TextConfig, KimiK3VisionConfig
from vllm_ascend.transformers_utils.processors.kimi_k3 import KimiK3Processor
from vllm_ascend.utils import vllm_version_is

apply_attn_res: (
    Callable[
        [torch.Tensor, torch.Tensor, nn.Module, nn.Module],
        torch.Tensor,
    ]
    | None
) = None
if HAS_TRITON:
    from vllm_ascend.ops.triton.kimi_k3.attention_residual import apply_attn_res as triton_apply_attn_res

    apply_attn_res = triton_apply_attn_res


_KIMI_MLAPO_KV_LORA_RANK = 512
_KIMI_MLA_KERNEL_ROPE_DIM = 64


def _kimi_reference_rowwise_attention_residual_impl(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    """Preserve Megatron's token-by-token FP32 reduction schedule.

    This implementation is intentionally row-wise. Combining tokens into a
    single reduction changes a few BF16-rounded activations on Ascend even
    though the mathematical expression is the same. Registering it as a
    custom op keeps the dynamic token loop opaque to ``torch.compile`` while
    retaining the exact eager arithmetic at runtime.
    """
    rows = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    if rows.shape[0] == 0:
        return torch.empty_like(prefix_sum)

    score_weight = norm_weight.float() * projection_weight.squeeze(0).float()
    outputs = []
    for token_idx in range(rows.shape[0]):
        rows_float = rows[token_idx : token_idx + 1].float()
        normalized = rows_float * torch.rsqrt(
            rows_float.square().mean(dim=-1, keepdim=True) + variance_epsilon
        )
        scores = (normalized * score_weight).sum(dim=-1)
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(
            (probabilities.unsqueeze(-1) * rows_float).sum(dim=-2).to(rows.dtype)
        )
    return torch.cat(outputs, dim=0)


def _kimi_reference_rowwise_attention_residual_fake(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    return torch.empty_like(prefix_sum)


direct_register_custom_op(
    op_name="kimi_reference_rowwise_attention_residual",
    op_func=_kimi_reference_rowwise_attention_residual_impl,
    fake_impl=_kimi_reference_rowwise_attention_residual_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


def _configure_kimi_mlapo_shape(mla_impl: Any, kv_lora_rank: int) -> bool:
    """Fall back from the fixed-shape A5 MLAPO prolog for reduced Kimi models."""
    if not mla_impl.enable_mlapo or kv_lora_rank == _KIMI_MLAPO_KV_LORA_RANK:
        return False
    mla_impl.enable_mlapo = False
    logger.warning_once(
        "Kimi MLAPO requires kv_lora_rank=%d; using the standard MLA "
        "preprocess path for reduced kv_lora_rank=%d.",
        _KIMI_MLAPO_KV_LORA_RANK,
        kv_lora_rank,
    )
    return True


def _parity_tap(name: str, tensor: torch.Tensor) -> None:
    """Persist opt-in prefill tensors without changing the production path."""
    output_dir = os.environ.get("KIMI_PARITY_TAP_DIR")
    if not output_dir:
        return
    include = os.environ.get("KIMI_PARITY_TAP_INCLUDE")
    if include:
        prefixes = tuple(value.strip() for value in include.split(",") if value.strip())
        if not prefixes or not name.startswith(prefixes):
            return
    expected_tokens = int(os.environ.get("KIMI_PARITY_TAP_EXPECTED_TOKENS", "32"))
    # vLLM prunes the prefill hidden states to the final sampling row before
    # compute_logits. Keep that one-row output so it can be compared with the
    # last row of Megatron's full-sequence logits.
    expected_rows = (expected_tokens, 1) if name == "11_logits" else (expected_tokens,)
    if tensor.ndim == 0 or tensor.shape[0] not in expected_rows:
        return
    os.makedirs(output_dir, exist_ok=True)
    torch.save(tensor.detach().cpu().contiguous(), os.path.join(output_dir, f"{name}.pt"))


def _routed_latent_quant_config(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    """Quantize latent MoE projections when their checkpoint weights are packed."""
    if quant_config is not None and quant_config.get_name() in {"ascend", "compressed-tensors"}:
        return quant_config
    return None


def _resolve_packed_expert_weight_name(
    name: str,
    params_dict: Mapping[str, object],
) -> str:
    """Map packed checkpoint weights to the parameter name used by the scheme."""
    if name in params_dict or not name.endswith("_weight"):
        return name
    packed_name = f"{name}_packed"
    return packed_name if packed_name in params_dict else name


def _is_vit_use_data_parallel(num_heads: int) -> bool:
    """Keep vision TP fallback compatible with the vLLM release branch."""
    if not vllm_version_is("0.25.1"):
        return is_vit_use_data_parallel(num_heads)

    # TODO: Remove this branch when vLLM 0.25.1 support is dropped.
    if num_heads % get_tensor_model_parallel_world_size() != 0:
        logger.warning_once(
            "The number of vision attention heads is not divisible by "
            "the tensor parallel size. Falling back to data parallelism "
            "for the vision encoder."
        )
        return True
    return is_vit_use_data_parallel()


def _move_module_to_device(
    module: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype | None,
) -> nn.Module:
    """Move materialized modules while preserving lazy meta parameters.

    Some quantization methods deliberately create parameters on the meta
    device. The model loader records and materializes those parameters after
    model construction, so calling ``Module.to`` here would fail before the
    loader gets that opportunity. Non-meta modules retain the explicit move
    used by the upstream Kimi multimodal implementation.
    """
    tensors = (*module.parameters(), *module.buffers())
    if any(tensor.is_meta for tensor in tensors):
        return module
    return module.to(device=device, dtype=dtype)


def navit_resize_image(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: int | None,
) -> dict[str, int]:
    """Mirror the checkpoint's NaViT resize and media-token calculation."""

    scale_by_total = math.sqrt(in_patch_limit / (max(1.0, width // patch_size) * max(1.0, height // patch_size)))
    scale_by_width = patch_limit_on_one_side * patch_size / width
    scale_by_height = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, scale_by_total, scale_by_width, scale_by_height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    max_side = patch_limit_on_one_side * patch_size
    new_width = min(new_width, max_side)
    new_height = min(new_height, max_side)

    factor = merge_kernel_size * patch_size
    pad_height = (factor - new_height % factor) % factor
    pad_width = (factor - new_width % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = fixed_output_tokens
    else:
        token_height = (new_height + pad_height) // factor
        token_width = (new_width + pad_width) // factor
        if token_height * merge_kernel_size > patch_limit_on_one_side:
            raise ValueError("Kimi K3 resized image exceeds the height patch limit")
        if token_width * merge_kernel_size > patch_limit_on_one_side:
            raise ValueError("Kimi K3 resized image exceeds the width patch limit")
        num_tokens = token_height * token_width

    return {
        "num_tokens": num_tokens,
        "new_width": new_width,
        "new_height": new_height,
        "pad_width": pad_width,
        "pad_height": pad_height,
        "sampled_nframes": 1,
    }


class KimiK3MediaPixelInputs(TensorSchema):
    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("np", 3, "ps", "ps"),
    ]
    grid_thws: Annotated[torch.Tensor, TensorShape("nm", 3)]


class KimiK3ProcessingInfo(BaseProcessingInfo):
    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)
        self.hf_config = self.get_hf_config()
        tokenizer = self.get_tokenizer()
        image_processor = cached_get_image_processor(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
        )
        config_token_id = self.hf_config.media_placeholder_token_id
        resolved_token_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        valid_resolved_id = isinstance(resolved_token_id, int) and (
            unk_token_id is None or resolved_token_id != unk_token_id
        )
        if valid_resolved_id and resolved_token_id != config_token_id:
            logger.warning_once(
                "Kimi-K3 config.media_placeholder_token_id (%d) disagrees "
                "with tokenizer mapping for <|media_pad|> (%d). "
                "Using tokenizer value.",
                config_token_id,
                resolved_token_id,
            )
            self.hf_config.media_placeholder_token_id = resolved_token_id
            self.media_token_id = resolved_token_id
        else:
            self.media_token_id = config_token_id
        self.hf_config.media_placeholder_token_id = self.media_token_id
        self.media_token = tokenizer.decode(self.media_token_id)
        self.image_processor = image_processor
        self.hf_processor = KimiK3Processor(image_processor, tokenizer)
        self.media_tokens_calculator = image_processor.media_tokens_calculator

    def get_hf_processor(self, **kwargs: object) -> KimiK3Processor:
        del kwargs
        return self.hf_processor

    def get_hf_config(self) -> KimiK3Config:
        return self.ctx.get_hf_config(KimiK3Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    @classmethod
    def get_max_image_size(
        cls,
        patch_size: int,
        merge_kernel_size: int,
        in_patch_limit: int,
        patch_limit_on_one_side: int,
        fixed_output_tokens: int | None,
    ) -> ImageSize:
        max_side = patch_limit_on_one_side * patch_size
        best_score = (-1, -1)
        best_size = (max_side, max_side)

        for width_patches in range(patch_limit_on_one_side + 1):
            width = min((width_patches + 1) * patch_size - 1, max_side)
            for height_patches in range(
                width_patches,
                patch_limit_on_one_side + 1,
            ):
                height = min((height_patches + 1) * patch_size - 1, max_side)
                resize_config = navit_resize_image(
                    width,
                    height,
                    patch_size,
                    merge_kernel_size,
                    in_patch_limit,
                    patch_limit_on_one_side,
                    fixed_output_tokens,
                )
                padded_width = resize_config["new_width"] + resize_config["pad_width"]
                padded_height = resize_config["new_height"] + resize_config["pad_height"]
                num_patches = padded_width // patch_size * (padded_height // patch_size)
                score = (resize_config["num_tokens"], num_patches)
                if score > best_score:
                    best_score = score
                    best_size = (width, height)

        return ImageSize(width=best_size[0], height=best_size[1])


class KimiK3DummyInputsBuilder(BaseDummyInputsBuilder[KimiK3ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.info.get_hf_config().image_placeholder * mm_counts.get(
            "image",
            0,
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        del seq_len
        media_proc_cfg = self.info.image_processor.media_proc_cfg
        max_size = self.info.get_max_image_size(
            media_proc_cfg["patch_size"],
            media_proc_cfg["merge_kernel_size"],
            media_proc_cfg["in_patch_limit"],
            media_proc_cfg["patch_limit_on_one_side"],
            media_proc_cfg["fixed_output_tokens"],
        )
        image_overrides = cast(
            ImageDummyOptions | None,
            mm_options.get("image"),
        )
        return {
            "image": self._get_dummy_images(
                height=max_size.height,
                width=max_size.width,
                num_images=mm_counts.get("image", 0),
                overrides=image_overrides,
            ),
        }


class KimiK3MultiModalProcessor(BaseMultiModalProcessor[KimiK3ProcessingInfo]):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_processor_mm_kwargs
        grid_thws = hf_inputs.get("grid_thws", torch.empty((0, 3)))
        grid_sizes = grid_thws.prod(-1)
        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes(
                "image",
                grid_sizes,
            ),
            "grid_thws": MultiModalFieldConfig.batched(
                "image",
                keep_on_cpu=True,
            ),
        }

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Always route through KimiK3Processor, which wraps bare images into
        # the media dictionaries expected by the checkpoint processor.
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        del (
            prompt_text,
            mm_items,
            hf_processor_mm_kwargs,
            tokenization_kwargs,
        )
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        del hf_processor_mm_kwargs, out_mm_kwargs
        media_token_id = self.info.media_token_id
        media_token = self.info.media_token
        image_placeholder = self.info.get_hf_config().image_placeholder

        def replacement(item_idx: int) -> PromptUpdateDetails[str]:
            images = mm_items.get_items("image", (ImageProcessorItems,))
            image = images.get(item_idx)
            if image is None:
                raise ValueError(f"Missing Kimi K3 image at index {item_idx}")
            num_media_tokens = self.info.media_tokens_calculator({"type": "image", "image": image})
            width, height = images.get_image_size(item_idx)
            full = (
                f"<|media_begin|>image {width}x{height}<|media_content|>{media_token * num_media_tokens}<|media_end|>"
            )
            return PromptUpdateDetails.select_token_id(full, media_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=image_placeholder,
                replacement=replacement,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    KimiK3MultiModalProcessor,
    info=KimiK3ProcessingInfo,
    dummy_inputs=KimiK3DummyInputsBuilder,
)
class AscendKimiK3ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
):
    supports_encoder_tp_data = True
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.layers.": "language_model.model.layers.",
            "mm_projector.proj.0": "mm_projector.linear_1",
            "mm_projector.proj.2": "mm_projector.linear_2",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        if modality == "image":
            return "<|kimi_image_placeholder|>"
        raise ValueError(f"Kimi K3 does not support modality: {modality}")

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiK3Config = model_config.hf_config
        self.config = config
        self.model_config = model_config
        self.quant_config = vllm_config.quant_config
        self.hidden_size = config.text_config.hidden_size
        self.device = current_platform.current_device()
        self.use_data_parallel = _is_vit_use_data_parallel(config.vision_config.num_attention_heads)
        vision_quant = self._maybe_ignore_quant_config(self.quant_config)

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = KimiK3VisionTower(
                config.vision_config,
                quant_config=vision_quant,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            tower_dtype = model_config.dtype if vision_quant is None else None
            self.vision_tower = _move_module_to_device(
                self.vision_tower,
                device=self.device,
                dtype=tower_dtype,
            )
            self.mm_projector = KimiK3MultiModalProjector(
                config.vision_config,
                quant_config=vision_quant,
                prefix=maybe_prefix(prefix, "mm_projector"),
            )
            self.mm_projector = _move_module_to_device(
                self.mm_projector,
                device=self.device,
                dtype=model_config.dtype,
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["KimiK3ForCausalLM"],
            )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
        self.media_placeholder = config.media_placeholder_token_id

    @staticmethod
    def _maybe_ignore_quant_config(
        quant_config: QuantizationConfig | None,
    ) -> QuantizationConfig | None:
        if quant_config is not None and (
            isinstance(quant_config, compressed_tensors.CompressedTensorsConfig)
            or quant_config.get_name() == "compressed-tensors"
        ):
            return None
        return quant_config

    def _parse_and_validate_media_input(self, **kwargs: object) -> KimiK3MediaPixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        grid_thws = kwargs.pop("grid_thws", None)
        if pixel_values is None:
            return None
        if isinstance(pixel_values, list):
            pixel_tensors: list[torch.Tensor] = []
            for pixel_value in pixel_values:
                if not isinstance(pixel_value, torch.Tensor):
                    raise TypeError(f"pixel_values entries must be tensors, got {type(pixel_value)}")
                pixel_tensors.append(pixel_value)
            pixel_values = torch.cat(pixel_tensors, dim=0)
        elif not isinstance(pixel_values, torch.Tensor):
            raise TypeError(f"pixel_values must be a tensor or list of tensors, got {type(pixel_values)}")
        if pixel_values.ndim in (3, 5):
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0] * pixel_values.shape[1],
                *pixel_values.shape[2:],
            )
        target_dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(target_dtype)
        if not isinstance(grid_thws, torch.Tensor):
            raise TypeError(f"grid_thws must be a tensor, got {type(grid_thws)}")
        grid_thws_tensor: torch.Tensor = grid_thws.reshape(-1, grid_thws.shape[-1])
        if grid_thws_tensor.ndim != 2 or grid_thws_tensor.shape[1] != 3:
            raise ValueError(f"Unexpected Kimi K3 grid_thws shape: {grid_thws_tensor.shape}")
        return KimiK3MediaPixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            grid_thws=grid_thws_tensor,
        )

    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
        media_input = self._parse_and_validate_media_input(**kwargs)
        if media_input is None:
            return None
        return vision_tower_forward(
            self.vision_tower,
            media_input["pixel_values"],
            media_input["grid_thws"],
            self.mm_projector,
            self.use_data_parallel,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        del kwargs
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor | None:
        del kwargs
        return self.language_model.compute_logits(hidden_states)

    def set_dspark_aux_capture_materialized(self, enabled: bool) -> None:
        self.language_model.set_dspark_aux_capture_materialized(enabled)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return AscendKimiK3ForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return AscendKimiK3ForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return AscendKimiK3ForCausalLM.get_mamba_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ModelSlim checkpoints may include an explicit projector rotation.
        # Build the optional layer before loading so streaming checkpoint
        # iterators can populate it, then release it when the weight is absent.
        rot_proj = getattr(self.mm_projector, "rot_proj", None)
        skip_prefixes = [] if rot_proj is not None else ["mm_projector.rot_proj."]
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        rot_proj_weight_names = (
            {name for name, _ in rot_proj.named_parameters(prefix="mm_projector.rot_proj")}
            if rot_proj is not None
            else set()
        )
        loaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        if rot_proj is not None and rot_proj_weight_names.isdisjoint(loaded_weights):
            # StageMissingLayer.__getattr__ delegates to the wrapped module, so
            # rot_proj above is the real one, but del must act on the module
            # that actually holds the registration. Unwrap the placeholder
            # (language-model-only / zero mm limit deployments) before deleting.
            # cast: the runtime value is the projector or its StageMissingLayer
            # wrapper, both nn.Module; the getattr default form confuses mypy.
            target = cast(nn.Module, getattr(self.mm_projector, "module", self.mm_projector))
            if "rot_proj" in target._modules:
                del target.rot_proj
        return loaded_weights


class KimiK3MLP(nn.Module):
    def __init__(
        self,
        config: KimiK3TextConfig,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.parity_tap_prefix = (
            "02_moe_shared" if "layers.1.block_sparse_moe" in prefix and "shared_experts" in prefix else None
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if config.hidden_act == "situ":
            self.act_fn = AscendSituAndMul(
                beta=config.activation_situ_beta or 1.0,
                linear_beta=config.activation_situ_linear_beta,
            )
        elif config.hidden_act == "silu":
            self.act_fn = SiluAndMul()
        else:
            raise ValueError(f"Unsupported Kimi K3 activation: {config.hidden_act}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.parity_tap_prefix is not None:
            _parity_tap(f"{self.parity_tap_prefix}_input", hidden_states)
        gate_up, _ = self.gate_up_proj(hidden_states)
        if self.parity_tap_prefix is not None:
            _parity_tap(f"{self.parity_tap_prefix}_gate_up", gate_up)
        hidden_states = self.act_fn(gate_up)
        if self.parity_tap_prefix is not None:
            _parity_tap(f"{self.parity_tap_prefix}_activation", hidden_states)
        hidden_states, _ = self.down_proj(hidden_states)
        if self.parity_tap_prefix is not None:
            _parity_tap(f"{self.parity_tap_prefix}_output", hidden_states)
        return hidden_states


class _KimiRoutedOutputTransform(nn.Module):
    """Non-owning callable used by MoERunner after routed expert combine."""

    _norm: nn.Module | None
    _up_proj: nn.Module

    def __init__(
        self,
        norm: nn.Module | None,
        up_proj: nn.Module,
        parity_tap_prefix: str | None = None,
    ) -> None:
        super().__init__()
        self._norm = norm
        self._up_proj = up_proj
        self.parity_tap_prefix = parity_tap_prefix

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Dynamo fullgraph can trace normal attribute access, but not an
        # explicit call to object.__getattribute__.
        norm = self._norm
        up_proj = self._up_proj
        if self.parity_tap_prefix is not None:
            _parity_tap(f"{self.parity_tap_prefix}_combined", hidden_states)
        if norm is not None:
            if kimi_runtime_flag(
                "VLLM_ASCEND_KIMI_REFERENCE_ROUTED_RMS_NORM",
                reduced_default=False,
            ):
                input_dtype = hidden_states.dtype
                normalized = hidden_states.float()
                normalized = normalized * torch.rsqrt(
                    normalized.square().mean(dim=-1, keepdim=True) + norm.variance_epsilon
                )
                hidden_states = norm.weight * normalized.to(input_dtype)
            else:
                hidden_states = norm(hidden_states)
        if self.parity_tap_prefix is not None:
            _parity_tap(f"{self.parity_tap_prefix}_normalized", hidden_states)
        hidden_states = up_proj(hidden_states)[0]
        if self.parity_tap_prefix is not None:
            _parity_tap(f"{self.parity_tap_prefix}_up_output", hidden_states)
        return hidden_states


class KimiK3MoE(nn.Module):
    def __init__(
        self,
        config: KimiK3TextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.hidden_act != "situ":
            raise ValueError("Kimi K3 routed experts require the SiTU activation")
        if config.routed_expert_hidden_size is None:
            raise ValueError("Kimi K3 requires routed_expert_hidden_size")

        self.config = config
        self.hidden_size = config.hidden_size
        self.moe_hidden_size = config.routed_expert_hidden_size
        self.num_shared_experts = config.num_shared_experts
        self.parity_tap_prefix = "02_moe" if "layers.1.block_sparse_moe" in prefix else None
        latent_quant_config = _routed_latent_quant_config(quant_config)
        # Routing always uses the original full-width hidden state.
        self.gate = ReplicatedLinear(
            self.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        # The Ascend MoE runner can schedule the router projection internally
        # with the shared-expert work. Its post-load FP32 copy is made from
        # this BF16 parameter, preserving Kimi's BF16-rounded-weight contract.
        self._use_internal_router_fp32 = kimi_runtime_flag(
            "VLLM_ASCEND_KIMI_REFERENCE_ROUTER_FP32",
            reduced_default=True,
        )
        if self._use_internal_router_fp32:
            self.gate.precast_fp32_weight = True
        self.register_buffer(
            "_reference_router_weight_fp32",
            None,
            persistent=False,
        )
        self.gate.e_score_correction_bias = nn.Parameter(torch.empty(config.num_experts))

        self.routed_expert_down_proj = ReplicatedLinear(
            self.hidden_size,
            self.moe_hidden_size,
            bias=False,
            quant_config=latent_quant_config,
            prefix=f"{prefix}.routed_expert_down_proj",
        )
        routed_norm_type = (
            _KimiDecomposedRMSNorm
            if kimi_runtime_flag(
                "VLLM_ASCEND_KIMI_DECOMPOSED_ROUTED_RMS_NORM",
                reduced_default=True,
            )
            else RMSNorm
        )
        self.routed_expert_norm = (
            routed_norm_type(self.moe_hidden_size, eps=config.rms_norm_eps) if config.latent_moe_use_norm else None
        )
        if isinstance(self.routed_expert_norm, _KimiDecomposedRMSNorm):
            self.routed_expert_norm.kimi_parity_name = f"{prefix}.routed_expert_norm"
        self.routed_expert_up_proj = ReplicatedLinear(
            self.moe_hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=latent_quant_config,
            prefix=f"{prefix}.routed_expert_up_proj",
        )
        routed_output_transform = _KimiRoutedOutputTransform(
            self.routed_expert_norm,
            self.routed_expert_up_proj,
            self.parity_tap_prefix,
        )

        if self.parity_tap_prefix is not None and os.environ.get("KIMI_PARITY_TAP_DIR"):

            def routed_down_hook(_module, _args, output) -> None:
                value = output[0] if isinstance(output, tuple) else output
                _parity_tap("02_moe_routed_down_output", value)

            self.routed_expert_down_proj.register_forward_hook(routed_down_hook)

        self.shared_experts: KimiK3MLP | None
        if self.num_shared_experts:
            self.shared_experts = KimiK3MLP(
                config,
                hidden_size=self.hidden_size,
                intermediate_size=config.moe_intermediate_size * self.num_shared_experts,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_token,
            hidden_size=self.moe_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.moe_renormalize,
            quant_config=quant_config,
            use_grouped_topk=config.use_grouped_topk,
            num_expert_group=config.num_expert_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.moe_router_activation_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=config.routed_scaling_factor,
            n_shared_experts=self.num_shared_experts,
            routed_input_transform=self.routed_expert_down_proj,
            routed_output_transform=routed_output_transform,
            gate=self.gate if self._use_internal_router_fp32 else None,
            activation=SituActivationConfig(
                beta=config.activation_situ_beta or 1.0,
                linear_beta=config.activation_situ_linear_beta,
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        if self.parity_tap_prefix is not None:
            _parity_tap("02_moe_input", hidden_states)
        use_reference_router_fp32 = kimi_runtime_flag(
            "VLLM_ASCEND_KIMI_REFERENCE_ROUTER_FP32",
            reduced_default=True,
        )
        if use_reference_router_fp32 and getattr(self, "_use_internal_router_fp32", False):
            if not self.experts.is_internal_router:
                raise RuntimeError("Kimi K3 internal FP32 router weight was not prepared after checkpoint loading")
            # AscendMoERunner computes the router from the original full-width
            # input. The placeholder is ignored when is_internal_router is
            # true, but keeps the common MoERunner interface tensor-only.
            output = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
            if self.parity_tap_prefix is not None:
                _parity_tap("02_moe_output", output)
            return output.view(num_tokens, hidden_size)
        if use_reference_router_fp32:
            router_weight_fp32 = self._reference_router_weight_fp32
            if router_weight_fp32 is None:
                # The model is inference-only after checkpoint loading. Cache
                # the BF16-rounded parameter in FP32 so decode does not launch
                # the same weight conversion for every token.
                router_weight_fp32 = self.gate.weight.detach().float()
                self._reference_router_weight_fp32 = router_weight_fp32
            router_logits = torch.nn.functional.linear(
                hidden_states.float(),
                router_weight_fp32,
            )
        else:
            router_logits, _ = self.gate(hidden_states)
        if self.parity_tap_prefix is not None:
            _parity_tap("02_moe_router_logits", router_logits)
        output = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        if self.parity_tap_prefix is not None:
            _parity_tap("02_moe_output", output)
        return output.view(num_tokens, hidden_size)


class _KimiReferenceRMSNorm(nn.Module):
    """Opt-in Kimi RMSNorm with the Megatron FP32 accumulation contract."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.get_default_dtype()))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


class _KimiDecomposedRMSNorm(_KimiReferenceRMSNorm):
    """Native NPU reduction with Kimi's BF16-before-weight contract."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__(hidden_size, eps)
        self.register_buffer(
            "unit_weight",
            torch.ones(hidden_size, dtype=torch.get_default_dtype()),
            persistent=False,
        )
        self.kimi_parity_name = "rms_norm"
        self.kimi_call_index = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized, _ = torch_npu.npu_rms_norm(
            hidden_states,
            self.unit_weight,
            self.variance_epsilon,
        )
        output = self.weight * normalized
        if os.environ.get("KIMI_RMSNORM_VALIDATE_REFERENCE") == "1":
            reference = super().forward(hidden_states)
            if self.kimi_call_index == 0:
                logger.info(
                    "KIMI_RMSNORM_FORMAT name=%s input=%s normalized=%s output=%s reference=%s",
                    self.kimi_parity_name,
                    torch_npu.get_npu_format(hidden_states),
                    torch_npu.get_npu_format(normalized),
                    torch_npu.get_npu_format(output),
                    torch_npu.get_npu_format(reference),
                )
            if not torch.equal(output, reference):
                output_dir = os.environ.get("KIMI_PARITY_TAP_DIR")
                if output_dir:
                    safe_name = self.kimi_parity_name.replace(".", "_")
                    prefix = f"rmsnorm_mismatch_{safe_name}_{self.kimi_call_index:04d}"
                    os.makedirs(output_dir, exist_ok=True)
                    for suffix, value in (
                        ("input", hidden_states),
                        ("native", output),
                        ("reference", reference),
                    ):
                        torch.save(
                            value.detach().cpu().contiguous(),
                            os.path.join(output_dir, f"{prefix}_{suffix}.pt"),
                        )
            self.kimi_call_index += 1
        return output


def _reference_mla_preprocess_decode(
    impl,
    q_c: torch.Tensor,
    kv_no_split: torch.Tensor,
    kv_cache: tuple[torch.Tensor, ...],
    attn_metadata,
):
    """Use explicit Kimi Q/K/V projections for the reduced decode path."""
    num_decode_tokens = attn_metadata.num_decode_tokens
    decode_q_c = q_c[:num_decode_tokens]
    decode_q = impl.q_proj(decode_q_c)[0].view(
        -1,
        impl.num_heads,
        impl.qk_head_dim,
    )
    decode_q_nope, decode_q_pe = decode_q.split(
        [impl.qk_nope_head_dim, impl.qk_rope_head_dim],
        dim=-1,
    )
    decode_meta = attn_metadata.decode
    assert decode_meta is not None
    decode_q_pe = impl.rope_single(
        decode_q_pe,
        decode_meta.cos,
        decode_meta.sin,
    )
    decode_slots = attn_metadata.slot_mapping[:num_decode_tokens]
    decode_kv_no_split = kv_no_split[:num_decode_tokens]
    decode_k_pe, decode_k_latent = impl.exec_kv_decode(
        decode_kv_no_split,
        decode_meta.cos,
        decode_meta.sin,
        kv_cache,
        decode_slots,
    )
    return SimpleNamespace(
        ql_nope=decode_q_nope,
        q_pe=decode_q_pe,
        k_nope=decode_k_latent,
        k_pe=decode_k_pe,
        dequant_scale_q_nope=None,
    )


def _reference_mla_forward_decode(
    impl,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_latent_cache: torch.Tensor,
    k_pe_cache: torch.Tensor,
    block_size: int,
    attn_metadata,
    dequant_scale_q_nope=None,
) -> torch.Tensor:
    """Materialize the reduced paged cache and run explicit FP32 attention."""
    del dequant_scale_q_nope
    decode_meta = attn_metadata.decode
    assert decode_meta is not None
    k_latent_cache = k_latent_cache.view(
        -1,
        impl.num_kv_heads,
        block_size,
        impl.kv_lora_rank,
    )
    k_pe_cache = k_pe_cache.view(
        -1,
        impl.num_kv_heads,
        block_size,
        impl.qk_rope_head_dim,
    )
    outputs = []
    for request_index, sequence_length in enumerate(decode_meta.seq_lens_list):
        positions = torch.arange(
            int(sequence_length),
            device=q_nope.device,
            dtype=torch.long,
        )
        logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
        block_offsets = positions.remainder(block_size)
        physical_blocks = decode_meta.block_table[request_index].to(torch.long)[logical_blocks]
        k_latent = k_latent_cache[
            physical_blocks,
            0,
            block_offsets,
        ]
        k_pe = k_pe_cache[
            physical_blocks,
            0,
            block_offsets,
        ]
        key_value = impl.kv_b_proj(k_latent)[0].view(
            -1,
            impl.num_heads,
            impl.qk_nope_head_dim + impl.v_head_dim,
        )
        k_nope, value = key_value.split(
            [impl.qk_nope_head_dim, impl.v_head_dim],
            dim=-1,
        )
        k_pe = k_pe.unsqueeze(1).expand(-1, impl.num_heads, -1)
        scores = torch.einsum(
            "hd,shd->hs",
            q_nope[request_index].float(),
            k_nope.float(),
        )
        scores.add_(
            torch.einsum(
                "hd,shd->hs",
                q_pe[request_index].float(),
                k_pe.float(),
            )
        )
        probabilities = torch.softmax(scores * impl.scale, dim=-1)
        output = torch.einsum(
            "hs,shd->hd",
            probabilities,
            value.float(),
        ).to(q_nope.dtype)
        if getattr(impl, "kimi_parity_layer", None) == 4:
            _parity_tap("04_mla_core_output", output.unsqueeze(0))
        outputs.append(output)
    return torch.stack(outputs).reshape(-1, impl.num_heads * impl.v_head_dim)


def _decomposed_mla_forward_decode(
    impl,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_latent_cache: torch.Tensor,
    k_pe_cache: torch.Tensor,
    block_size: int,
    attn_metadata,
    dequant_scale_q_nope=None,
) -> torch.Tensor:
    """Run reduced MLA decode from replay-updated device metadata.

    FULL graph capture cannot use ``decode.seq_lens_list`` because that Python
    list is frozen at capture time. Materialize the fixed paged-cache capacity,
    then select the attention result for the device-side sequence length. Each
    candidate keeps the same softmax extent as the eager rowwise reference so
    the selected result remains byte exact for the reduced correctness path.
    """
    del dequant_scale_q_nope
    decode_meta = attn_metadata.decode
    assert decode_meta is not None
    k_latent_cache = k_latent_cache.view(
        -1,
        impl.num_kv_heads,
        block_size,
        impl.kv_lora_rank,
    )
    k_pe_cache = k_pe_cache.view(
        -1,
        impl.num_kv_heads,
        block_size,
        impl.qk_rope_head_dim,
    )
    cache_capacity = decode_meta.block_table.shape[1] * block_size
    positions = torch.arange(
        cache_capacity,
        device=q_nope.device,
        dtype=torch.long,
    )
    logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
    block_offsets = positions.remainder(block_size)
    outputs = []
    for request_index in range(q_nope.shape[0]):
        physical_blocks = decode_meta.block_table[request_index].to(torch.long)[logical_blocks]
        k_latent = k_latent_cache[
            physical_blocks,
            0,
            block_offsets,
        ]
        k_pe = k_pe_cache[
            physical_blocks,
            0,
            block_offsets,
        ]
        key_value = impl.kv_b_proj(k_latent)[0].view(
            -1,
            impl.num_heads,
            impl.qk_nope_head_dim + impl.v_head_dim,
        )
        k_nope, value = key_value.split(
            [impl.qk_nope_head_dim, impl.v_head_dim],
            dim=-1,
        )
        k_pe = k_pe.unsqueeze(1).expand(-1, impl.num_heads, -1)
        scores = torch.einsum(
            "hd,shd->hs",
            q_nope[request_index].float(),
            k_nope.float(),
        )
        scores.add_(
            torch.einsum(
                "hd,shd->hs",
                q_pe[request_index].float(),
                k_pe.float(),
            )
        )
        sequence_length = decode_meta.seq_lens_device[request_index]
        output = torch.zeros(
            (impl.num_heads, impl.v_head_dim),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
        for candidate_length in range(1, cache_capacity + 1):
            probabilities = torch.softmax(
                scores[:, :candidate_length] * impl.scale,
                dim=-1,
            )
            candidate = torch.einsum(
                "hs,shd->hd",
                probabilities,
                value[:candidate_length].float(),
            ).to(q_nope.dtype)
            output = torch.where(
                sequence_length == candidate_length,
                candidate,
                output,
            )
        outputs.append(output)
    return torch.stack(outputs).reshape(-1, impl.num_heads * impl.v_head_dim)


def _kimi_mla_prefill_context_length(
    prefill_meta,
    request_index: int,
    query_length: int,
) -> int:
    """Return the already-computed prefix length for one prefill request."""
    sequence_lengths = getattr(prefill_meta, "context_lens", None)
    if sequence_lengths is None:
        return 0
    context_length = int(sequence_lengths[request_index]) - query_length
    if context_length < 0:
        raise ValueError("Kimi MLA prefill sequence length is shorter than its query length")
    return context_length


def _kimi_mla_cached_prefix(
    impl,
    kv_c_and_k_pe_cache: tuple[torch.Tensor, ...],
    prefill_meta,
    request_index: int,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize and up-project one reduced Kimi MLA cached prefix."""
    if context_length <= 0:
        raise ValueError("Kimi MLA cached-prefix materialization requires a positive length")
    if len(kv_c_and_k_pe_cache) < 2:
        raise ValueError("Kimi MLA cached prefill requires latent and positional KV caches")

    k_latent_cache, k_pe_cache = kv_c_and_k_pe_cache[:2]
    block_size = k_latent_cache.shape[1]
    k_latent_cache = k_latent_cache.view(
        -1,
        impl.num_kv_heads,
        block_size,
        impl.kv_lora_rank,
    )
    k_pe_cache = k_pe_cache.view(
        -1,
        impl.num_kv_heads,
        block_size,
        impl.qk_rope_head_dim,
    )
    positions = torch.arange(
        context_length,
        device=k_latent_cache.device,
        dtype=torch.long,
    )
    logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
    block_offsets = positions.remainder(block_size)
    physical_blocks = prefill_meta.block_table[request_index].to(torch.long)[logical_blocks]
    k_latent = k_latent_cache[physical_blocks, 0, block_offsets]
    k_pe = k_pe_cache[physical_blocks, 0, block_offsets]
    key_value = impl.kv_b_proj(k_latent)[0].view(
        -1,
        impl.num_heads,
        impl.qk_nope_head_dim + impl.v_head_dim,
    )
    k_nope, value = key_value.split(
        [impl.qk_nope_head_dim, impl.v_head_dim],
        dim=-1,
    )
    k_pe = k_pe.unsqueeze(1).expand(-1, impl.num_heads, -1)
    return k_nope, k_pe, value


def _reference_mla_forward_prefill(
    impl,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    value: torch.Tensor,
    kv_c_and_k_pe_cache: tuple[torch.Tensor, ...],
    attn_metadata,
) -> torch.Tensor:
    """Evaluate causal prefill one query row at a time like cached decode."""
    prefill_meta = attn_metadata.prefill
    assert prefill_meta is not None
    outputs = []
    start = 0
    for request_index, end_value in enumerate(prefill_meta.actual_seq_lengths_q):
        end = int(end_value)
        query_length = end - start
        context_length = _kimi_mla_prefill_context_length(
            prefill_meta,
            request_index,
            query_length,
        )
        if context_length:
            prefix_k_nope, prefix_k_pe, prefix_value = _kimi_mla_cached_prefix(
                impl,
                kv_c_and_k_pe_cache,
                prefill_meta,
                request_index,
                context_length,
            )
        else:
            prefix_k_nope = k_nope[start:start]
            prefix_k_pe = k_pe[start:start]
            prefix_value = value[start:start]
        sequence_k_nope = torch.cat((prefix_k_nope, k_nope[start:end]), dim=0)
        sequence_k_pe = torch.cat((prefix_k_pe, k_pe[start:end]), dim=0)
        sequence_value = torch.cat((prefix_value, value[start:end]), dim=0)
        for token_idx in range(start, end):
            current_offset = token_idx - start
            visible_length = context_length + current_offset + 1
            scores = torch.einsum(
                "hd,shd->hs",
                q_nope[token_idx].float(),
                sequence_k_nope[:visible_length].float(),
            )
            scores.add_(
                torch.einsum(
                    "hd,shd->hs",
                    q_pe[token_idx].float(),
                    sequence_k_pe[:visible_length].float(),
                )
            )
            probabilities = torch.softmax(scores * impl.scale, dim=-1)
            outputs.append(
                torch.einsum(
                    "hs,shd->hd",
                    probabilities,
                    sequence_value[:visible_length].float(),
                ).to(q_nope.dtype)
            )
        start = end
    if start != q_nope.shape[0]:
        raise ValueError("Kimi MLA prefill cumulative sequence lengths do not cover all rows")
    return torch.stack(outputs).reshape(-1, impl.num_heads * impl.v_head_dim)


def _decomposed_mla_forward_prefill(
    impl,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    value: torch.Tensor,
    kv_c_and_k_pe_cache: tuple[torch.Tensor, ...],
    attn_metadata,
) -> torch.Tensor:
    """Run reduced Kimi prefill with vectorized FP32 attention reductions."""
    prefill_meta = attn_metadata.prefill
    assert prefill_meta is not None
    outputs = []
    start = 0
    for request_index, end_value in enumerate(prefill_meta.actual_seq_lengths_q):
        end = int(end_value)
        sequence_length = end - start
        if sequence_length == 0:
            continue
        context_length = _kimi_mla_prefill_context_length(
            prefill_meta,
            request_index,
            sequence_length,
        )
        q_nope_sequence = q_nope[start:end].float()
        q_pe_sequence = q_pe[start:end].float()
        if context_length:
            prefix_k_nope, prefix_k_pe, prefix_value = _kimi_mla_cached_prefix(
                impl,
                kv_c_and_k_pe_cache,
                prefill_meta,
                request_index,
                context_length,
            )
        else:
            prefix_k_nope = k_nope[start:start]
            prefix_k_pe = k_pe[start:start]
            prefix_value = value[start:start]
        k_nope_sequence = torch.cat((prefix_k_nope, k_nope[start:end]), dim=0).float()
        k_pe_sequence = torch.cat((prefix_k_pe, k_pe[start:end]), dim=0).float()
        value_sequence = torch.cat((prefix_value, value[start:end]), dim=0).float()
        scores = torch.einsum(
            "thd,shd->hts",
            q_nope_sequence,
            k_nope_sequence,
        )
        scores.add_(
            torch.einsum(
                "thd,shd->hts",
                q_pe_sequence,
                k_pe_sequence,
            )
        )
        causal_mask = torch.triu(
            torch.ones(
                (sequence_length, context_length + sequence_length),
                dtype=torch.bool,
                device=q_nope.device,
            ),
            diagonal=context_length + 1,
        )
        probabilities = torch.softmax(
            scores.masked_fill(causal_mask.unsqueeze(0), float("-inf")) * impl.scale,
            dim=-1,
        )
        outputs.append(
            torch.einsum(
                "hts,shd->thd",
                probabilities,
                value_sequence,
            ).to(q_nope.dtype)
        )
        start = end
    if start != q_nope.shape[0]:
        raise ValueError("Kimi MLA prefill cumulative sequence lengths do not cover all rows")
    return torch.cat(outputs).reshape(-1, impl.num_heads * impl.v_head_dim)


class KimiK3MLAAttention(nn.Module):
    """Q-LoRA MLA with a position-independent q/k slice and output gate."""

    def __init__(
        self,
        config: KimiK3TextConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        if not config.mla_use_nope or config.mla_use_rope:
            raise ValueError("Kimi K3 MLA must use the explicit no-RoPE path")
        if not config.mla_use_output_gate:
            raise ValueError("Kimi K3 MLA requires its output gate")

        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size:
            raise ValueError("num_attention_heads must be divisible by tensor parallel size")
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5

        self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            hidden_size,
            [q_lora_rank, kv_lora_rank + qk_rope_head_dim],
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        if kimi_runtime_flag(
            "VLLM_ASCEND_KIMI_REFERENCE_MLA_RMS_NORM",
            reduced_default=False,
        ):
            norm_type = _KimiReferenceRMSNorm
        elif kimi_runtime_flag(
            "VLLM_ASCEND_KIMI_DECOMPOSED_MLA_RMS_NORM",
            reduced_default=True,
        ):
            norm_type = _KimiDecomposedRMSNorm
        else:
            norm_type = RMSNorm
        self.q_a_layernorm = norm_type(q_lora_rank, eps=config.rms_norm_eps)
        if isinstance(self.q_a_layernorm, _KimiDecomposedRMSNorm):
            self.q_a_layernorm.kimi_parity_name = f"{prefix}.q_a_layernorm"
        self.q_b_proj = ColumnParallelLinear(
            q_lora_rank,
            num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = norm_type(kv_lora_rank, eps=config.rms_norm_eps)
        if isinstance(self.kv_a_layernorm, _KimiDecomposedRMSNorm):
            self.kv_a_layernorm.kimi_parity_name = f"{prefix}.kv_a_layernorm"
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.g_proj = ColumnParallelLinear(
            hidden_size,
            num_heads * v_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        )
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=None,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=None,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=None,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
        )
        # MLAModules is an upstream dataclass without K3 fields.  Dynamic
        # attributes preserve compatibility while the Ascend OOT wrapper
        # consumes the model-specific modules.
        mla_modules.g_proj = self.g_proj
        mla_modules.use_output_gate = True
        mla_modules.use_mla_rope = False
        self.mla_attn = MultiHeadLatentAttentionWrapper(
            hidden_size,
            self.num_local_heads,
            self.scaling,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
        )
        mla_impl = self.mla_attn.mla_attn.impl
        mla_impl.kimi_reduced_shape_decode = _configure_kimi_mlapo_shape(
            mla_impl,
            kv_lora_rank,
        )
        if mla_impl.kimi_reduced_shape_decode:
            mla_impl.kimi_reduced_shape_rope_dim = _KIMI_MLA_KERNEL_ROPE_DIM
        reference_mla = kimi_runtime_flag(
            "VLLM_ASCEND_KIMI_REFERENCE_MLA_DECODE",
            reduced_default=False,
        )
        decomposed_mla = kimi_runtime_flag(
            "VLLM_ASCEND_KIMI_DECOMPOSED_MLA_ATTENTION",
            reduced_default=True,
        )
        if reference_mla or decomposed_mla:
            # The A5 fused MLA prolog requires the production 512-wide KV
            # latent.  Keep the normal paged cache, but route this deliberately
            # reduced 128-wide fixture through architecture-preserving explicit
            # Kimi projections and attention math.  The decomposed path batches
            # prefill reductions while the reference path retains its rowwise
            # oracle implementation.
            mla_impl.enable_mlapo = False
            if "layers.3.self_attn" in prefix:
                mla_impl.kimi_parity_layer = 4
            mla_impl.mla_preprocess_decode = MethodType(
                _reference_mla_preprocess_decode,
                mla_impl,
            )
            prefill_forward = (
                _decomposed_mla_forward_prefill
                if decomposed_mla
                else _reference_mla_forward_prefill
            )
            mla_impl._forward_prefill = MethodType(prefill_forward, mla_impl)
            decode_forward = (
                _decomposed_mla_forward_decode
                if decomposed_mla
                else _reference_mla_forward_decode
            )
            mla_impl._forward_decode = MethodType(decode_forward, mla_impl)
        elif os.environ.get("VLLM_ASCEND_KIMI_DECOMPOSED_MLA_PREFILL") == "1":
            mla_impl._forward_prefill = MethodType(
                _decomposed_mla_forward_prefill,
                mla_impl,
            )

        tap_layer = next(
            (layer for layer in (4, 8) if f"layers.{layer - 1}.self_attn" in prefix),
            None,
        )
        if tap_layer is not None and os.environ.get("KIMI_PARITY_TAP_DIR"):

            def capture(name: str):
                def hook(_module, _args, output) -> None:
                    value = output[0] if isinstance(output, tuple) else output
                    _parity_tap(f"{tap_layer:02d}_mla_{name}", value)

                return hook

            for module, name in (
                (self.fused_qkv_a_proj, "qkv_a_fused"),
                (self.q_a_layernorm, "q_a_norm"),
                (self.q_b_proj, "q_b"),
                (self.kv_a_layernorm, "kv_a_norm"),
                (self.kv_b_proj, "kv_b"),
                (self.g_proj, "gate_raw"),
                (self.o_proj, "o_proj"),
            ):
                module.register_forward_hook(capture(name))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        output[:] = self.mla_attn(positions, hidden_states)


def _apply_attention_residual(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection: nn.Module,
    norm: RMSNorm,
) -> torch.Tensor:
    """Apply K3's learned normalized mixture over residual block starts."""
    if kimi_runtime_flag(
        "VLLM_ASCEND_KIMI_REFERENCE_ATTN_RES",
        reduced_default=True,
    ):
        if kimi_runtime_flag(
            "VLLM_ASCEND_KIMI_VECTORIZED_ATTN_RES",
            reduced_default=False,
        ):
            rows = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
            score_weight = norm.weight.float() * projection.weight.squeeze(0).float()
            rows_float = rows.float()
            if kimi_runtime_flag(
                "VLLM_ASCEND_KIMI_ATTN_RES_NATIVE_RMS_NORM",
                reduced_default=False,
            ):
                normalized, _ = torch_npu.npu_rms_norm(
                    rows_float,
                    norm.weight.float(),
                    norm.variance_epsilon,
                )
                score_weight = projection.weight.squeeze(0).float()
            else:
                normalized = rows_float * torch.rsqrt(
                    rows_float.square().mean(dim=-1, keepdim=True) + norm.variance_epsilon
                )
            if kimi_runtime_flag(
                "VLLM_ASCEND_KIMI_ATTN_RES_NATIVE_SCORE_MATMUL",
                reduced_default=False,
            ):
                scores = torch.matmul(
                    normalized,
                    score_weight.unsqueeze(-1),
                ).squeeze(-1)
            else:
                scores = (normalized * score_weight).sum(dim=-1)
            probabilities = torch.softmax(scores, dim=-1)
            if kimi_runtime_flag(
                "VLLM_ASCEND_KIMI_ATTN_RES_NATIVE_MIX_MATMUL",
                reduced_default=False,
            ):
                return (
                    torch.matmul(
                        probabilities.unsqueeze(1),
                        rows_float,
                    )
                    .squeeze(1)
                    .to(rows.dtype)
                )
            return (probabilities.unsqueeze(-1) * rows_float).sum(dim=-2).to(rows.dtype)
        if prefix_sum.device.type == "npu":
            return torch.ops.vllm.kimi_reference_rowwise_attention_residual(
                prefix_sum,
                block_residual,
                projection.weight,
                norm.weight,
                norm.variance_epsilon,
            )
        return _kimi_reference_rowwise_attention_residual_impl(
            prefix_sum,
            block_residual,
            projection.weight,
            norm.weight,
            norm.variance_epsilon,
        )

    use_triton_attention_residual = (
        apply_attn_res is not None
        and not kimi_runtime_flag(
            "VLLM_ASCEND_KIMI_NATIVE_ATTN_RES",
            reduced_default=True,
        )
        and prefix_sum.device.type == "npu"
        and prefix_sum.numel() > 0
    )
    if use_triton_attention_residual:
        mixed = apply_attn_res(prefix_sum, block_residual, projection, norm)
    else:
        values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
        values_fp32 = values.float()
        normalized, _ = torch_npu.npu_rms_norm(
            values_fp32,
            norm.weight.float(),
            norm.variance_epsilon,
        )
        scores = torch.matmul(normalized, projection.weight.t().float()).squeeze(-1)
        probabilities = scores.softmax(-1).unsqueeze(1)
        mixed = torch.matmul(probabilities, values_fp32).squeeze(1).to(values.dtype)
    if _EXTRA_CTX.flash_comm_v1_enabled:
        # FlashComm changes the first decoder layer from the global token
        # layout to a TP-local layout.  The learned-residual arithmetic above
        # can specialize that derived token dimension to the compile example
        # size.  Re-anchor it to prefix_sum through the existing no-op residual
        # helper so later KDA/MLA layers keep the TP-local SymInt dynamic.
        mixed = torch.ops.vllm.maybe_chunk_residual(prefix_sum, mixed)
    return mixed


class KimiK3DecoderLayer(nn.Module):
    def __init__(self, config: KimiK3TextConfig, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.rsplit(".", 1)[1])
        self.is_vl_first_layer = bool(uses_kimi_k3_global_inputs_embeds(vllm_config) and self.layer_idx == 0)
        quant_config = vllm_config.quant_config

        if config.is_kda_layer(self.layer_idx):
            # Instantiate by the upstream registered class name so vLLM's
            # PluggableLayer mechanism selects the Ascend OOT implementation.
            self.self_attn = KimiGatedDeltaNetAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = KimiK3MLAAttention(
                config=config,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                cache_config=vllm_config.cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )

        if (
            config.num_experts is not None
            and self.layer_idx >= config.first_k_dense_replace
            and self.layer_idx % config.moe_layer_freq == 0
        ):
            # Keep the registered module name identical to the checkpoint.
            # A local ``self.mlp`` alias would move every MoE parameter under
            # ``layers.N.mlp`` and make AutoWeightsLoader miss
            # ``layers.N.block_sparse_moe`` weights.
            self.block_sparse_moe = KimiK3MoE(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
            )
        else:
            self.mlp = KimiK3MLP(
                config,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        attn_res_block_size = config.attn_res_block_size
        if attn_res_block_size is None:
            raise ValueError("Kimi K3 requires attn_res_block_size")
        self.attn_res_block_size = attn_res_block_size
        self.self_attention_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention_res_proj = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.self_attention_res_proj",
        )
        self.mlp_res_proj = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.mlp_res_proj",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _parity_tap(f"{self.layer_idx + 1:02d}_layer_input", hidden_states)
        _parity_tap(f"{self.layer_idx + 1:02d}_block_residual_input", block_residual)
        prefix_sum: torch.Tensor | None = hidden_states
        if block_residual.shape[1] > 0:
            hidden_states = _apply_attention_residual(
                prefix_sum,
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            )

        if self.layer_idx % self.attn_res_block_size == 0:
            assert prefix_sum is not None
            block_residual = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)
        _parity_tap(f"{self.layer_idx + 1:02d}_attention_input", hidden_states)
        if self.is_vl_first_layer and _EXTRA_CTX.flash_comm_v1_enabled:
            tp_size = get_tensor_model_parallel_world_size()
            num_local_tokens = hidden_states.shape[0] // tp_size
            attention_output = torch.empty(
                (num_local_tokens, hidden_states.shape[-1]),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        else:
            attention_output = torch.empty_like(hidden_states)
        self.self_attn(positions=positions, hidden_states=hidden_states, output=attention_output)
        _parity_tap(f"{self.layer_idx + 1:02d}_attention_output", attention_output)

        # The multimodal first layer transitions from full inputs_embeds to a
        # FlashComm token shard.  The token axis is dim 0 for both tensors, so
        # reuse the framework residual sharding op directly.  The singleton
        # block axis on the output anchor keeps fake/meta and runtime 3-D.
        if self.is_vl_first_layer and _EXTRA_CTX.flash_comm_v1_enabled:
            block_residual = torch.ops.vllm.maybe_chunk_residual(
                attention_output.unsqueeze(1),
                block_residual,
            )
        prefix_sum = attention_output if prefix_sum is None else prefix_sum + attention_output
        hidden_states = _apply_attention_residual(
            prefix_sum,
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        _parity_tap(f"{self.layer_idx + 1:02d}_mlp_input", hidden_states)
        if hasattr(self, "block_sparse_moe"):
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        _parity_tap(f"{self.layer_idx + 1:02d}_mlp_output", hidden_states)
        layer_output = prefix_sum + hidden_states
        _parity_tap(f"{self.layer_idx + 1:02d}_layer_output", layer_output)
        _parity_tap(f"{self.layer_idx + 1:02d}_block_residual_output", block_residual)
        return layer_output, block_residual


@support_torch_compile
class KimiK3TextModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: KimiK3TextConfig = vllm_config.model_config.hf_text_config
        reduced_w4a8 = configure_kimi_reduced_w4a8_runtime(vllm_config)
        if reduced_w4a8:
            logger.info(
                "KIMI_REDUCED_W4A8_DEFAULT_PROFILE hidden=%d layers=%d "
                "heads=%d kv_lora_rank=%d",
                config.hidden_size,
                config.num_hidden_layers,
                config.num_attention_heads,
                config.kv_lora_rank,
            )
        self.config = config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            return KimiK3DecoderLayer(config, vllm_config, prefix)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.output_attn_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.output_attn_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.output_attn_res_proj",
            )
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.output_attn_res_norm = PPMissingLayer()
            self.output_attn_res_proj = PPMissingLayer()
            self.norm = PPMissingLayer()
        # Legacy Qwen3/GQA DSpark checkpoints were trained from the
        # materialized Kimi residual stream. Keep the PR #13071 raw-boundary
        # behavior as the default for MLA-style draft checkpoints.
        self.dspark_aux_capture_materialized = False

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def initial_block_count(self) -> int:
        block_size = self.config.attn_res_block_size
        if block_size is None:
            raise ValueError("Kimi K3 requires attn_res_block_size")
        return sum(layer_idx % block_size == 0 for layer_idx in range(self.start_layer))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        del kwargs
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            _parity_tap("00_embedding", hidden_states)
            block_residual = hidden_states.new_zeros((hidden_states.shape[0], 0, hidden_states.shape[-1]))
        else:
            if intermediate_tensors is None:
                raise ValueError("intermediate_tensors are required on non-first PP ranks")
            hidden_states = intermediate_tensors["hidden_states"]
            block_residual = intermediate_tensors["block_residual"]

        aux_hidden_states: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer],
            start=self.start_layer,
        ):
            # The GQA drafter consumes the materialized input to the next
            # target layer. Kimi K3 stores part of that stream separately in
            # block_residual, so fold it through that layer's projection.
            if self.dspark_aux_capture_materialized and layer_idx in self.aux_hidden_state_layers:
                if block_residual.shape[1] == 0:
                    aux_hidden_states.append(hidden_states)
                else:
                    aux_hidden_states.append(
                        _apply_attention_residual(
                            hidden_states,
                            block_residual,
                            layer.self_attention_res_proj,
                            layer.self_attention_res_norm,
                        )
                    )
            hidden_states, block_residual = layer(positions, hidden_states, block_residual)
            if not self.dspark_aux_capture_materialized and (layer_idx + 1) in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "block_residual": block_residual})

        hidden_states = _apply_attention_residual(
            hidden_states,
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        )
        _parity_tap("09_final_norm_input", hidden_states)
        hidden_states = self.norm(hidden_states)
        _parity_tap("10_final_norm_output", hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor] | tuple[str, torch.Tensor, dict[str, Any]]],
    ) -> set[str]:
        stacked_params_mapping = [
            (".fused_qkv", ".q_proj", "q"),
            (".fused_qkv", ".k_proj", "k"),
            (".fused_qkv", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
        ]
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_experts,
        )
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for args in weights:
            name, loaded_weight = args[:2]
            loader_kwargs: dict[str, Any] = args[2] if len(args) == 3 else {}
            if "rotary_emb" in name:
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ".experts." in name and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".weight_packed") and name not in params_dict:
                    unpacked_name = name[: -len("_packed")]
                    if unpacked_name in params_dict:
                        name = unpacked_name
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    break
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    name = _resolve_packed_expert_weight_name(name, params_dict)
                    if is_pp_missing_parameter(name, self):
                        break
                    param = params_dict[name]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=shard_id,
                    )
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None or is_pp_missing_parameter(name, self):
                        continue
                    if name.endswith(".weight_packed") and name not in params_dict:
                        unpacked_name = name[: -len("_packed")]
                        if unpacked_name in params_dict:
                            name = unpacked_name
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight, **loader_kwargs)
            loaded_params.add(name)
        return loaded_params


class AscendKimiK3ForCausalLM(nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid):
    packed_modules_mapping = {
        "fused_qkv": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts": ["experts.0.w1", "experts.0.w3", "experts.0.w2"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config: KimiK3TextConfig = self.model_config.hf_text_config
        self.quant_config = vllm_config.quant_config
        self.model = KimiK3TextModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            scale=getattr(self.config, "logit_scale", 1.0),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_dspark_aux_capture_materialized(self, enabled: bool) -> None:
        self.model.dspark_aux_capture_materialized = enabled

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        _parity_tap("11_logits", logits)
        return logits

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros((batch_size, self.config.hidden_size), dtype=dtype, device=device),
                "block_residual": torch.zeros(
                    (batch_size, self.model.initial_block_count(), self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
            }
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        parallel_config = vllm_config.parallel_config
        config = vllm_config.model_config.hf_text_config
        num_spec = vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0
        return kimi_kda_state_shape(
            parallel_config.tensor_parallel_size,
            config.linear_attn_config["num_heads"],
            config.linear_attn_config["head_dim"],
            config.linear_attn_config["short_conv_kernel_size"],
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


def get_spec_layer_idx_from_weight_name(config: KimiK3TextConfig, weight_name: str) -> int | None:
    num_nextn_predict_layers = getattr(
        config,
        "num_nextn_predict_layers",
        0,
    )
    for index in range(num_nextn_predict_layers):
        layer_idx = config.num_hidden_layers + index
        if f"layers.{layer_idx}." in weight_name:
            return layer_idx
    return None


class KimiK3VisionPatchEmbed(nn.Module):
    def __init__(self, config: KimiK3VisionConfig) -> None:
        super().__init__()
        configured_patch_size: int | Sequence[int] = config.patch_size
        if isinstance(configured_patch_size, int):
            self.patch_size = (configured_patch_size, configured_patch_size)
        elif isinstance(configured_patch_size, Sequence) and len(configured_patch_size) == 2:
            self.patch_size = (configured_patch_size[0], configured_patch_size[1])
        else:
            raise ValueError(f"Invalid Kimi K3 patch size: {configured_patch_size}")
        self.proj = nn.Conv2d(
            3,
            config.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=config.patch_embed_proj_bias,
        )
        self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
            height=config.init_pos_emb_height,
            width=config.init_pos_emb_width,
            num_frames=config.init_pos_emb_time,
            dim=config.hidden_size,
            # The released K3 reference constructor leaves this at the
            # LearnablePosEmbInterp default, which is bicubic. Its config.json
            # contains a bilinear field but the reference model never consumes
            # it, so honoring that field here would change non-64x64 grids.
            interpolation_mode="bicubic",
        )

    def forward(self, pixels: torch.Tensor, grid_thws: torch.Tensor | list[list[int]]) -> torch.Tensor:
        hidden_states = self.proj(pixels).view(pixels.shape[0], -1)
        return self.pos_emb(hidden_states, grid_thws)


class KimiK3VisionMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: nn.Module,
        quant_config: QuantizationConfig | None,
        prefix: str,
        use_data_parallel: bool,
        bias: bool,
    ) -> None:
        super().__init__()
        self.fc0 = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc0",
            disable_tp=use_data_parallel,
        )
        self.fc1 = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
            disable_tp=use_data_parallel,
        )
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc0(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.fc1(hidden_states)[0]


class KimiK3VisionEncoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.use_data_parallel = _is_vit_use_data_parallel(config.num_attention_heads)
        self.hidden_dim = config.hidden_size
        self.qkv_hidden_size = config.qkv_hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.qkv_hidden_size // self.num_heads
        if self.qkv_hidden_size % self.num_heads:
            raise ValueError("K3 qkv_hidden_size must be divisible by vision heads")
        self.tp_size = 1 if self.use_data_parallel else get_tensor_model_parallel_world_size()
        self.num_local_heads = divide(self.num_heads, self.tp_size)

        if config.norm_type != "rmsnorm":
            raise ValueError(f"K3 vision requires RMSNorm, got {config.norm_type}")
        # The released K3 implementation intentionally leaves encoder eps at
        # torch.nn.RMSNorm's dtype-dependent default.  Projector RMSNorm below
        # is the only vision norm with an explicit 1e-5 checkpoint contract.
        self.norm0 = nn.RMSNorm(self.hidden_dim)
        self.norm1 = nn.RMSNorm(self.hidden_dim)
        self.mlp = KimiK3VisionMLP(
            self.hidden_dim,
            config.intermediate_size,
            get_act_fn(config.activation_func),
            quant_config,
            f"{prefix}.mlp",
            self.use_data_parallel,
            config.linear_bias,
        )
        self.wqkv = QKVParallelLinear(
            hidden_size=self.hidden_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_heads,
            bias=config.attn_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.wqkv",
            disable_tp=self.use_data_parallel,
        )
        self.wo = RowParallelLinear(
            self.qkv_hidden_size,
            self.hidden_dim,
            bias=config.attn_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.wo",
            disable_tp=self.use_data_parallel,
        )
        self.attn = MMEncoderAttention(
            num_heads=self.num_local_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            prefix=f"{prefix}.attn",
        )

    def attention(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        qkv = self.wqkv(hidden_states)[0].view(
            num_tokens,
            3,
            self.num_local_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=1)
        # QKVParallelLinear shards heads, while the shared RoPE table is head
        # independent and therefore needs no TP slicing.
        query, key = apply_rope(query, key, rope_freqs_cis)
        output = self.attn(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        output = output.reshape(num_tokens, self.num_local_heads * self.head_dim)
        return self.wo(output)[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        hidden_states = residual + self.attention(
            hidden_states,
            cu_seqlens,
            rope_freqs_cis,
            max_seqlen,
            sequence_lengths,
        )
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        return residual + self.mlp(hidden_states)


class KimiK3VisionEncoder(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.rope_2d = Rope2DPosEmbRepeated(
            config.qkv_hidden_size // config.num_attention_heads,
            512,
            512,
        )
        self.blocks = nn.ModuleList(
            [
                KimiK3VisionEncoderLayer(config, quant_config, f"{prefix}.blocks.{layer_idx}")
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.final_layernorm = nn.RMSNorm(config.hidden_size)

    def prepare_encoder_metadata(
        self,
        grid_thw_list: list[list[int]],
        device: torch.device,
    ) -> dict[str, torch.Tensor | None]:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_thw_list, device=device)
        grid = np.asarray(grid_thw_list, dtype=np.int32)
        lengths = grid[:, 0] * grid[:, 1] * grid[:, 2]
        cu_seqlens_np = np.concatenate((np.zeros(1, dtype=np.int32), lengths.cumsum(dtype=np.int32)))
        backend = self.blocks[0].attn.attn_backend
        sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(backend, cu_seqlens_np, device)
        max_seqlen = torch.tensor(
            MMEncoderAttention.compute_max_seqlen(backend, cu_seqlens_np),
            dtype=torch.int32,
        )
        cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
            backend,
            cu_seqlens_np,
            self.blocks[0].hidden_dim,
            self.blocks[0].tp_size,
            device,
        )
        return {
            "rope_freqs_cis": rope_freqs_cis,
            "sequence_lengths": sequence_lengths,
            "max_seqlen": max_seqlen,
            "cu_seqlens": cu_seqlens,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thws: torch.Tensor | list[list[int]],
        encoder_metadata: dict[str, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        grid_thw_list = grid_thws if isinstance(grid_thws, list) else grid_thws.tolist()
        if encoder_metadata is None:
            encoder_metadata = self.prepare_encoder_metadata(grid_thw_list, hidden_states.device)
        rope_freqs_cis = encoder_metadata["rope_freqs_cis"]
        cu_seqlens = encoder_metadata["cu_seqlens"]
        max_seqlen = encoder_metadata["max_seqlen"]
        if rope_freqs_cis is None or cu_seqlens is None or max_seqlen is None:
            raise RuntimeError("Incomplete Kimi K3 vision attention metadata")
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens,
                rope_freqs_cis,
                max_seqlen,
                encoder_metadata["sequence_lengths"],
            )
        return self.final_layernorm(hidden_states)


class KimiK3VisionTower(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = deepcopy(config)
        self.patch_size = config.patch_size
        self.merge_kernel_size = config.merge_kernel_size
        self.merge_type = config.merge_type
        self.patch_embed = KimiK3VisionPatchEmbed(config)
        self.encoder = KimiK3VisionEncoder(config, quant_config, maybe_prefix(prefix, "encoder"))

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor | list[list[int]],
        encoder_metadata: dict[str, torch.Tensor | None] | None = None,
    ) -> list[torch.Tensor]:
        grid_thw_list = grid_thws if isinstance(grid_thws, list) else grid_thws.tolist()
        if encoder_metadata is None:
            encoder_metadata = self.encoder.prepare_encoder_metadata(grid_thw_list, pixel_values.device)
        hidden_states = self.patch_embed(pixel_values, grid_thw_list)
        hidden_states = self.encoder(hidden_states, grid_thw_list, encoder_metadata)
        if self.merge_type != "sd2_tpool":
            raise ValueError(f"Unsupported Kimi K3 merge type: {self.merge_type}")
        return tpool_patch_merger(hidden_states, grid_thw_list, self.merge_kernel_size)


class KimiK3MultiModalProjector(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        merge_size = config.merge_kernel_size[0] * config.merge_kernel_size[1]
        self.input_size = config.mm_hidden_size * merge_size
        if config.mm_projector_type != "patchmergerv2":
            raise ValueError(f"Unsupported Kimi K3 projector: {config.mm_projector_type}")
        self.linear_1 = ReplicatedLinear(
            self.input_size,
            self.input_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
        )
        self.linear_2 = ReplicatedLinear(
            self.input_size,
            config.text_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
        )
        self.act = get_act_fn(config.projector_hidden_act)
        self.post_norm = RMSNorm(config.text_hidden_size, eps=config.projector_ln_eps)
        # ModelSlim rotates K3's FP4 activations before INT4 inference. Text
        # embeddings fold this matrix into their input projection, but the
        # vision path ends in RMSNorm, so the rotation must remain explicit.
        self.rot_proj: ReplicatedLinear | None = ReplicatedLinear(
            config.text_hidden_size,
            config.text_hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.rot_proj",
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = image_features.reshape(-1, self.input_size)
        hidden_states = self.linear_1(hidden_states)[0]
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)[0]
        hidden_states = self.post_norm(hidden_states)
        rot_proj = getattr(self, "rot_proj", None)
        if rot_proj is not None:
            hidden_states = rot_proj(hidden_states)[0]
        return hidden_states


@torch.inference_mode()
def vision_tower_forward(
    vision_tower: KimiK3VisionTower,
    pixel_values: torch.Tensor,
    grid_thws: torch.Tensor,
    mm_projector: KimiK3MultiModalProjector,
    use_data_parallel: bool,
) -> list[torch.Tensor]:
    grid_thw_list = grid_thws.tolist()
    if use_data_parallel:
        tower_outputs = run_dp_sharded_mrope_vision_model(
            vision_model=vision_tower,
            pixel_values=pixel_values,
            grid_thw_list=grid_thw_list,
            rope_type="rope_2d",
        )
    else:
        metadata = vision_tower.encoder.prepare_encoder_metadata(grid_thw_list, pixel_values.device)
        tower_outputs = vision_tower(pixel_values, grid_thw_list, metadata)

    lengths = [item.shape[0] for item in tower_outputs]
    batched = torch.cat(list(tower_outputs), dim=0)
    projected = mm_projector(batched)
    return list(projected.split(lengths, dim=0))


__all__ = [
    "AscendKimiK3ForCausalLM",
    "AscendKimiK3ForConditionalGeneration",
    "KimiK3DecoderLayer",
    "KimiK3MLAAttention",
    "KimiK3MLP",
    "KimiK3MoE",
    "KimiK3MultiModalProjector",
    "KimiK3TextModel",
    "KimiK3VisionEncoderLayer",
    "KimiK3VisionTower",
    "vision_tower_forward",
]
