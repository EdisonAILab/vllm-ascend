#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

import os
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.logger import logger
from vllm.utils.math_utils import cdiv

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.device.mxfp_compat import (
    FLOAT8_E8M0FNU_DTYPE,
    ensure_mxfp8_linear_available,
    ensure_mxfp8_moe_available,
)
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input

from .base import AscendLinearScheme, AscendMoEScheme, QuantType, get_moe_num_logical_experts
from .registry import register_scheme

_DENSE_BI_DECOMPOSE = os.environ.get("VLLM_MXFP8_DENSE_BI_DECOMPOSE") == "1"
_GROUPED_BI_DECOMPOSE = os.environ.get("VLLM_MXFP8_GROUPED_BI_DECOMPOSE") == "1"
_GROUPED_BI_GRAPH_NATIVE = (
    os.environ.get("VLLM_MXFP8_GROUPED_BI_GRAPH_NATIVE") == "1"
)
_DENSE_BI_NOTICE_PRINTED = False
_GROUPED_BI_NOTICE_PRINTED = False
_GROUPED_WEIGHT_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}
_NATIVE_NPU_GROUPED_MATMUL = torch_npu.npu_grouped_matmul


def _e8m0_to_f32(scale: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 scales without introducing another reduction."""
    raw = scale if scale.dtype == torch.uint8 else scale.contiguous().view(torch.uint8)
    return torch.exp2(raw.to(torch.float32) - 127.0)


def _dequant_activation(
    value: torch.Tensor, packed_scale: torch.Tensor, group_size: int
) -> torch.Tensor:
    rows, width = value.shape
    scale = _e8m0_to_f32(packed_scale).reshape(rows, -1)
    scale = scale.repeat_interleave(group_size, dim=1)[:, :width]
    return value.to(torch.bfloat16).to(torch.float32) * scale


def _dequant_weight(
    weight: torch.Tensor, packed_scale: torch.Tensor, group_size: int
) -> torch.Tensor:
    width, columns = weight.shape
    scale = _e8m0_to_f32(packed_scale)
    if scale.ndim == 2:
        scale = scale.reshape(-1, columns)
    elif scale.ndim == 3:
        scale = scale.permute(0, 2, 1).reshape(-1, columns)
    else:
        raise RuntimeError(
            f"unsupported MXFP8 weight-scale rank {scale.ndim}; expected 2 or 3"
        )
    scale = scale.repeat_interleave(group_size, dim=0)[:width]
    return weight.to(torch.bfloat16).to(torch.float32) * scale


def _dense_bi_matmul(
    layer: torch.nn.Module,
    quantized_x: torch.Tensor,
    pertoken_scale: torch.Tensor,
    group_size: int,
    output_dtype: torch.dtype,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    global _DENSE_BI_NOTICE_PRINTED
    if not _DENSE_BI_NOTICE_PRINTED:
        print(
            "[BI_MXFP8_DENSE] dequantize + bf16 fixed-order matmul enabled",
            flush=True,
        )
        _DENSE_BI_NOTICE_PRINTED = True

    cached = getattr(layer, "_bi_dense_bf16_weight", None)
    if cached is None:
        cached = _dequant_weight(layer.weight, layer.weight_scale, group_size).to(
            torch.bfloat16
        )
        layer._bi_dense_bf16_weight = cached

    x_bf16 = _dequant_activation(quantized_x, pertoken_scale, group_size).to(
        torch.bfloat16
    )
    import batch_invariant_ops  # noqa: F401

    output = torch.ops.batch_invariant_ops.npu_mm_batch_invariant(
        x_bf16.contiguous(), cached.contiguous()
    )
    if bias is not None:
        output = output + bias.to(output.dtype)
    return output.to(output_dtype)


def _group_offsets(
    group_list: torch.Tensor, group_list_type: int | None, total: int
) -> list[int]:
    del group_list_type
    groups = group_list.to(torch.int64)
    cumulative = bool((groups[1:] >= groups[:-1]).all().item()) and int(
        groups[-1].item()
    ) == total
    if not cumulative:
        groups = groups.cumsum(0)
    offsets = torch.empty(
        groups.numel() + 1, dtype=torch.int64, device=groups.device
    )
    offsets[0] = 0
    offsets[1:] = groups
    return offsets.tolist()


def _grouped_weight_bf16(
    weight: torch.Tensor,
    packed_scale: torch.Tensor,
    expert: int,
    group_size: int,
    tag: str,
) -> torch.Tensor:
    key = (weight.untyped_storage().data_ptr(), expert, tag)
    cached = _GROUPED_WEIGHT_CACHE.get(key)
    if cached is None:
        cached = _dequant_weight(weight[expert], packed_scale[expert], group_size).to(
            torch.bfloat16
        )
        _GROUPED_WEIGHT_CACHE[key] = cached
    return cached


def _grouped_weights_bf16_native(
    weight: torch.Tensor,
    packed_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Replace grouped FP8 storage with BF16 once for graph-safe GMM."""
    if weight.dtype == torch.bfloat16:
        return weight

    converted = torch.empty(
        weight.shape,
        dtype=torch.bfloat16,
        device=weight.device,
    )
    for expert in range(weight.shape[0]):
        converted[expert] = _dequant_weight(
            weight[expert], packed_scale[expert], group_size
        ).to(torch.bfloat16)
    weight.data = converted
    return weight


def _grouped_gmm2_graph_bi(
    *,
    values: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    token_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_list: torch.Tensor,
    group_list_type: int | None,
    output_dtype: torch.dtype | None,
) -> list[torch.Tensor]:
    values_bf16 = _dequant_activation(values, token_scale, group_size=32).to(
        torch.bfloat16
    )
    weights_bf16 = _grouped_weights_bf16_native(weights, scales, 32)
    call_kwargs = {
        "x": [values_bf16],
        "weight": [weights_bf16],
        "split_item": 2,
        "group_list_type": group_list_type,
        "group_type": 0,
        "group_list": group_list,
        "output_dtype": output_dtype or torch.bfloat16,
    }
    if bias is not None:
        call_kwargs["bias"] = [bias]
    return _NATIVE_NPU_GROUPED_MATMUL(**call_kwargs)


def _grouped_gmm1_graph_bi(
    *,
    x: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
):
    values_bf16 = _dequant_activation(x, x_scale, group_size=32).to(
        torch.bfloat16
    )
    weights_bf16 = _grouped_weights_bf16_native(weights, scales, 32)
    hidden = _NATIVE_NPU_GROUPED_MATMUL(
        x=[values_bf16],
        weight=[weights_bf16],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=group_list,
        output_dtype=torch.bfloat16,
    )[0]
    activated = torch_npu.npu_swiglu(hidden)
    return torch_npu.npu_dynamic_mx_quant(
        activated, dst_type=torch.float8_e4m3fn
    )


def _grouped_gmm2_bi(
    *,
    x,
    weight,
    scale=None,
    bias=None,
    per_token_scale=None,
    group_list=None,
    group_list_type=None,
    output_dtype=None,
    **kwargs,
):
    del kwargs
    values = x[0] if isinstance(x, (list, tuple)) else x
    weights = weight[0] if isinstance(weight, (list, tuple)) else weight
    scales = scale[0] if isinstance(scale, (list, tuple)) else scale
    token_scale = (
        per_token_scale[0]
        if isinstance(per_token_scale, (list, tuple))
        else per_token_scale
    )
    if _GROUPED_BI_GRAPH_NATIVE:
        return _grouped_gmm2_graph_bi(
            values=values,
            weights=weights,
            scales=scales,
            token_scale=token_scale,
            bias=bias,
            group_list=group_list,
            group_list_type=group_list_type,
            output_dtype=output_dtype,
        )

    experts, _, columns = weights.shape
    offsets = _group_offsets(group_list, group_list_type, values.shape[0])
    values_bf16 = _dequant_activation(values, token_scale, group_size=32).to(
        torch.bfloat16
    )
    result_dtype = output_dtype or torch.bfloat16
    output = torch.zeros(
        values.shape[0], columns, dtype=result_dtype, device=values.device
    )
    for expert in range(experts):
        start, end = offsets[expert], offsets[expert + 1]
        if end <= start:
            continue
        expert_weight = _grouped_weight_bf16(
            weights, scales, expert, 32, "gmm2"
        )
        part = torch.ops.batch_invariant_ops.npu_mm_batch_invariant(
            values_bf16[start:end].contiguous(), expert_weight.contiguous()
        )
        if bias is not None:
            part = part + bias[expert].to(part.dtype)
        output[start:end] = part.to(result_dtype)
    return [output]


def _grouped_gmm1_bi(
    *, x, weight, group_list=None, weight_scale=None, x_scale=None, **kwargs
):
    del kwargs
    weights = weight[0] if isinstance(weight, (list, tuple)) else weight
    scales = (
        weight_scale[0]
        if isinstance(weight_scale, (list, tuple))
        else weight_scale
    )
    if _GROUPED_BI_GRAPH_NATIVE:
        return _grouped_gmm1_graph_bi(
            x=x,
            weights=weights,
            scales=scales,
            x_scale=x_scale,
            group_list=group_list,
        )

    experts, _, columns = weights.shape
    offsets = _group_offsets(group_list, 0, x.shape[0])
    values_bf16 = _dequant_activation(x, x_scale, group_size=32).to(
        torch.bfloat16
    )
    hidden = torch.zeros(
        x.shape[0], columns, dtype=torch.bfloat16, device=x.device
    )
    for expert in range(experts):
        start, end = offsets[expert], offsets[expert + 1]
        if end <= start:
            continue
        expert_weight = _grouped_weight_bf16(
            weights, scales, expert, 32, "gmm1"
        )
        hidden[start:end] = torch.ops.batch_invariant_ops.npu_mm_batch_invariant(
            values_bf16[start:end].contiguous(), expert_weight.contiguous()
        )
    activated = torch_npu.npu_swiglu(hidden)
    return torch_npu.npu_dynamic_mx_quant(
        activated, dst_type=torch.float8_e4m3fn
    )


def _install_grouped_bi_decompose() -> None:
    global _GROUPED_BI_NOTICE_PRINTED
    import batch_invariant_ops  # noqa: F401

    torch_npu.npu_grouped_matmul = _grouped_gmm2_bi
    torch_npu.npu_grouped_matmul_swiglu_quant_v2 = _grouped_gmm1_bi
    if not _GROUPED_BI_NOTICE_PRINTED:
        implementation = (
            "dequantized bf16 graph-safe grouped matmul"
            if _GROUPED_BI_GRAPH_NATIVE
            else "per-expert bf16 fixed-order matmul"
        )
        print(f"[BI_MXFP8_GROUPED] {implementation} enabled", flush=True)
        _GROUPED_BI_NOTICE_PRINTED = True


if _GROUPED_BI_DECOMPOSE:
    _install_grouped_bi_decompose()


@register_scheme("W8A8_MXFP8", "linear")
class AscendW8A8MXFP8DynamicLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W8A8_MXFP8 (Microscaling FP8) quantization.

    This scheme uses microscaling FP8 quantization with per-group scales.
    The activation is dynamically quantized to FP8 (E4M3FN format) with
    microscaling, and weights are stored in FP8 format with per-group scales.
    """

    model_dtype = None

    def __init__(self):
        ensure_mxfp8_linear_available("W8A8_MXFP8 linear quantization")
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn)}
        return params_dict

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, cdiv(input_size, self.group_size), dtype=torch.uint8)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        if isinstance(x, tuple):
            quantized_x, pertoken_scale = x
            original_shape = quantized_x.shape
            output_dtype = torch.bfloat16
        else:
            # reshape x for Qwen VL models
            original_shape = x.shape
            if x.dim() > 2:
                x = x.view(-1, x.shape[-1])
            quantized_x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
            output_dtype = x.dtype

        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)

        if _DENSE_BI_DECOMPOSE:
            output = _dense_bi_matmul(
                layer,
                quantized_x,
                pertoken_scale,
                self.group_size,
                output_dtype,
                bias,
            )
        else:
            output = torch_npu.npu_quant_matmul(
                quantized_x,
                layer.weight,
                layer.weight_scale,
                scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                pertoken_scale=pertoken_scale,
                pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                bias=bias,
                output_dtype=output_dtype,
                group_sizes=[1, 1, self.group_size],
            )
        # reshape output for Qwen VL models
        if len(original_shape) > 2:
            output = output.view(*original_shape[:-1], -1)

        return output

    def process_weights_after_loading(self, layer):
        """Process weights after loading for MXFP8 inference.

        This method transforms weights for NPU MXFP8 computation:
        - weight: (output_size, input_size) -> (input_size, output_size)
        - weight_scale: (n_dim, k_dim) -> (k_dim//2, n_dim, 2)

        For RL training scenarios where weights need to be reloaded multiple times,
        this method stores original shapes and can be called multiple times safely.
        Use restore_weights_for_rl_loading() before weight reload, then call this
        method again after loading.

        Address stability for ACL graph:
        The transformed buffer is what the ACL graph captures and replays. It is
        allocated once and cached on the layer; subsequent calls copy the
        (re)loaded original-shape data in place into the cached buffer so its
        data_ptr never changes across RL weight reloads.
        """

        # Check if already transformed to avoid double transformation
        if getattr(layer, "_mxfp8_transformed", False):
            return

        # Store original shapes for RL weight reloading
        # Only store on first call (when shapes are in original format)
        if not hasattr(layer, "_mxfp8_original_shapes"):
            layer._mxfp8_original_shapes = {
                "weight": tuple(layer.weight.data.shape),
                "weight_scale": tuple(layer.weight_scale.data.shape),
            }

        n_dim, k_dim = layer.weight_scale.data.shape
        # Shape should be padded if it cannot be divided by 2
        if layer.weight_scale.data.shape[-1] % 2 != 0:
            reshaped_scale = F.pad(layer.weight_scale.data, (0, 1), mode="constant", value=0)
            reshaped_scale = reshaped_scale.reshape(n_dim, k_dim // 2 + 1, 2)
        else:
            reshaped_scale = layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)
        target_scale = reshaped_scale.transpose(0, 1)

        if not hasattr(layer, "_mxfp8_weight_buf"):
            # First call: allocate the persistent transformed buffers.
            layer._mxfp8_weight_buf = layer.weight.data.transpose(0, 1).contiguous()
            layer._mxfp8_scale_buf = target_scale.contiguous()
        else:
            # Subsequent calls (RL reload path): copy in place to keep data_ptr stable.
            layer._mxfp8_weight_buf.copy_(layer.weight.data.transpose(0, 1).contiguous())
            layer._mxfp8_scale_buf.copy_(target_scale.contiguous())

        layer.weight.data = layer._mxfp8_weight_buf
        layer.weight_scale.data = layer._mxfp8_scale_buf

        # Mark as transformed
        layer._mxfp8_transformed = True

    def restore_weights_for_rl_loading(self, layer):
        """Restore weights to original shapes for RL weight reloading.

        This method must be called BEFORE model.load_weights() in RL training
        loops to restore the tensors to their original shapes that the weight
        loader expects.

        After weight loading, call process_weights_after_loading() again to
        re-apply the MXFP8 transformations.

        Shape transformations reversed:
        - weight: (input_size, output_size) -> (output_size, input_size)
        - weight_scale: (k_dim//2, n_dim, 2) -> (n_dim, k_dim)
        """

        if not getattr(layer, "_mxfp8_transformed", False):
            # Not transformed, nothing to restore
            return

        if not hasattr(layer, "_mxfp8_original_shapes"):
            err_msg = (
                "[vllm-ascend/W8A8_MXFP8] Cannot restore weights: original "
                "shapes not recorded. "
                "This should not happen if process_weights_after_loading was called first."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        orig_shapes = layer._mxfp8_original_shapes
        orig_scale_shape = orig_shapes["weight_scale"]

        # Restore weight: (input_size, output_size) -> (output_size, input_size)
        target_weight = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = layer.weight.data.transpose(0, 1)
        layer.weight.data.copy_(target_weight)

        # Restore weight_scale: (k_dim//2, n_dim, 2) -> (n_dim, k_dim)
        # Current shape: (k_dim//2, n_dim, 2)
        # Target shape: (n_dim, k_dim)
        target_scale = layer.weight_scale.data.transpose(0, 1).reshape(orig_scale_shape).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1).reshape(orig_scale_shape)
        layer.weight_scale.data.copy_(target_scale)

        # Mark as not transformed (ready for weight loading)
        layer._mxfp8_transformed = False


@register_scheme("W8A8_MXFP8", "moe")
class AscendW8A8MXFP8DynamicFusedMoEMethod(AscendMoEScheme):
    """FusedMoe method for Ascend W8A8_MXFP8."""

    model_dtype = None
    quant_type: QuantType = QuantType.W8A8MXFP

    def __init__(self):
        ensure_mxfp8_moe_available("W8A8_MXFP8 MoE quantization")

        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)
        ascend_config = get_ascend_config()
        self.use_aclgraph = (
            vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not vllm_config.model_config.enforce_eager
        )
        self.dynamic_eplb = ascend_config.eplb_config.dynamic_eplb

    @staticmethod
    def get_weight(
        num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes, dtype=torch.float8_e4m3fn
        )
        param_dict["w2_weight"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition, dtype=torch.float8_e4m3fn
        )
        return param_dict

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // self.group_size, dtype=torch.uint8
        )

        param_dict["w2_weight_scale"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.group_size, dtype=torch.uint8
        )
        return param_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = True,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_shared_experts = getattr(layer, "n_shared_experts", 0)
        if num_shared_experts is None:
            num_shared_experts = 0
        num_logical_experts = get_moe_num_logical_experts(
            layer,
            num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=num_shared_experts,
        )
        assert router_logits.shape[1] == num_logical_experts, "Number of global experts mismatch (excluding redundancy)"
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_logical_experts,
            tid2eid=tid2eid,
        )

        if topk_weights is None or topk_ids is None:
            raise RuntimeError("topk_weights and topk_ids must be set before fused MoE execution.")

        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0), num_logical_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        if x.dtype not in [torch.float8_e4m3fn]:
            topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                mxfp_act_quant_type=torch.float8_e4m3fn,
                mxfp_weight_quant_type=torch.float8_e4m3fn,
                mxfp_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                mxfp_per_token_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                mxfp_use_bf16=(x.dtype in [torch.bfloat16, torch.float8_e4m3fn]),
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                swiglu_limit=layer.swiglu_limit,
            )
        )

    def process_weights_after_loading(self, layer):
        """Process weights after loading for MXFP8 inference.

        This method transforms weights for NPU MXFP8 computation:
        - w13_weight: (g_num, n_size, k_size) -> (g_num, k_size, n_size)
        - w2_weight: (g_num, n_size, k_size) -> (g_num, k_size, n_size)
        - w13_weight_scale: (g_num, n_size, k_size) -> (g_num, k_size//2, n_size, 2)
        - w2_weight_scale: (g_num, n_size, k_size) -> (g_num, k_size//2, n_size, 2)

        For RL training scenarios where weights need to be reloaded multiple times,
        this method stores original shapes and can be called multiple times safely.
        Use restore_weights_for_rl_loading() before weight reload, then call this
        method again after loading.
        """

        # Check if already transformed to avoid double transformation
        if getattr(layer, "_mxfp8_transformed", False):
            return

        # Store original shapes for RL weight reloading
        # Only store on first call (when shapes are in original format)
        if not hasattr(layer, "_mxfp8_original_shapes"):
            layer._mxfp8_original_shapes = {
                "w13_weight": tuple(layer.w13_weight.data.shape),
                "w13_weight_scale": tuple(layer.w13_weight_scale.data.shape),
                "w2_weight": tuple(layer.w2_weight.data.shape),
                "w2_weight_scale": tuple(layer.w2_weight_scale.data.shape),
            }

        def _transform_scale(scale: torch.Tensor) -> torch.Tensor:
            g_num, n_size, k_size = scale.shape
            if k_size % 2:
                if not _GROUPED_BI_DECOMPOSE:
                    raise RuntimeError(
                        "MXFP8 grouped scale packing requires an even group count; "
                        "enable VLLM_MXFP8_GROUPED_BI_DECOMPOSE for the "
                        "deterministic unpacked path"
                    )
                return scale.transpose(1, 2)
            return scale.reshape(g_num, n_size, k_size // 2, 2).transpose(1, 2)

        layer.w13_weight_scale.data = _transform_scale(layer.w13_weight_scale.data)
        layer.w2_weight_scale.data = _transform_scale(layer.w2_weight_scale.data)
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)

        # Mark as transformed
        layer._mxfp8_transformed = True

    def restore_weights_for_rl_loading(self, layer):
        """Restore weights to original shapes for RL weight reloading.

        This method must be called BEFORE model.load_weights() in RL training
        loops to restore the tensors to their original shapes that the weight
        loader expects.

        After weight loading, call process_weights_after_loading() again to
        re-apply the MXFP8 transformations.

        Shape transformations reversed:
        - w13_weight: (g_num, k_size, n_size) -> (g_num, n_size, k_size)
        - w2_weight: (g_num, k_size, n_size) -> (g_num, n_size, k_size)
        - w13_weight_scale: (g_num, k_size//2, n_size, 2) -> (g_num, n_size, k_size)
        - w2_weight_scale: (g_num, k_size//2, n_size, 2) -> (g_num, n_size, k_size)
        """

        if not getattr(layer, "_mxfp8_transformed", False):
            # Not transformed, nothing to restore
            return

        if not hasattr(layer, "_mxfp8_original_shapes"):
            err_msg = (
                "[vllm-ascend/W8A8_MXFP8] Cannot restore weights: original "
                "shapes not recorded. "
                "This should not happen if process_weights_after_loading was called first."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        orig_shapes = layer._mxfp8_original_shapes

        def _restore(weight_key: str, scale_key: str):
            """Helper to restore a single MoE weight and its scale using safe memory copies."""
            # --- 1. Restore Weight ---
            weight_tensor = getattr(layer, weight_key)
            target_weight = weight_tensor.data.transpose(1, 2).contiguous()
            weight_tensor.data = weight_tensor.data.transpose(1, 2)
            weight_tensor.data.copy_(target_weight)

            # --- 2. Restore Weight Scale ---
            scale_tensor = getattr(layer, scale_key)
            orig_scale_shape = orig_shapes[scale_key]

            target_scale = scale_tensor.data.transpose(1, 2).reshape(orig_scale_shape).contiguous()
            scale_tensor.data = scale_tensor.data.transpose(1, 2).view(orig_scale_shape)
            scale_tensor.data.copy_(target_scale)

        _restore("w13_weight", "w13_weight_scale")
        _restore("w2_weight", "w2_weight_scale")

        # Mark as not transformed (ready for weight loading)
        layer._mxfp8_transformed = False
