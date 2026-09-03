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
import torch_npu
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.mxfp_compat import (
    FLOAT8_E8M0FNU_DTYPE,
    ensure_mxfp4_linear_available,
)
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input

from .base import AscendLinearScheme, AscendMoEScheme, QuantType, get_moe_num_logical_experts
from .registry import register_scheme

_PARITY_MOE_CALL_INDEX = 0


def _parity_tap(name: str, tensor: torch.Tensor) -> None:
    """Persist opt-in MoE routing tensors for the reduced parity run."""
    output_dir = os.environ.get("KIMI_PARITY_TAP_DIR")
    if not output_dir:
        return
    expected_tokens = int(os.environ.get("KIMI_PARITY_TAP_EXPECTED_TOKENS", "32"))
    if tensor.ndim == 0 or tensor.shape[0] != expected_tokens:
        return
    os.makedirs(output_dir, exist_ok=True)
    torch.save(tensor.detach().cpu().contiguous(), os.path.join(output_dir, f"{name}.pt"))


def _megatron_reference_select_experts(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    scoring_func: str,
    routed_scaling_factor: float,
    expert_bias: torch.Tensor | None,
    use_grouped_topk: bool,
    num_expert_group: int | None,
    topk_group: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror Megatron Core's unfused Kimi top-k routing arithmetic."""
    if scoring_func == "sigmoid":
        scores = torch.sigmoid(router_logits.float())
    elif scoring_func == "softmax":
        scores = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    else:
        raise NotImplementedError(
            f"Kimi parity routing only supports sigmoid and softmax scoring, not {scoring_func!r}."
        )
    selection_scores = scores + expert_bias.float() if scoring_func == "sigmoid" and expert_bias is not None else scores
    if use_grouped_topk:
        if num_expert_group is None or topk_group is None:
            raise ValueError("Grouped Kimi parity routing requires both group counts.")
        num_tokens, num_experts = selection_scores.shape
        group_scores = (
            selection_scores.view(num_tokens, num_expert_group, -1).topk(top_k // topk_group, dim=-1)[0].sum(dim=-1)
        )
        group_ids = torch.topk(
            group_scores,
            k=topk_group,
            dim=-1,
            sorted=False,
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_ids, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, num_expert_group, num_experts // num_expert_group)
            .reshape(num_tokens, -1)
        )
        masked_scores = selection_scores.masked_fill(~score_mask.bool(), float("-inf"))
        _, topk_ids = torch.topk(masked_scores, k=top_k, dim=-1)
        topk_weights = torch.gather(scores, dim=1, index=topk_ids)
    elif scoring_func == "sigmoid" and expert_bias is not None:
        _, topk_ids = torch.topk(
            selection_scores,
            k=top_k,
            dim=-1,
            sorted=torch.is_grad_enabled(),
        )
        topk_weights = torch.gather(scores, dim=1, index=topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(
            scores,
            k=top_k,
            dim=-1,
            sorted=torch.is_grad_enabled(),
        )
    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    if routed_scaling_factor:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.type_as(router_logits), topk_ids.to(torch.int32)


@register_scheme("W4A8_MXFP", "linear")
class AscendW4A8MXFPDynamicLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W4A8_MXFP (Microscaling) quantization."""

    def __init__(self, *, use_weight_packed: bool = False):
        ensure_mxfp4_linear_available("W8A8_MXFP8 linear quantization")
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)

    @staticmethod
    def get_weight(input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size // 2, dtype=torch.uint8)}
        return params_dict

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, input_size // self.group_size, dtype=torch.uint8)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        if isinstance(x, tuple):
            quantized_x, dynamic_scale = x
            output_dtype = torch.bfloat16
        else:
            quantized_x, dynamic_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
            output_dtype = x.dtype

        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=dynamic_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=output_dtype,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            group_sizes=[0, 0, self.group_size],
        )

        return output

    def process_weights_after_loading(self, layer):
        layer.weight.data = torch_npu.npu_format_cast(
            layer.weight.data, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
        )
        layer.weight.data = layer.weight.data.transpose(-1, -2)
        n, k = layer.weight_scale.shape
        layer.weight_scale.data = layer.weight_scale.data.reshape(n, k // 2, 2).transpose(-3, -2)


@register_scheme("W4A8_MXFP", "moe")
class AscendW4A8MXFPDynamicFusedMoEMethod(AscendMoEScheme):
    """FusedMoe method for Ascend W4A8_DYNAMIC."""

    quant_type: QuantType = QuantType.W4A8MXFP

    def __init__(self, *, use_weight_packed: bool = False):
        self.use_weight_packed = use_weight_packed
        self.ep_group = get_ep_group()

        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)
        ascend_config = get_ascend_config()
        self.use_aclgraph = (
            vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not vllm_config.model_config.enforce_eager
        )
        self.dynamic_eplb = ascend_config.eplb_config.dynamic_eplb

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        param_dict = {}

        w13_weight_name = "w13_weight_packed" if self.use_weight_packed else "w13_weight"
        w2_weight_name = "w2_weight_packed" if self.use_weight_packed else "w2_weight"

        param_dict[w13_weight_name] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes // 2,
            dtype=torch.uint8,
        )
        param_dict[w2_weight_name] = torch.empty(
            num_experts,
            hidden_sizes,
            intermediate_size_per_partition // 2,
            dtype=torch.uint8,
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
        global _PARITY_MOE_CALL_INDEX
        if os.environ.get("VLLM_ASCEND_W4A8_EXECUTION_PROOF") == "1":
            logger.info(
                "KIMI_W4A8_EXECUTION_PROOF tokens=%d packed=%s weight=MXFP4_E2M1 activation=MXFP8_E4M3FN scale=E8M0",
                x.shape[0],
                self.use_weight_packed,
            )
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
        if os.environ.get("VLLM_ASCEND_KIMI_REFERENCE_ROUTING") == "1":
            if custom_routing_function is not None:
                raise NotImplementedError("Kimi parity routing does not support a custom routing function.")
            topk_weights, topk_ids = _megatron_reference_select_experts(
                router_logits=router_logits,
                top_k=top_k,
                renormalize=renormalize,
                scoring_func=scoring_func,
                routed_scaling_factor=routed_scaling_factor,
                expert_bias=e_score_correction_bias,
                use_grouped_topk=use_grouped_topk,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
            )
        else:
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
                e_score_correction_bias=e_score_correction_bias,
                routed_scaling_factor=routed_scaling_factor,
                num_experts=num_logical_experts,
                tid2eid=tid2eid,
            )
        expected_tokens = int(os.environ.get("KIMI_PARITY_TAP_EXPECTED_TOKENS", "32"))
        if x.shape[0] == expected_tokens:
            _PARITY_MOE_CALL_INDEX += 1
            if _PARITY_MOE_CALL_INDEX == 1:
                _parity_tap("02_moe_topk_weights", topk_weights)
                _parity_tap("02_moe_topk_ids", topk_ids)

        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0), num_logical_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        if x.dtype not in [torch.float8_e4m3fn]:
            topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = get_forward_context().moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=getattr(
                    layer,
                    "w13_weight_packed" if self.use_weight_packed else "w13_weight",
                ),
                w2=getattr(
                    layer,
                    "w2_weight_packed" if self.use_weight_packed else "w2_weight",
                ),
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
                mxfp_weight_quant_type=torch_npu.float4_e2m1fn_x2,
                mxfp_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                mxfp_per_token_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                mxfp_use_bf16=(x.dtype in [torch.bfloat16, torch.float8_e4m3fn]),
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                swiglu_limit=layer.swiglu_limit,
            )
        )

    def process_weights_after_loading(self, layer):
        w13_weight = getattr(
            layer,
            "w13_weight_packed" if self.use_weight_packed else "w13_weight",
        )
        w2_weight = getattr(
            layer,
            "w2_weight_packed" if self.use_weight_packed else "w2_weight",
        )
        w13_weight.data = torch_npu.npu_format_cast(
            w13_weight.data, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
        )
        w2_weight.data = torch_npu.npu_format_cast(
            w2_weight.data, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
        )
        w13_weight.data = w13_weight.data.transpose(1, 2)
        w2_weight.data = w2_weight.data.transpose(1, 2)
        g, n, k = layer.w13_weight_scale.shape
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.reshape(g, n, k // 2, 2).transpose(-3, -2)
        g, n, k = layer.w2_weight_scale.shape
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.reshape(g, n, k // 2, 2).transpose(-3, -2)
