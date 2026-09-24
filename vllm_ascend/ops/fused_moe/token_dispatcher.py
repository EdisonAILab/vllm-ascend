# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
import os
from abc import ABC, abstractmethod
from typing import Generic

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import get_ep_group, get_tp_group
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import get_dispatch_v2_tokens_capacity, get_mc2_tokens_capacity
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.lora.fused_moe import (
    all2all_lora_indices,
    postprocess_lora_indices,
    preprocess_lora_indices,
)
from vllm_ascend.ops.activation import SituActivationConfig
from vllm_ascend.ops.fused_moe.comm_utils import async_all_to_all, gather_from_sequence_parallel_region
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEAllGatherCombineMetadata,
    MoEAllToAllCombineMetadata,
    MoEMC2CombineMetadata,
    MoETokenDispatchInput,
    MoETokenDispatchOutput,
    TMoECombineMetadata,
)
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import (
    AscendDeviceType,
    get_ascend_device_type,
    should_skip_allreduce_across_dp_group,
)

EXPERT_TOKEN_NUMS_TYPE_CUMSUM = 0
EXPERT_TOKEN_NUMS_TYPE_COUNT = 1

_TRAINING_PARITY = os.getenv("VLLM_ASCEND_TRAINING_PARITY", "0") == "1"
_KIMI_FIXED_ORDER_MOE_TP_REDUCTION = (
    os.getenv(
        "VLLM_ASCEND_KIMI_REFERENCE_TP_MOE_FIXED_ORDER",
        "0",
    )
    == "1"
)


def _kimi_fixed_order_moe_tp_reduce_impl(
    hidden_states: torch.Tensor,
    group_name: str,
    world_size: int,
    destination_rank: int,
) -> torch.Tensor:
    """Reduce expert-TP contributions in an explicit rank order."""
    from vllm.distributed.parallel_state import _groups

    if group_name not in _groups:
        raise ValueError(f"tensor-parallel group {group_name!r} is unavailable")
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"tensor-parallel group {group_name!r} is destroyed")
    if group.world_size != world_size:
        raise ValueError(
            "Kimi fixed-order MoE reduction world-size mismatch: "
            f"group={group.world_size}, requested={world_size}"
        )
    if not 0 <= destination_rank < world_size:
        raise ValueError(
            f"Kimi fixed-order MoE destination rank {destination_rank} "
            f"is outside [0, {world_size})"
        )
    if hidden_states.shape[0] % world_size:
        raise ValueError(
            "Kimi fixed-order MoE input rows must be divisible by TP size: "
            f"rows={hidden_states.shape[0]}, tp={world_size}"
        )

    assignments_per_source_rank = hidden_states.shape[0] // world_size
    local_chunk = hidden_states.reshape(
        world_size,
        assignments_per_source_rank,
        *hidden_states.shape[1:],
    )[destination_rank].contiguous()
    gathered = group._all_gather_out_place(
        local_chunk.unsqueeze(0),
        0,
    )
    contributions = gathered.reshape(
        world_size,
        assignments_per_source_rank,
        *hidden_states.shape[1:],
    )
    rank_order = list(range(world_size))
    reduced = contributions[rank_order[0]].float().contiguous()
    for rank in rank_order[1:]:
        reduced.add_(contributions[rank].float())
    return reduced


def _kimi_fixed_order_moe_tp_reduce_fake(
    hidden_states: torch.Tensor,
    group_name: str,
    world_size: int,
    destination_rank: int,
) -> torch.Tensor:
    del group_name, destination_rank
    return torch.empty(
        (hidden_states.shape[0] // world_size, *hidden_states.shape[1:]),
        dtype=torch.float32,
        device=hidden_states.device,
    )


if _KIMI_FIXED_ORDER_MOE_TP_REDUCTION and not hasattr(
    torch.ops.vllm,
    "kimi_fixed_order_moe_tp_reduce",
):
    direct_register_custom_op(
        op_name="kimi_fixed_order_moe_tp_reduce",
        op_func=_kimi_fixed_order_moe_tp_reduce_impl,
        fake_impl=_kimi_fixed_order_moe_tp_reduce_fake,
        dispatch_key="PrivateUse1",
    )


def _get_expert_token_nums_type(token_dispatch_input: MoETokenDispatchInput) -> int:
    # grouped_matmul_swiglu_quant_v2 consumes per-expert counts; existing
    # MC2 grouped-matmul paths consume prefix sums.
    if token_dispatch_input.quant.use_w4a8_per_channel_gmm_swiglu:
        return EXPERT_TOKEN_NUMS_TYPE_COUNT
    return EXPERT_TOKEN_NUMS_TYPE_CUMSUM


class MoETokenDispatcher(ABC, Generic[TMoECombineMetadata]):
    def __init__(self, **kwargs) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self.top_k = kwargs.get("top_k", 0)
        self.num_experts = kwargs.get("num_experts", 0)
        self.lora_context = None

    def set_lora_context(self, lora_context) -> None:
        self.lora_context = lora_context

    @property
    def ep_group(self):
        """Get expert model parallel group."""
        return get_ep_group().device_group

    @property
    def ep_rank(self):
        return get_ep_group().rank_in_group

    @property
    def ep_size(self):
        return get_ep_group().world_size

    @abstractmethod
    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ) -> MoETokenDispatchOutput[TMoECombineMetadata]:
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_combine(
        self,
        hidden_states: torch.Tensor,
        combine_metadata: TMoECombineMetadata,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Combine function not implemented.")


class TokenDispatcherWithMC2(MoETokenDispatcher[MoEMC2CombineMetadata]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        device_group = get_mc2_group().device_group
        # TODO: Try local_rank = ep_group.rank_in_group
        local_rank = torch.distributed.get_rank(group=device_group)
        backend = device_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        self.ep_rank_id = get_mc2_group().rank_in_group
        self.ep_world_size = get_mc2_group().world_size
        self.enable_dispatch_v2 = hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")
        self.need_extra_args = get_ascend_device_type() in [AscendDeviceType.A3, AscendDeviceType.A5]
        self.a5_need_extra_args = get_ascend_device_type() == AscendDeviceType.A5
        self.mc2_comm_alg = get_ascend_config().get_mc2_comm_alg()

        # When enable hierarchical communication or A5 case, param `expert_scales` need to be passed in.
        self.need_expert_scale = self.a5_need_extra_args or self.mc2_comm_alg == "hierarchy"

        # Here we need to calculate the global_bs = max_bs_per_rank * ep_world_size to execute
        # dispatch & combine operators with different input num_tokens per rank.
        vllm_config = get_current_vllm_config()
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        mc2_tokens_capacity = get_mc2_tokens_capacity()
        if not kwargs.get("is_fused_mc2", False):
            mc2_tokens_capacity = get_dispatch_v2_tokens_capacity() or mc2_tokens_capacity
        num_tokens_per_tp_rank = mc2_tokens_capacity // tp_size
        # Surface the per-rank capacity for CANN MegaMoe's get_symm_buffer
        # sizing (used by FusedMC2CommImpl._get_cann_symm_buffer). Without
        # this, MegaMoe falls back to hidden_states.shape[0] which jitters
        # under eager mode and forces sym-buffer rebuilds every step.
        self.max_num_tokens_per_rank = num_tokens_per_tp_rank
        _max_global_bs = num_tokens_per_tp_rank * self.ep_world_size

        # When allreduce across DP is not skipped, tokens are uniform across ranks:
        # use global_bs=0 (uniform mode) and pass mc2_mask.
        # When allreduce is skipped, tokens may differ per rank:
        # use the real global_bs and do NOT pass mc2_mask.
        self.global_bs = _max_global_bs if should_skip_allreduce_across_dp_group(vllm_config) else 0

        if not self.enable_dispatch_v2 and self.mc2_comm_alg == "hierarchy":
            raise RuntimeError(
                "PTA and CANN version is too old to support mc2 hierarchy comm, please upgrade your version."
            )

    def refresh_hccl_group(self) -> None:
        """Refresh MC2 communicator metadata after HCCL groups are recreated."""
        device_group = get_mc2_group().device_group
        local_rank = torch.distributed.get_rank(group=device_group)
        backend = device_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)

    def get_dispatch_mc2_kwargs(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids
        expert_map = token_dispatch_input.routing.expert_map
        global_redundant_expert_num = token_dispatch_input.routing.global_redundant_expert_num
        comm_quant_mode = token_dispatch_input.quant.comm_quant_mode

        assert expert_map is not None, "expert_map is required for MC2 token dispatch."
        # NOTE: quant_mode differs by quant feature:
        # - Legacy int communication quantization uses quant_mode=2.
        # - A5 MXFP communication uses quant_mode=4.
        if comm_quant_mode is not None:
            quant_mode = comm_quant_mode
        elif token_dispatch_input.quant.dispatch_with_quant:
            quant_mode = 4 if self.a5_need_extra_args and token_dispatch_input.quant.is_mxfp else 2
        else:
            quant_mode = 0
        self.moe_expert_num = len(expert_map) + global_redundant_expert_num
        expert_token_nums_type = _get_expert_token_nums_type(token_dispatch_input)
        kwargs_mc2 = {
            "x": hidden_states,
            "expert_ids": topk_ids,
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.moe_expert_num,
            "global_bs": self.global_bs,
            "expert_token_nums_type": expert_token_nums_type,
        }
        if self.global_bs == 0:
            kwargs_mc2["x_active_mask"] = token_dispatch_input.routing.mc2_mask

        stage1_kwargs = {
            "scales": None,
            "quant_mode": quant_mode,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
            "comm_alg": self.mc2_comm_alg,
        }
        if self.need_extra_args:
            stage1_kwargs.update(
                {
                    "group_tp": self.moe_all_to_all_group_name,
                    "tp_world_size": 1,
                    "tp_rank_id": 0,
                }
            )
        # Only dispatch-enabled MXFP paths pass y_dtype through MC2.
        if (
            self.a5_need_extra_args
            and (token_dispatch_input.quant.is_mxfp or token_dispatch_input.quant.is_fp8)
            and token_dispatch_input.quant.dispatch_with_quant
        ):
            y_dtype = torch.float8_e4m3fn
            if (
                token_dispatch_input.quant.mxfp is not None
                and token_dispatch_input.quant.mxfp.act_quant_type is not None
            ):
                y_dtype = token_dispatch_input.quant.mxfp.act_quant_type
            stage1_kwargs.update({"tp_world_size": 1, "tp_rank_id": 0, "y_dtype": y_dtype})
        if self.need_expert_scale:
            stage1_kwargs.update(
                {
                    "expert_scales": topk_weights.to(torch.float32),
                }
            )

        kwargs_mc2.update(stage1_kwargs)
        return kwargs_mc2

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        kwargs_mc2 = self.get_dispatch_mc2_kwargs(token_dispatch_input)
        output = (
            torch_npu.npu_moe_distribute_dispatch_v2(**kwargs_mc2)
            if self.enable_dispatch_v2
            else torch_npu.npu_moe_distribute_dispatch(**kwargs_mc2)
        )
        # comm_stream.wait_stream(torch.npu.current_stream())
        (
            expand_x,
            dynamic_scale,
            assist_info_for_combine,
            expert_token_nums,
            ep_recv_counts,
            tp_recv_counts,
            expand_scales,
        ) = output[0:7]

        group_list_type = kwargs_mc2["expert_token_nums_type"]
        return MoETokenDispatchOutput(
            hidden_states=expand_x,
            dynamic_scale=dynamic_scale,
            group_list=expert_token_nums,
            group_list_type=group_list_type,
            combine_metadata=MoEMC2CombineMetadata(
                topk_ids=token_dispatch_input.topk_ids,
                topk_weights=token_dispatch_input.topk_weights,
                expert_map=token_dispatch_input.routing.expert_map,
                ep_recv_counts=ep_recv_counts,
                tp_recv_counts=tp_recv_counts,
                assist_info_for_combine=assist_info_for_combine,
                expand_scales=expand_scales,
                quant=token_dispatch_input.quant,
                mc2_mask=token_dispatch_input.routing.mc2_mask if self.global_bs == 0 else None,
            ),
        )

    def get_combine_mc_kwargs(self, hidden_states: torch.Tensor, combine_metadata: MoEMC2CombineMetadata):
        expert_map = combine_metadata.expert_map
        topk_ids = combine_metadata.topk_ids
        topk_weights = combine_metadata.topk_weights
        ep_recv_counts = combine_metadata.ep_recv_counts
        tp_recv_counts = combine_metadata.tp_recv_counts
        assist_info_for_combine = combine_metadata.assist_info_for_combine
        expand_scales = combine_metadata.expand_scales
        quant_type = combine_metadata.quant.quant_type
        comm_quant_mode = combine_metadata.quant.comm_quant_mode

        assert expert_map is not None
        # NOTE: quant_mode differs by quant features:
        # - A5 MXFP communication uses quant_mode=4 only for W8A8MXFP currently.
        if comm_quant_mode is not None:
            quant_mode = comm_quant_mode
        elif quant_type == QuantType.W8A8MXFP:
            quant_mode = 4
        else:
            quant_mode = 0
        kwargs_mc2 = {
            "expand_x": hidden_states,
            "expert_ids": topk_ids,
            "expert_scales": topk_weights.to(torch.float32),
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.moe_expert_num,
            "global_bs": self.global_bs,
        }
        if self.global_bs == 0:
            kwargs_mc2["x_active_mask"] = combine_metadata.mc2_mask

        if combine_metadata.quant.dispatch_with_quant:
            tp_recv_counts = torch.empty(1, dtype=torch.int32, device=hidden_states.device)

        stage3_kwargs = {
            "ep_send_counts": ep_recv_counts,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
            "expand_scales": expand_scales,
            "comm_quant_mode": quant_mode,
            "comm_alg": self.mc2_comm_alg,
        }

        if self.enable_dispatch_v2:
            stage3_kwargs["assist_info_for_combine"] = assist_info_for_combine
        else:
            stage3_kwargs["expand_idx"] = assist_info_for_combine

        if self.need_extra_args:
            stage3_kwargs.update(
                {
                    "tp_send_counts": tp_recv_counts,
                    "group_tp": self.moe_all_to_all_group_name,
                    "tp_world_size": 1,
                    "tp_rank_id": 0,
                }
            )

        kwargs_mc2.update(stage3_kwargs)
        return kwargs_mc2

    def token_combine(self, hidden_states, combine_metadata, bias=None):
        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."

        kwargs_mc2 = self.get_combine_mc_kwargs(hidden_states, combine_metadata)
        combined_output = (
            torch_npu.npu_moe_distribute_combine_v2(**kwargs_mc2)
            if self.enable_dispatch_v2
            else torch_npu.npu_moe_distribute_combine(**kwargs_mc2)
        )

        return combined_output


class TokenDispatcherWithAllGather(MoETokenDispatcher[MoEAllGatherCombineMetadata]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_num_tokens = kwargs.get("max_num_tokens")
        num_experts_local = kwargs.get("num_local_experts", 0)
        self.num_experts_local = (
            num_experts_local.item() if torch.is_tensor(num_experts_local) else int(num_experts_local)
        )
        self._assignment_reduce_enabled = False
        self._kimi_tp_moe_replication_enabled = False
        self._kimi_tp_moe_restore_order = None
        self._kimi_tp_moe_local_restore_order = None
        self._kimi_tp_moe_rank_major_token_indices = None
        self._kimi_tp_moe_precombine_enabled = False
        self._kimi_tp_moe_fixed_order_enabled = False
        self._reduce_scatter_enabled = False
        self._assignment_reduce_call_index = 0
        self._assignment_reduce_tap_done = False

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        quant_type = token_dispatch_input.quant.quant_type
        dynamic_scale = token_dispatch_input.routing.pertoken_scale
        unquantized_mxfp4_dispatch = quant_type == QuantType.W4A4MXFP and dynamic_scale is None
        # Without prepare-stage scales, MXFP4 stays unquantized in dispatch and
        # is quantized again inside the MLP path.
        with_quant = token_dispatch_input.quant.dispatch_with_quant and quant_type != QuantType.W8A8FP
        with_quant = with_quant and not unquantized_mxfp4_dispatch
        is_mxfp = token_dispatch_input.quant.is_mxfp
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids
        expert_map = token_dispatch_input.routing.expert_map
        act_quant_type = (
            token_dispatch_input.quant.mxfp.act_quant_type
            if token_dispatch_input.quant.mxfp is not None and not unquantized_mxfp4_dispatch
            else None
        )
        global_redundant_expert_num = token_dispatch_input.routing.global_redundant_expert_num
        restore_shape = hidden_states.shape
        # Fuse the first dynamic quant of moe_mlp into initrouting when
        # dispatch_with_quant is on but got a None dynamic_scale.
        if with_quant and dynamic_scale is None:
            if quant_type == QuantType.W4A4MXFP:
                quant_mode = 9
            else:
                quant_mode = 3 if is_mxfp else 1
        else:
            quant_mode = -1

        num_tokens = hidden_states.shape[:-1].numel()
        is_situ_w4a8_mxfp = quant_type == QuantType.W4A8MXFP and isinstance(
            token_dispatch_input.activation, SituActivationConfig
        )
        kimi_tp_moe_reduction = (
            is_situ_w4a8_mxfp
            and os.environ.get("VLLM_ASCEND_KIMI_REFERENCE_TP_MOE_REDUCTION", "0") == "1"
        )
        self._assignment_reduce_enabled = _TRAINING_PARITY or kimi_tp_moe_reduction
        self._kimi_tp_moe_replication_enabled = (
            kimi_tp_moe_reduction and get_tp_group().world_size > 1
        )
        self._kimi_tp_moe_precombine_enabled = (
            self._kimi_tp_moe_replication_enabled
            and os.environ.get(
                "VLLM_ASCEND_KIMI_REFERENCE_TP_MOE_PRECOMBINE_REDUCTION",
                "0",
            )
            == "1"
        )
        self._kimi_tp_moe_fixed_order_enabled = (
            self._kimi_tp_moe_replication_enabled
            and _KIMI_FIXED_ORDER_MOE_TP_REDUCTION
        )
        if (
            self._kimi_tp_moe_fixed_order_enabled
            and self._kimi_tp_moe_precombine_enabled
        ):
            raise ValueError(
                "Kimi TP MoE fixed-order and precombine reductions are "
                "mutually exclusive"
            )
        self._reduce_scatter_enabled = (
            is_situ_w4a8_mxfp
            and os.environ.get(
                "VLLM_ASCEND_KIMI_REFERENCE_TP_MOE_REDUCE_SCATTER", "0"
            )
            == "1"
        )
        if self._assignment_reduce_enabled and self._reduce_scatter_enabled:
            raise ValueError("Kimi TP MoE reduction reference modes are mutually exclusive")
        if self._assignment_reduce_enabled:
            # vLLM's EP group also spans TP ranks for MoE models when expert
            # parallelism is disabled.  expert_map, rather than ep_size,
            # distinguishes genuinely sharded experts from TP-local shards.
            if expert_map is not None:
                raise ValueError("training parity AllGather MoE does not support expert-sharded routing")
            flat_experts = topk_ids.reshape(-1)
            # Ascend executes integer argsort on AiCPU. Expert ids are small
            # exact integers in FP32, so this keeps the same stable ordering
            # on AiCore and avoids TP rank-progress skew at TP8.
            assignment_order = torch.argsort(flat_experts.float(), stable=True)
            token_indices = torch.arange(num_tokens, device=flat_experts.device, dtype=torch.long).repeat_interleave(
                self.top_k
            )
            self._training_parity_sorted_token_indices = token_indices.index_select(0, assignment_order)
            # assignment_order maps expert-sorted positions to the original
            # token-major top-k positions.  Megatron's token-unpermute reduces
            # each token in that original top-k order; keep the inverse so the
            # reference path can reproduce the same BF16 accumulation order.
            local_restore_order = torch.empty_like(assignment_order)
            local_restore_order.scatter_(
                0,
                assignment_order,
                torch.arange(
                    assignment_order.numel(),
                    dtype=assignment_order.dtype,
                    device=assignment_order.device,
                ),
            )
            self._kimi_tp_moe_local_restore_order = local_restore_order
        apply_router_weight_on_input = token_dispatch_input.routing.apply_router_weight_on_input
        if apply_router_weight_on_input:
            assert topk_weights.dim() == 2, "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert topk == 1, "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        if expert_map is not None:
            global_num_experts = len(expert_map) + global_redundant_expert_num
            mask = expert_map[topk_ids] != -1
            topk_weights = topk_weights * mask
            first_expert_idx = get_ep_group().rank_in_group * self.num_experts_local
            last_expert_idx = first_expert_idx + self.num_experts_local
        else:
            first_expert_idx = 0
            last_expert_idx = self.num_experts_local
            global_num_experts = self.num_experts_local
        # The fused routing quantizer uses a different E8M0 scale-rounding
        # policy from npu_dynamic_mx_quant. Kimi's SiTU MoE must use the same
        # MXFP8 values as its standalone quantization path, so route in BF16
        # first and quantize the routed rows explicitly.
        explicit_mxfp8_dispatch = is_situ_w4a8_mxfp or (
            os.environ.get("VLLM_ASCEND_KIMI_REFERENCE_MXFP8_DISPATCH") == "1"
            and quant_type == QuantType.W4A8MXFP
        )
        routing_quant_mode = -1 if explicit_mxfp8_dispatch else quant_mode
        sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale = DeviceOperator.npu_moe_init_routing(
            hidden_states,
            topk_ids,
            scale=dynamic_scale,
            active_num=num_tokens * self.top_k,
            expert_num=global_num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[first_expert_idx, last_expert_idx],
            quant_mode=routing_quant_mode,
            act_quant_type=act_quant_type,
        )
        expanded_rows = None
        if self._kimi_tp_moe_replication_enabled:
            if not explicit_mxfp8_dispatch:
                raise ValueError(
                    "Kimi TP MoE pre-expert replication requires explicit MXFP8 dispatch"
                )
            tp_size = get_tp_group().world_size
            num_assignments = sorted_hidden_states.shape[0]
            tp_group = get_tp_group()
            sorted_expert_ids = flat_experts.index_select(0, assignment_order)
            rank_major_hidden_states = tp_group.all_gather(sorted_hidden_states, dim=0)
            rank_major_expert_ids = tp_group.all_gather(sorted_expert_ids, dim=0)
            if self._kimi_tp_moe_precombine_enabled:
                rank_major_token_indices = tp_group.all_gather(
                    self._training_parity_sorted_token_indices, dim=0
                )
                source_rank_offsets = torch.arange(
                    tp_size,
                    dtype=rank_major_token_indices.dtype,
                    device=rank_major_token_indices.device,
                ).repeat_interleave(num_assignments)
                self._kimi_tp_moe_rank_major_token_indices = (
                    rank_major_token_indices + source_rank_offsets * num_tokens
                )
            else:
                self._kimi_tp_moe_rank_major_token_indices = None
            expanded_rows = torch.argsort(rank_major_expert_ids.float(), stable=True)

            # Megatron gathers BF16 assignments before expert activation
            # quantization, preserving each source rank's values, then groups
            # them by expert. Reproduce that order here. Ascend also does not
            # support index_select on an already-MXFP8 tensor.
            sorted_hidden_states = rank_major_hidden_states.index_select(0, expanded_rows)
            expert_tokens = tp_group.all_gather(expert_tokens, dim=0)
            expert_tokens = expert_tokens.reshape(tp_size, -1).sum(dim=0)

            restore_order = torch.empty_like(expanded_rows)
            restore_order.scatter_(
                0,
                expanded_rows,
                torch.arange(
                    expanded_rows.numel(),
                    dtype=torch.long,
                    device=expanded_rows.device,
                ),
            )
            self._kimi_tp_moe_restore_order = restore_order
        else:
            self._kimi_tp_moe_restore_order = None

        if explicit_mxfp8_dispatch:
            sorted_hidden_states, dynamic_scale = torch_npu.npu_dynamic_mx_quant(
                sorted_hidden_states,
                axis=-1,
                dst_type=act_quant_type,
            )
            dynamic_scale = DeviceOperator.maybe_normalize_mxfp_scale_layout(dynamic_scale)
        expert_tokens = expert_tokens.to(torch.int64)
        group_list_type = 1  # `count` mode

        topk_scales = None
        combine_topk_weights = topk_weights
        if _TRAINING_PARITY:
            if with_quant:
                raise ValueError("training parity MoE currently supports BF16 only")
            flat_weights = topk_weights.reshape(-1)
            sorted_weights = torch.empty_like(flat_weights)
            sorted_weights.scatter_(0, expanded_row_idx.abs().long(), flat_weights)
            topk_scales = sorted_weights.unsqueeze(-1)
            combine_topk_weights = torch.ones_like(topk_weights)
        elif is_situ_w4a8_mxfp:
            sorted_indices = torch.argsort(expanded_row_idx.float())
            topk_scales = topk_weights.reshape(-1)[sorted_indices].unsqueeze(-1)
            combine_topk_weights = torch.ones_like(topk_weights)
        if expanded_rows is not None and topk_scales is not None:
            topk_scales = get_tp_group().all_gather(topk_scales, dim=0)
            topk_scales = topk_scales.index_select(0, expanded_rows)

        return MoETokenDispatchOutput(
            hidden_states=sorted_hidden_states,
            dynamic_scale=dynamic_scale if with_quant else None,
            group_list=expert_tokens,
            group_list_type=group_list_type,
            topk_scales=topk_scales,
            combine_metadata=MoEAllGatherCombineMetadata(
                topk_weights=combine_topk_weights,
                expanded_row_idx=expanded_row_idx,
                restore_shape=restore_shape,
            ),
        )

    def token_combine(self, hidden_states, combine_metadata, bias=None):
        if self._reduce_scatter_enabled:
            if bias is not None:
                raise ValueError("Kimi TP MoE reduce-scatter does not support bias")
            final_hidden_states = DeviceOperator.npu_moe_token_unpermute(
                permuted_tokens=hidden_states,
                sorted_indices=combine_metadata.expanded_row_idx,
                probs=combine_metadata.topk_weights,
            )
            tp_group = get_tp_group()
            tap_dir = os.environ.get("KIMI_TP_MOE_TAP_DIR")
            expected_tokens = int(
                os.environ.get("KIMI_TP_MOE_TAP_EXPECTED_TOKENS", "-1")
            )
            should_tap = (
                tap_dir is not None
                and not self._assignment_reduce_tap_done
                and combine_metadata.restore_shape[0] == expected_tokens
            )
            if should_tap:
                os.makedirs(tap_dir, exist_ok=True)
                rank = tp_group.rank_in_group
                torch.save(
                    hidden_states.detach().cpu().contiguous(),
                    os.path.join(tap_dir, f"rank_{rank:02d}_assignments_before.pt"),
                )
                torch.save(
                    final_hidden_states.detach().cpu().contiguous(),
                    os.path.join(tap_dir, f"rank_{rank:02d}_combined_before.pt"),
                )
            if tp_group.world_size > 1:
                repeated = final_hidden_states.repeat(
                    (tp_group.world_size,) + (1,) * (final_hidden_states.ndim - 1)
                ).contiguous()
                reduced = torch.empty_like(final_hidden_states)
                torch.distributed.reduce_scatter_tensor(
                    reduced,
                    repeated,
                    group=tp_group.device_group,
                )
                if should_tap:
                    torch.save(
                        reduced.detach().cpu().contiguous(),
                        os.path.join(tap_dir, f"rank_{rank:02d}_reduce_scatter.pt"),
                    )
                tp_group.broadcast(reduced, src=0)
                if should_tap:
                    torch.save(
                        reduced.detach().cpu().contiguous(),
                        os.path.join(tap_dir, f"rank_{rank:02d}_broadcast.pt"),
                    )
                final_hidden_states = reduced.mul_(1.0 / tp_group.world_size)
            if should_tap:
                self._assignment_reduce_tap_done = True
            if len(combine_metadata.restore_shape) == 3:
                final_hidden_states = final_hidden_states.view(combine_metadata.restore_shape)
            return final_hidden_states
        if self._assignment_reduce_enabled:
            tp_group = get_tp_group()
            logical_tokens = int(combine_metadata.restore_shape[:-1].numel())
            output_dtype = hidden_states.dtype
            fp32_assignment_combine = (
                self._kimi_tp_moe_replication_enabled
                and os.environ.get(
                    "VLLM_ASCEND_KIMI_REFERENCE_TP_MOE_FP32_COMBINE",
                    "0",
                )
                == "1"
            )
            native_topk_combine = (
                self._kimi_tp_moe_replication_enabled
                and os.environ.get(
                    "VLLM_ASCEND_KIMI_REFERENCE_TP_MOE_NATIVE_TOPK_COMBINE",
                    "0",
                )
                == "1"
            )
            precombine_reduction = self._kimi_tp_moe_precombine_enabled
            if precombine_reduction and fp32_assignment_combine:
                raise ValueError(
                    "Kimi TP MoE precombine and FP32 postcombine are mutually exclusive"
                )
            if self._kimi_tp_moe_fixed_order_enabled and (
                fp32_assignment_combine or native_topk_combine
            ):
                raise ValueError(
                    "Kimi TP MoE fixed-order reduction cannot be combined "
                    "with diagnostic combine modes"
                )
            if tp_group.world_size <= 0 or tp_group.world_size & (
                tp_group.world_size - 1
            ):
                raise ValueError("training parity MoE assignment reduction requires a power-of-two TP size")
            if tp_group.world_size > 1:
                # Megatron expert TP reduces each expert assignment before
                # unpermuting top-k assignments back to tokens.  vLLM's normal
                # path does the same sums in the opposite order.
                tap_dir = os.environ.get("KIMI_TP_MOE_TAP_DIR")
                expected_assignments = int(
                    os.environ.get("KIMI_TP_MOE_TAP_EXPECTED_ASSIGNMENTS", "-1")
                )
                should_tap = (
                    tap_dir is not None
                    and not self._assignment_reduce_tap_done
                    and hidden_states.shape[0] == expected_assignments
                )
                if should_tap:
                    os.makedirs(tap_dir, exist_ok=True)
                    rank = tp_group.rank_in_group
                    torch.save(
                        hidden_states.detach().cpu().contiguous(),
                        os.path.join(tap_dir, f"rank_{rank:02d}_assignments_before.pt"),
                    )
                    torch.save(
                        self._training_parity_sorted_token_indices.detach()
                        .cpu()
                        .contiguous(),
                        os.path.join(tap_dir, f"rank_{rank:02d}_token_indices.pt"),
                    )
                    self._assignment_reduce_tap_done = True
                if self._kimi_tp_moe_replication_enabled:
                    restore_order = self._kimi_tp_moe_restore_order
                    if restore_order is None:
                        raise RuntimeError("missing Kimi TP MoE assignment restore order")
                    hidden_states = hidden_states.index_select(0, restore_order)
                    if should_tap:
                        torch.save(
                            hidden_states.detach().cpu().contiguous(),
                            os.path.join(
                                tap_dir,
                                f"rank_{rank:02d}_reduce_scatter_input.pt",
                            ),
                        )
                    if precombine_reduction:
                        rank_major_token_indices = (
                            self._kimi_tp_moe_rank_major_token_indices
                        )
                        if rank_major_token_indices is None:
                            raise RuntimeError(
                                "missing Kimi TP MoE rank-major token indices"
                            )
                        # Megatron AllGather MoE first unpermutes (and thereby
                        # combines top-k assignments) independently on every
                        # expert-TP rank, then reduce-scatters those token rows.
                        # The legacy vLLM parity path did these operations in
                        # the opposite order; the two are not BF16-associative.
                        rank_major_combined = torch.zeros(
                            (tp_group.world_size * logical_tokens, hidden_states.shape[-1]),
                            dtype=hidden_states.dtype,
                            device=hidden_states.device,
                        )
                        was_enabled = torch.are_deterministic_algorithms_enabled()
                        torch.use_deterministic_algorithms(True)
                        try:
                            rank_major_combined.index_add_(
                                0, rank_major_token_indices, hidden_states
                            )
                        finally:
                            torch.use_deterministic_algorithms(was_enabled)
                        # Megatron's AllGather dispatcher unpermutes the BF16
                        # expert output first, then sends that BF16 tensor
                        # directly through reduce-scatter.  Keeping FP32 here
                        # changes the HCCL reduction's rounding boundary.
                        reduce_scatter_input = rank_major_combined.contiguous()
                    else:
                        reduce_scatter_input = hidden_states.float().contiguous()
                    if self._kimi_tp_moe_fixed_order_enabled:
                        reduced = torch.ops.vllm.kimi_fixed_order_moe_tp_reduce(
                            hidden_states,
                            tp_group.unique_name,
                            tp_group.world_size,
                            tp_group.rank_in_group,
                        )
                    else:
                        reduced = torch.empty(
                            (
                                reduce_scatter_input.shape[0]
                                // tp_group.world_size,
                            )
                            + reduce_scatter_input.shape[1:],
                            dtype=reduce_scatter_input.dtype,
                            device=hidden_states.device,
                        )
                        torch.distributed.reduce_scatter_tensor(
                            reduced,
                            reduce_scatter_input,
                            group=tp_group.device_group,
                        )
                    reduced_bf16 = reduced.to(output_dtype)
                    # Megatron retains FP32 assignment rows through top-k
                    # token combination and casts the combined token once.
                    # Casting each assignment here adds a second BF16
                    # reduction and changes cancellation-heavy coordinates.
                    hidden_states = (
                        reduced if fp32_assignment_combine else reduced_bf16
                    )
                else:
                    hidden_states = tp_group.all_reduce(hidden_states)
                if should_tap:
                    torch.save(
                        hidden_states.detach().cpu().contiguous(),
                        os.path.join(tap_dir, f"rank_{rank:02d}_assignments_after.pt"),
                    )
            if bias is not None:
                raise ValueError("training parity MoE combine does not support bias")
            num_tokens = combine_metadata.restore_shape[:-1].numel()
            if precombine_reduction:
                final_hidden_states = hidden_states
            else:
                sorted_token_indices = self._training_parity_sorted_token_indices
                if native_topk_combine:
                    local_restore_order = self._kimi_tp_moe_local_restore_order
                    if local_restore_order is None:
                        raise RuntimeError("missing Kimi TP MoE local restore order")
                    hidden_states = hidden_states.index_select(0, local_restore_order)
                    sorted_token_indices = sorted_token_indices.index_select(
                        0, local_restore_order
                    )
                final_hidden_states = torch.zeros(
                    (num_tokens, hidden_states.shape[-1]),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                was_enabled = torch.are_deterministic_algorithms_enabled()
                torch.use_deterministic_algorithms(True)
                try:
                    final_hidden_states.index_add_(
                        0, sorted_token_indices, hidden_states
                    )
                finally:
                    torch.use_deterministic_algorithms(was_enabled)
                if fp32_assignment_combine:
                    final_hidden_states = final_hidden_states.to(output_dtype)
            if tp_group.world_size > 1 and not self._kimi_tp_moe_replication_enabled:
                # MoERunner still performs its standard late TP all-reduce.
                # Every rank now has the already-reduced result, so scale by
                # TP here; the later sum of identical power-of-two-scaled BF16
                # values restores the result exactly.
                final_hidden_states.mul_(1.0 / tp_group.world_size)
            if len(combine_metadata.restore_shape) == 3:
                final_hidden_states = final_hidden_states.view(combine_metadata.restore_shape)
            self._assignment_reduce_call_index += 1
            return final_hidden_states
        final_hidden_states = DeviceOperator.npu_moe_token_unpermute(
            permuted_tokens=hidden_states,
            sorted_indices=combine_metadata.expanded_row_idx,
            probs=combine_metadata.topk_weights,
        )
        if len(combine_metadata.restore_shape) == 3:
            final_hidden_states = final_hidden_states.view(combine_metadata.restore_shape)

        # these values are no longer used, so they need to be set to None for memory release.
        return final_hidden_states


class TokenDispatcherWithAll2AllV(MoETokenDispatcher[MoEAllToAllCombineMetadata]):
    """
    The implementation of the AlltoAll-based token dispatcher, which handles token
    dispatching on the sequence level instead of token level. The core of this implementation
    lies in each device dispatching on the entire sequence, with the hidden state being partitioned.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_local_experts = kwargs.get("num_local_experts", 0)

        assert self.num_local_experts > 0, "Expected at least one expert"
        if self.num_local_experts > 1:
            self.expert_ids_per_ep_rank = torch.tensor(
                [i % self.num_local_experts for i in range(self.num_experts)],
                dtype=torch.int32,
                device=torch.npu.current_device(),
            )

        local_expert_indices_offset = self.ep_rank * self.num_local_experts

        self.local_expert_indices = [local_expert_indices_offset + i for i in range(self.num_local_experts)]
        assert len(self.local_expert_indices) == self.num_local_experts, "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1, (
                "local_expert_indices must be continuous"
            )

        # TODO: Try local_rank = ep_group.rank_in_group
        local_rank = torch.distributed.get_rank(group=self.ep_group)
        backend = self.ep_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        use_mxfp_quant = token_dispatch_input.quant.is_mxfp
        with_quant = token_dispatch_input.quant.dispatch_with_quant
        dst_type = token_dispatch_input.quant.get_dst_type
        scale_type = token_dispatch_input.quant.get_scale_type
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids

        (
            permutated_local_input_tokens,
            reversed_local_input_permutation_mapping,
            tokens_per_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            hidden_shape,
            hidden_shape_before_permute,
        ) = self._dispatch_preprocess(hidden_states, topk_ids)

        dynamic_scale_after_all2all = None
        if with_quant:
            permutated_local_input_tokens, dynamic_scale = DeviceOperator.npu_dynamic_quant(
                permutated_local_input_tokens, act_quant_type=dst_type, use_mxfp_quant=use_mxfp_quant
            )
            _, dynamic_scale_after_all2all, permute2_ep_all_to_all_handle = async_all_to_all(
                dynamic_scale, output_splits, input_splits, self.ep_group
            )
            permute2_ep_all_to_all_handle.wait()
            dynamic_scale.untyped_storage().resize_(0)

        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens, output_splits, input_splits, self.ep_group
        )
        permute1_ep_all_to_all_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

        if self.lora_context is not None:
            all2all_lora_indices(
                self.lora_context,
                output_splits=output_splits,
                input_splits=input_splits,
                ep_group=self.ep_group,
            )

        # Postprocess
        global_input_tokens, dynamic_scale_final, reversed_global_input_permutation_mapping = (
            self._dispatch_postprocess(
                global_input_tokens,
                dynamic_scale_after_all2all,
                global_input_tokens_local_experts_indices,
                with_quant,
                dst_type,
                scale_type,
            )
        )

        return MoETokenDispatchOutput(
            hidden_states=global_input_tokens,
            dynamic_scale=dynamic_scale_final,
            group_list=tokens_per_expert,
            group_list_type=1,
            combine_metadata=MoEAllToAllCombineMetadata(
                input_splits=input_splits,
                output_splits=output_splits,
                topk_weights=topk_weights,
                reversed_local_input_permutation_mapping=reversed_local_input_permutation_mapping,
                reversed_global_input_permutation_mapping=reversed_global_input_permutation_mapping,
                hidden_shape=hidden_shape,
                hidden_shape_before_permute=hidden_shape_before_permute,
            ),
        )

    def token_combine(self, hidden_states, combine_metadata, bias=None):
        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."

        # 1. Preprocess using metadata
        hidden_states = self._combine_preprocess(hidden_states, combine_metadata)

        # 2. AllToAll
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states,
            combine_metadata.input_splits,
            combine_metadata.output_splits,
            self.ep_group,
        )
        handle.wait()
        hidden_states.untyped_storage().resize_(0)

        # 3. Postprocess using metadata
        output = self._combine_postprocess(permutated_local_input_tokens, combine_metadata)

        return output

    def _dispatch_preprocess(self, hidden_states, topk_ids):
        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        (
            tokens_per_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            num_out_tokens,
        ) = self._preprocess(topk_ids)
        hidden_shape_before_permute = hidden_states.shape

        permutated_local_input_tokens, reversed_local_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            tokens=hidden_states,
            indices=topk_ids,
            num_out_tokens=num_out_tokens,
        )

        if self.lora_context is not None:
            preprocess_lora_indices(
                self.lora_context,
                topk_ids=topk_ids,
                reversed_permutation_mapping=reversed_local_input_permutation_mapping,
            )

        return (
            permutated_local_input_tokens,
            reversed_local_input_permutation_mapping,
            tokens_per_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            hidden_shape,
            hidden_shape_before_permute,
        )

    def _preprocess(self, topk_ids: torch.Tensor):
        num_local_tokens_per_expert = torch.histc(topk_ids, bins=self.num_experts, min=0, max=self.num_experts)

        ep_size = self.ep_size
        num_out_tokens = topk_ids.numel()

        input_splits = (
            num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
            .sum(axis=1)
            .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )

        num_global_tokens_per_expert = gather_from_sequence_parallel_region(
            num_local_tokens_per_expert, group=self.ep_group
        ).reshape(ep_size, self.num_experts)
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ]
        if num_global_tokens_per_local_expert is None:
            raise ValueError("num_global_tokens_per_local_expert must be set before sum.")

        output_splits = (
            num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu"), non_blocking=True).numpy()
        )
        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)

        global_input_tokens_local_experts_indices = None
        if self.num_local_experts > 1:
            if num_global_tokens_per_local_expert is None:
                raise ValueError("num_global_tokens_per_local_expert must be set before operations.")
            global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel()
            )
        else:
            torch.npu.synchronize()

        return (
            num_tokens_per_local_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            num_out_tokens,
        )

    def _dispatch_postprocess(
        self,
        global_input_tokens,
        dynamic_scale_after_all2all,
        global_input_tokens_local_experts_indices,
        with_quant,
        dst_type,
        scale_type,
    ):
        # Early return if no local experts or no tokens
        if self.num_local_experts <= 1:
            return global_input_tokens, dynamic_scale_after_all2all, None

        assert global_input_tokens_local_experts_indices is not None, (
            "global_input_tokens_local_experts_indices must be provided"
        )

        if with_quant:
            if scale_type == torch.float8_e8m0fnu:
                experts_indices_2d_copy = global_input_tokens_local_experts_indices.reshape(
                    global_input_tokens_local_experts_indices.shape[0], 1
                )
                dynamic_scale_for_routing = dynamic_scale_after_all2all.view(torch.float8_e8m0fnu)
                global_input_tokens, reversed_global_input_permutation_mapping, _, routed_scale = (
                    torch_npu.npu_moe_init_routing_v2(
                        global_input_tokens,
                        experts_indices_2d_copy,
                        scale=dynamic_scale_for_routing,
                        active_num=experts_indices_2d_copy.shape[0],
                        expert_num=self.num_local_experts,
                        expert_tokens_num_type=1,
                        expert_tokens_num_flag=True,
                        active_expert_range=[0, self.num_local_experts],
                        x_dtype=dst_type,
                    )
                )
                dynamic_scale_after_all2all = routed_scale.view(torch.uint8)
                experts_indices_2d_copy.untyped_storage().resize_(0)
                return global_input_tokens, dynamic_scale_after_all2all, reversed_global_input_permutation_mapping
            dynamic_scale_after_all2all, _ = torch_npu.npu_moe_token_permute(
                dynamic_scale_after_all2all.unsqueeze(-1), global_input_tokens_local_experts_indices
            )
            dynamic_scale_after_all2all = dynamic_scale_after_all2all.squeeze(-1)

        # Non-quantized case
        global_input_tokens, reversed_global_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            global_input_tokens, global_input_tokens_local_experts_indices
        )
        if self.lora_context is not None:
            postprocess_lora_indices(
                self.lora_context,
                reversed_permutation_mapping=reversed_global_input_permutation_mapping,
            )
        return global_input_tokens, dynamic_scale_after_all2all, reversed_global_input_permutation_mapping

    def _combine_preprocess(
        self, hidden_states: torch.Tensor, combine_metadata: MoEAllToAllCombineMetadata
    ) -> torch.Tensor:
        # Unpermutation 2: expert output to AlltoAll input
        rev_global = combine_metadata.reversed_global_input_permutation_mapping
        if hidden_states.shape[0] > 0 and self.num_local_experts > 1 and rev_global is not None:
            hidden_states = torch_npu.npu_moe_token_unpermute(hidden_states, rev_global)
        return hidden_states

    def _combine_postprocess(
        self,
        permutated_local_input_tokens: torch.Tensor,
        combine_metadata: MoEAllToAllCombineMetadata,
    ) -> torch.Tensor:
        # Unpermutation 1: AlltoAll output to output
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=permutated_local_input_tokens,
            sorted_indices=combine_metadata.reversed_local_input_permutation_mapping.to(torch.int32),
            probs=combine_metadata.topk_weights,
            restore_shape=combine_metadata.hidden_shape_before_permute,
        )
        output = output.view(combine_metadata.hidden_shape)
        return output
