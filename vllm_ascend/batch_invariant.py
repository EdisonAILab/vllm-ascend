# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/batch_invariant.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
#
import os

import torch
import torch_npu
import vllm.envs as envs
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON

# in case recursive call in reduce_sum.
torch_sum = torch.sum


if HAS_TRITON:
    from vllm_ascend.ops.triton.batch_invariant.matmul import (
        addmm_batch_invariant,
        bmm_batch_invariant,
        linear_batch_invariant,
        matmul_batch_invariant,
        mm_batch_invariant,
    )
    from vllm_ascend.ops.triton.batch_invariant.softmax import softmax_batch_invariant


try:
    import batch_invariant_ops  # type: ignore[import-not-found] # noqa

    HAS_ASCENDC_BATCH_INVARIANT = True
except ImportError:
    HAS_ASCENDC_BATCH_INVARIANT = False


def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
):
    """AclnnAddRmsNorm can't ensure batch invariant,
    so we need to split it into add and rms_norm.
    """
    x_ = x + residual
    residual_ = x_
    x_, _ = torch_npu.npu_rms_norm(x_, weight, eps)
    return x_, None, residual_


def reduce_sum(
    x: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run single-axis NPU reductions through the last-axis-only BI kernel."""
    native_kwargs = {}
    if dtype is not None:
        native_kwargs["dtype"] = dtype
    if out is not None:
        native_kwargs["out"] = out
    if x.device.type != "npu" or not isinstance(dim, int) or native_kwargs:
        return torch_sum(x, dim, keepdim, **native_kwargs)

    ndim = x.dim()
    normalized_dim = dim + ndim if dim < 0 else dim
    if normalized_dim < 0 or normalized_dim >= ndim:
        return torch_sum(x, dim, keepdim, **native_kwargs)

    if normalized_dim == ndim - 1:
        return torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant(x, -1, keepdim)

    moved = torch.movedim(x, normalized_dim, -1).contiguous()
    outer_shape = moved.shape[:-1]
    rows = moved.flatten(0, -2)
    result = torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant(rows, -1, False)
    result = result.reshape(outer_shape)
    if keepdim:
        result = torch.movedim(result.unsqueeze(-1), -1, normalized_dim)
    return result


def override_envs_for_invariance():
    from vllm_ascend.ascend_config import get_ascend_config

    ascend_config = get_ascend_config()
    ascend_config.weight_nz_mode = 0

    os.environ["HCCL_DETERMINISTIC"] = "strict"
    os.environ["LCCL_DETERMINISTIC"] = "1"

    # Enable deterministic computation for operators. Some operators on Ascend A5
    # do not have deterministic mode enabled by default and must be explicitly set.
    torch.use_deterministic_algorithms(True, warn_only=True)

    logger.debug(
        "Batch-invariant env override: weight_nz_mode=0, HCCL_DETERMINISTIC=strict, "
        "LCCL_DETERMINISTIC=1, use_deterministic_algorithms=True",
    )


_batch_invariant_LIB = None
_training_parity_matmul_LIB = None


def enable_batch_invariant_mode():
    global _batch_invariant_LIB
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")
    logger.debug(
        "Batch-invariant op registration: Triton=%s, AscendC=%s",
        HAS_TRITON,
        HAS_ASCENDC_BATCH_INVARIANT,
    )

    # Register operators only implemented in triton.
    if HAS_TRITON:
        _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::bmm", bmm_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::softmax", softmax_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::_softmax", softmax_batch_invariant, "NPU")

    # Register operators implemented in Ascend batch-invariant ops in priority.
    if HAS_ASCENDC_BATCH_INVARIANT:
        _batch_invariant_LIB.impl("aten::mm", torch.ops.batch_invariant_ops.npu_mm_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::matmul", torch.ops.batch_invariant_ops.npu_matmul_batch_invariant, "NPU")
        # torch_npu.npu_fused_infer_attention_score is a function of torch_npu, not a torch.ops.Operator,
        # so we need to patch it directly.
        torch_npu.npu_fused_infer_attention_score = (
            torch.ops.batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant
        )
        # patch npu_add_rms_norm to ensure batch invariant.
        torch_npu.npu_add_rms_norm = add_rms_norm
        # torch.sum can't be replaced by dispatch logic, so we patch it directly.
        torch.sum = reduce_sum

    # register triton implementations if ascendc is not available.
    elif HAS_TRITON:
        _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::matmul", matmul_batch_invariant, "NPU")

        # linear call matmul internally, so register linear only when ascendc
        # is not available. it will get better performance with ascendc.
        _batch_invariant_LIB.impl("aten::linear", linear_batch_invariant, "NPU")


def init_batch_invariance():
    """
    Initialize batch-invariant mode for vLLM on Ascend NPU.

    This function:
    1. Sets environment variables for deterministic computation
    2. Registers batch-invariant implementations for torch operators
    3. Enables batch-invariant flash attention

    Call this function early in your application, or set VLLM_BATCH_INVARIANT=1
    environment variable to enable automatically.
    """
    if envs.VLLM_BATCH_INVARIANT:
        if HAS_TRITON or HAS_ASCENDC_BATCH_INVARIANT:
            logger.info(
                "Enabling batch-invariant mode for vLLM on Ascend NPU.",
            )
            override_envs_for_invariance()
            enable_batch_invariant_mode()
        else:
            logger.warning(
                "Batch-invariant mode requested but Triton or AscendC batch-invariant "
                "ops is not available.skipping batch-invariant initialization."
            )


def init_training_parity_matmul():
    """Enable only BI matrix multiplication for the opt-in training oracle.

    Training-parity decode deliberately keeps the training engine's standard
    attention implementation.  Registering the complete vLLM BI mode here
    would also replace FIA, norms, sums, and softmax, changing more than the
    row-count-dependent projection operation isolated by logdiff.
    """
    global _training_parity_matmul_LIB

    if os.getenv("VLLM_ASCEND_TRAINING_PARITY", "0") != "1":
        return
    if envs.VLLM_BATCH_INVARIANT:
        return
    if not HAS_ASCENDC_BATCH_INVARIANT:
        raise RuntimeError("VLLM_ASCEND_TRAINING_PARITY requires Ascend batch-invariant operators for mm/matmul")
    if _training_parity_matmul_LIB is not None:
        return

    logger.info("Enabling opt-in training-parity BI mm/matmul only.")
    _training_parity_matmul_LIB = torch.library.Library("aten", "IMPL")
    _training_parity_matmul_LIB.impl(
        "aten::mm",
        torch.ops.batch_invariant_ops.npu_mm_batch_invariant,
        "NPU",
    )
    _training_parity_matmul_LIB.impl(
        "aten::matmul",
        torch.ops.batch_invariant_ops.npu_matmul_batch_invariant,
        "NPU",
    )
