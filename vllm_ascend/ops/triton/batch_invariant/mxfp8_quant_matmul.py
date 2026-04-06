# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""
MXFP8 Batch-Invariant Operators for Ascend NPU.

Verification results (tested on Ascend A5, CANN 8.5.0):
- npu_dynamic_mx_quant: inherently batch-invariant (per-token operation)
- npu_quant_matmul: batch-invariant (tested M=[1-2048], K=[4096-14336], N=[2048-18432])
- npu_grouped_matmul: batch-invariant across different group_lists
- npu_grouped_matmul_swiglu_quant_v2: batch-invariant — same token through
  same expert produces identical output regardless of other experts' token
  counts, total batch size (M=16-512), or expert's own token count (1-32).

Reference: docs/vllm-ascend-batch-invariant-analysis.md Section 7
"""

from __future__ import annotations

import torch
import torch_npu

from vllm_ascend.device.mxfp_compat import FLOAT8_E8M0FNU_DTYPE


def npu_quant_matmul_batch_invariant(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    scale_dtype: torch.dtype = FLOAT8_E8M0FNU_DTYPE,
    pertoken_scale: torch.Tensor,
    pertoken_scale_dtype: torch.dtype = FLOAT8_E8M0FNU_DTYPE,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    group_sizes: list[int] | None = None,
) -> torch.Tensor:
    """Batch-invariant wrapper for npu_quant_matmul.

    Verified batch-invariant on Ascend 910/A5 with CANN 8.5.0:
    - npu_quant_matmul processes each row's K-dimension reduction identically
      regardless of the total number of rows (M dimension)
    - The microscaling dequantization (scale × FP8 value) is applied per-element
    - Internal kernel parallelization does not change the per-row reduction order

    This wrapper passes through to the native operator directly.
    """
    return torch_npu.npu_quant_matmul(
        x,
        weight,
        weight_scale,
        scale_dtype=scale_dtype,
        pertoken_scale=pertoken_scale,
        pertoken_scale_dtype=pertoken_scale_dtype,
        bias=bias,
        output_dtype=output_dtype,
        group_sizes=group_sizes,
    )


def npu_dynamic_mx_quant_batch_invariant(
    x: torch.Tensor,
    dst_type: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-invariant wrapper for npu_dynamic_mx_quant.

    Verified batch-invariant: npu_dynamic_mx_quant computes per-token
    microscaling quantization where each row is processed independently.
    The per-group absmax and quantization are along the hidden dimension
    (K), not the batch dimension (M).
    """
    return torch_npu.npu_dynamic_mx_quant(x, dst_type=dst_type)


def mxfp8_linear_batch_invariant(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int = 32,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full MXFP8 linear operation with batch-invariance guarantee.

    Pipeline: npu_dynamic_mx_quant → npu_quant_matmul
    Both operations verified batch-invariant.
    """
    original_shape = x.shape
    output_dtype = x.dtype

    if x.dim() > 2:
        x = x.view(-1, x.shape[-1])

    quantized_x, dynamic_scale = npu_dynamic_mx_quant_batch_invariant(
        x, dst_type=torch.float8_e4m3fn
    )

    if bias is not None and bias.dtype != torch.float32:
        bias = bias.to(torch.float32)

    output = npu_quant_matmul_batch_invariant(
        quantized_x,
        weight,
        weight_scale,
        scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        pertoken_scale=dynamic_scale,
        pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        bias=bias,
        output_dtype=output_dtype,
        group_sizes=[1, 1, group_size],
    )

    if len(original_shape) > 2:
        output = output.view(*original_shape[:-1], -1)

    return output
