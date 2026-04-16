# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Batch-invariant SiluAndMul (SwiGLU activation) Triton kernel for Ascend NPU.

SwiGLU: y = SiLU(x[..., :H]) * x[..., H:]    where H = x.shape[-1] // 2

Element-wise operation, naturally batch-invariant since each output element
depends only on two input elements at the same row.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _silu_and_mul_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    H,  # half of input's last dim (= output's last dim)
    BLOCK_SIZE: tl.constexpr,
):
    """Each program handles multiple rows; within each row processes BLOCK_SIZE
    elements at a time. Per-element computation, no reduction across batch."""
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    rows_per_program = (n_rows + n_programs - 1) // n_programs
    start_row = pid * rows_per_program
    end_row = tl.minimum(start_row + rows_per_program, n_rows)

    for row_idx in range(start_row, end_row):
        in_row_ptr = input_ptr + row_idx * input_row_stride
        out_row_ptr = output_ptr + row_idx * output_row_stride

        for col_offset in range(0, H, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < H

            # Gate: SiLU(x[:H]); Up: x[H:]
            gate = tl.load(in_row_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)
            up = tl.load(in_row_ptr + H + col_idx, mask=mask, other=0.0).to(tl.float32)

            # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
            silu_gate = gate / (1.0 + tl.exp(-gate))
            result = silu_gate * up

            tl.store(out_row_ptr + col_idx, result, mask=mask)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: SiLU(x[..., :H]) * x[..., H:].

    Args:
        x: Input tensor of shape (..., 2 * H)
    Returns:
        Tensor of shape (..., H)
    """
    assert x.shape[-1] % 2 == 0, "Last dim must be even (got {})".format(x.shape[-1])

    original_shape = x.shape
    H = x.shape[-1] // 2
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    n_rows = x_2d.shape[0]

    output = torch.empty(n_rows, H, dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 1024
    max_grid = triton.runtime.driver.active.utils.get_device_properties(
        torch.npu.current_device()
    )["num_vectorcore"]
    grid = (min(n_rows, max_grid),)

    _silu_and_mul_kernel[grid](
        x_2d,
        output,
        x_2d.stride(0),
        output.stride(0),
        n_rows,
        H,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.reshape(*original_shape[:-1], H)


def silu_and_mul_batch_invariant(x: torch.Tensor) -> torch.Tensor:
    """Batch-invariant wrapper for SwiGLU."""
    return silu_and_mul(x)
