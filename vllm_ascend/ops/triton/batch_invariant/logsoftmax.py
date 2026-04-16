# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Batch-invariant log_softmax Triton kernel for Ascend NPU.

log_softmax(x) = x - amax(x) - log(sum(exp(x - amax(x))))

Reduction is per-row (along last dim), each row processed by a single
program → no cross-batch reduction → batch-invariant.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _logsoftmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-row log_softmax. Each program handles multiple rows."""
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    rows_per_program = (n_rows + n_programs - 1) // n_programs
    start_row = pid * rows_per_program
    end_row = tl.minimum(start_row + rows_per_program, n_rows)

    for row_idx in range(start_row, end_row):
        in_row_ptr = input_ptr + row_idx * input_row_stride
        out_row_ptr = output_ptr + row_idx * output_row_stride

        # Pass 1: find row max
        row_max = tl.full([1], -float("inf"), dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols
            vals = tl.load(in_row_ptr + col_idx, mask=mask,
                           other=-float("inf")).to(tl.float32)
            block_max = tl.max(tl.where(mask, vals, -float("inf")))
            row_max = tl.maximum(row_max, block_max)

        # Pass 2: compute sum(exp(x - row_max))
        sum_exp = tl.zeros([1], dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols
            vals = tl.load(in_row_ptr + col_idx, mask=mask,
                           other=-float("inf")).to(tl.float32)
            exp_vals = tl.exp(vals - row_max)
            sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

        log_sum_exp = tl.log(sum_exp)

        # Pass 3: write x - row_max - log_sum_exp
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols
            vals = tl.load(in_row_ptr + col_idx, mask=mask,
                           other=0.0).to(tl.float32)
            output = vals - row_max - log_sum_exp
            tl.store(out_row_ptr + col_idx, output.to(in_row_ptr.dtype.element_ty),
                     mask=mask)


def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """log_softmax along the last dimension.

    Args:
        x: Input tensor; reduction along last dim.
        dim: Must be -1 or x.dim()-1 for now.
    """
    if dim != -1 and dim != x.dim() - 1:
        raise NotImplementedError("log_softmax only supports dim=-1 currently.")

    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    n_rows, n_cols = x_2d.shape

    output = torch.empty_like(x_2d)

    BLOCK_SIZE = 1024
    max_grid = triton.runtime.driver.active.utils.get_device_properties(
        torch.npu.current_device()
    )["num_vectorcore"]
    grid = (min(n_rows, max_grid),)

    _logsoftmax_kernel[grid](
        x_2d,
        output,
        x_2d.stride(0),
        output.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.reshape(original_shape)


def logsoftmax_batch_invariant(x: torch.Tensor, dim: int = -1,
                                dtype: torch.dtype | None = None) -> torch.Tensor:
    """Batch-invariant wrapper for log_softmax."""
    out = log_softmax(x, dim=dim)
    if dtype is not None:
        out = out.to(dtype)
    return out
