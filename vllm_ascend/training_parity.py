# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Small shared state for the opt-in training-parity correctness path."""

from contextvars import ContextVar

_sequence_length: ContextVar[int | None] = ContextVar(
    "training_parity_sequence_length",
    default=None,
)


def set_training_parity_sequence_length(sequence_length: int) -> None:
    _sequence_length.set(sequence_length)


def get_training_parity_sequence_length() -> int | None:
    return _sequence_length.get()
