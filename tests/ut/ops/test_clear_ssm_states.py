# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.triton.fla.utils import clear_ssm_states


@pytest.mark.parametrize("shape", [(1, 2), (4, 3, 5, 7), (6, 5, 25, 41)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_clear_ssm_states_uses_native_row_mask_on_a5(shape, dtype):
    torch.manual_seed(sum(shape))
    states = torch.randn(shape, dtype=dtype)
    has_initial_state = torch.tensor(
        [(index % 3) == 0 for index in range(shape[0])],
        dtype=torch.bool,
    )
    expected = states.clone()
    expected[~has_initial_state] = 0

    with (
        patch("vllm_ascend.ops.triton.fla.utils.is_950", return_value=True),
        patch("vllm_ascend.ops.triton.fla.utils._clear_ssm_states_kernel") as triton_kernel,
    ):
        clear_ssm_states(states, has_initial_state)

    assert torch.equal(states, expected)
    triton_kernel.assert_not_called()


def test_clear_ssm_states_a5_clears_non_finite_values():
    states = torch.tensor([[float("nan"), float("inf")], [1.0, 2.0]])
    has_initial_state = torch.tensor([False, True])

    with patch("vllm_ascend.ops.triton.fla.utils.is_950", return_value=True):
        clear_ssm_states(states, has_initial_state)

    assert torch.equal(states, torch.tensor([[0.0, 0.0], [1.0, 2.0]]))
