# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.ops.gdn_gating import fused_gdn_gating_native


@pytest.mark.parametrize("rows,heads", [(1, 1), (37, 8), (129, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fused_gdn_gating_native_contract(rows, heads, dtype):
    torch.manual_seed(rows + heads)
    A_log = torch.randn(heads, dtype=dtype)
    a = torch.randn(rows, heads, dtype=dtype)
    b = torch.randn(rows, heads, dtype=dtype)
    dt_bias = torch.randn(heads, dtype=dtype)

    g, beta = fused_gdn_gating_native(A_log, a, b, dt_bias)

    expected_g = -torch.exp(A_log.float()).unsqueeze(0) * torch.nn.functional.softplus(
        a.float() + dt_bias.float().unsqueeze(0)
    )
    expected_beta = torch.sigmoid(b.float()).to(dtype)
    assert g.shape == (1, rows, heads)
    assert g.dtype == torch.float32
    assert beta.shape == (1, rows, heads)
    assert beta.dtype == dtype
    assert torch.isfinite(g).all()
    assert torch.isfinite(beta).all()
    torch.testing.assert_close(g.squeeze(0), expected_g, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(beta.squeeze(0), expected_beta, rtol=0, atol=0)


@pytest.mark.parametrize("target", [0, 3, 16])
def test_fused_gdn_gating_native_is_exactly_row_invariant(target):
    torch.manual_seed(1234)
    A_log = torch.randn(8, dtype=torch.bfloat16)
    a = torch.randn(17, 8, dtype=torch.bfloat16)
    b = torch.randn(17, 8, dtype=torch.bfloat16)
    dt_bias = torch.randn(8, dtype=torch.bfloat16)

    batch_g, batch_beta = fused_gdn_gating_native(A_log, a, b, dt_bias)
    single_g, single_beta = fused_gdn_gating_native(
        A_log,
        a[target : target + 1],
        b[target : target + 1],
        dt_bias,
    )

    assert torch.equal(batch_g[:, target : target + 1], single_g)
    assert torch.equal(batch_beta[:, target : target + 1], single_beta)
