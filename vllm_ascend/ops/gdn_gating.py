# SPDX-License-Identifier: Apache-2.0

import torch


def fused_gdn_gating_native(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Graph-capturable PyTorch decomposition of fused GDN gating."""
    compute_dtype = torch.float32
    A_log_f = A_log.to(compute_dtype)
    a_f = a.to(compute_dtype)
    b_f = b.to(compute_dtype)
    dt_bias_f = dt_bias.to(compute_dtype)

    x = a_f + dt_bias_f.unsqueeze(0)
    beta_x = beta * x
    softplus_x = torch.where(
        beta_x <= threshold,
        (1.0 / beta) * torch.log1p(torch.exp(beta_x)),
        x,
    )
    g = -torch.exp(A_log_f).unsqueeze(0) * softplus_x
    beta_output = torch.sigmoid(b_f).to(b.dtype)
    return g.unsqueeze(0), beta_output.unsqueeze(0)
