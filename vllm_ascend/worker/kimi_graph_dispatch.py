"""Graph-dispatch guards for Kimi models.

This module intentionally has no torch, vLLM, or vLLM-Ascend imports so the
shape/state predicate can be unit tested without initializing the NPU plugin.
"""

from collections.abc import Sequence


def has_kimi_initial_token_graph_hazard(
    *,
    model_type: str | None,
    num_tokens: int,
    num_reqs: int,
    num_scheduled_tokens: Sequence[int],
    num_computed_tokens: Sequence[int],
    force_uniform_decode: bool | None,
) -> bool:
    """Return whether a FULL graph could read a just-created Kimi KV row.

    A Kimi request with no computed tokens is still in its initial prefill.
    When every scheduled request contributes exactly one token, vLLM's normal
    uniform-decode shape test can otherwise send that mixed semantic state to
    FULL graph replay.  The graph may then read capture-time paged-KV contents
    while writing the request's first real KV row.
    """
    if (
        model_type != "kimi_k3"
        or force_uniform_decode is not None
        or num_reqs <= 0
        or num_tokens != num_reqs
        or len(num_scheduled_tokens) < num_reqs
        or len(num_computed_tokens) < num_reqs
    ):
        return False

    scheduled = num_scheduled_tokens[:num_reqs]
    computed = num_computed_tokens[:num_reqs]
    return all(int(value) == 1 for value in scheduled) and any(int(value) == 0 for value in computed)
