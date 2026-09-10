import pytest

from vllm_ascend.worker.kimi_graph_dispatch import (
    has_kimi_initial_token_graph_hazard,
)


def hazard(
    computed: list[int],
    *,
    scheduled: list[int] | None = None,
    model_type: str = "kimi_k3",
    force_uniform_decode: bool | None = None,
) -> bool:
    num_reqs = len(computed)
    scheduled = scheduled if scheduled is not None else [1] * num_reqs
    return has_kimi_initial_token_graph_hazard(
        model_type=model_type,
        num_tokens=sum(scheduled),
        num_reqs=num_reqs,
        num_scheduled_tokens=scheduled,
        num_computed_tokens=computed,
        force_uniform_decode=force_uniform_decode,
    )


@pytest.mark.parametrize(
    "computed",
    (
        [0],
        [0, 0],
        [0, 0, 0],
        [8, 0],
        [8, 0, 15],
    ),
)
def test_initial_kimi_request_skips_full_graph(computed):
    assert hazard(computed)


@pytest.mark.parametrize("computed", ([1], [1, 2], [8, 9, 10]))
def test_pure_decode_keeps_graph_dispatch(computed):
    assert not hazard(computed)


@pytest.mark.parametrize("force_uniform_decode", (False, True))
def test_graph_capture_dummy_keeps_requested_dispatch(force_uniform_decode):
    assert not hazard([0], force_uniform_decode=force_uniform_decode)


def test_multi_token_prefill_does_not_match_uniform_decode_shape():
    assert not hazard([0], scheduled=[2])


def test_non_kimi_model_keeps_graph_dispatch():
    assert not hazard([0], model_type="qwen3")


def test_incomplete_scheduler_vectors_fail_closed():
    assert not has_kimi_initial_token_graph_hazard(
        model_type="kimi_k3",
        num_tokens=2,
        num_reqs=2,
        num_scheduled_tokens=[1],
        num_computed_tokens=[0],
        force_uniform_decode=None,
    )
