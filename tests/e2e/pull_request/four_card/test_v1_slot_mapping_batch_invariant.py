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

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

MODEL = os.getenv(
    "BI_REGRESSION_MODEL",
    "vllm-ascend/Qwen3-30B-A3B-W8A8",
)
TARGET_PROMPT = "The capital of France is Paris. The capital of Germany is"
BATCH_PROMPTS = [
    "Write a concise explanation of why the sky appears blue during daylight.",
    "List three practical ways to reduce household energy consumption.",
    TARGET_PROMPT,
    "Describe the water cycle in four short sentences.",
]


@pytest.fixture(autouse=True)
def enable_batch_invariance(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_TP_FIXED_ORDER_ALLREDUCE", "1")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_BI_DECOMPOSE", "1")
    monkeypatch.setenv("VLLM_MXFP8_GROUPED_BI_DECOMPOSE", "1")
    monkeypatch.setenv("VLLM_BI_FIA_DECOMPOSE", "0")
    monkeypatch.setenv("VLLM_BI_CONTIGUOUS_KV", "0")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def _run_cell(prompts: list[str], target_index: int, output: Path) -> dict[str, Any]:
    helper = Path(__file__).with_name("_v1_slot_mapping_batch_invariant_cell.py")
    command = [
        sys.executable,
        str(helper),
        "--model",
        MODEL,
        "--prompts-json",
        json.dumps(prompts),
        "--target-index",
        str(target_index),
        "--max-tokens",
        os.getenv("BI_REGRESSION_MAX_TOKENS", "800"),
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=1000,
    )
    assert result.returncode == 0, (
        f"TP4 regression cell failed with rc={result.returncode}\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
    return json.loads(output.read_text())


def _first_difference(left: list[Any], right: list[Any]) -> int | None:
    mismatch = next(
        (step for step, values in enumerate(zip(left, right)) if values[0] != values[1]),
        None,
    )
    if mismatch is not None:
        return mismatch
    return min(len(left), len(right)) if len(left) != len(right) else None


def test_tp4_slot_mapping_is_batch_invariant(tmp_path: Path):
    singleton = _run_cell([TARGET_PROMPT], 0, tmp_path / "singleton.json")
    batched = _run_cell(BATCH_PROMPTS, 2, tmp_path / "batch4.json")

    singleton_ids = singleton["output_ids"]
    batched_ids = batched["output_ids"]
    singleton_logprobs = singleton["step_logprobs"]
    batched_logprobs = batched["step_logprobs"]
    first_token_diff = _first_difference(singleton_ids, batched_ids)
    first_logprob_diff = _first_difference(singleton_logprobs, batched_logprobs)
    assert first_token_diff is None and first_logprob_diff is None, (
        "TP4 slot mapping is not batch-invariant: "
        f"first token divergence={first_token_diff}, "
        f"first top-5 logprob divergence={first_logprob_diff}"
    )
