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

"""Fresh-process cell for the TP4 slot-mapping batch-invariance regression."""

import argparse
import json
from pathlib import Path
from typing import Any


def _logprob_value(value: Any) -> float:
    return float(value.logprob if hasattr(value, "logprob") else value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-json", required=True)
    parser.add_argument("--target-index", required=True, type=int)
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    prompts = json.loads(args.prompts_json)
    if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
        raise TypeError("--prompts-json must decode to list[str]")
    if not 0 <= args.target_index < len(prompts):
        raise ValueError("--target-index is outside the prompt list")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=4,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=16,
        gpu_memory_utilization=0.7,
        enable_prefix_caching=False,
        disable_log_stats=True,
        distributed_executor_backend="mp",
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        seed=1234,
        logprobs=5,
    )
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    completion = outputs[args.target_index].outputs[0]
    if completion.logprobs is None:
        raise RuntimeError("vLLM did not return requested logprobs")

    step_logprobs = []
    for selected_id, candidates in zip(completion.token_ids, completion.logprobs, strict=True):
        top = sorted([[int(token_id), _logprob_value(value)] for token_id, value in candidates.items()])
        step_logprobs.append(
            {
                "selected_id": int(selected_id),
                "top": top,
            }
        )

    args.output.write_text(
        json.dumps(
            {
                "output_ids": [int(token_id) for token_id in completion.token_ids],
                "step_logprobs": step_logprobs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
