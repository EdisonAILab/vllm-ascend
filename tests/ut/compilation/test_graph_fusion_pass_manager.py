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

from unittest.mock import MagicMock, patch

import vllm.envs as envs

from vllm_ascend.compilation.graph_fusion_pass_manager import GraphFusionPassManager


def _qknorm_only_config():
    config = MagicMock()
    config.additional_config = {
        "ascend_compilation_config": {
            "fuse_norm_quant": False,
            "fuse_qknorm_rope": True,
            "fuse_allreduce_rms": False,
            "fuse_muls_add": False,
        }
    }
    config.compilation_config.pass_config.enable_sp = False
    return config


def test_qknorm_rope_fusion_enabled_by_default():
    config = _qknorm_only_config()

    with (
        patch.object(envs, "VLLM_BATCH_INVARIANT", False),
        patch("vllm_ascend.compilation.passes.qknorm_rope_fusion_pass.QKNormRopeFusionPass") as pass_cls,
    ):
        manager = GraphFusionPassManager()
        manager.configure(config)

    pass_cls.assert_called_once_with(config)
    assert manager.passes == [pass_cls.return_value]


def test_batch_invariant_mode_disables_qknorm_rope_fusion():
    config = _qknorm_only_config()

    with (
        patch.object(envs, "VLLM_BATCH_INVARIANT", True),
        patch("vllm_ascend.compilation.passes.qknorm_rope_fusion_pass.QKNormRopeFusionPass") as pass_cls,
    ):
        manager = GraphFusionPassManager()
        manager.configure(config)

    pass_cls.assert_not_called()
    assert manager.passes == []
