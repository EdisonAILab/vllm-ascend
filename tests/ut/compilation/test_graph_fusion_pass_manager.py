from unittest.mock import patch

import vllm.envs as envs
from vllm.config import VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config
from vllm_ascend.compilation.graph_fusion_pass_manager import GraphFusionPassManager


class TestGraphFusionPassManagerConfig(TestBase):
    def tearDown(self):
        clear_ascend_config()

    def _qknorm_only_config(self):
        vllm_config = VllmConfig()
        vllm_config.additional_config = {
            "ascend_compilation_config": {
                "fuse_norm_quant": False,
                "fuse_qknorm_rope": True,
                "fuse_muls_add": False,
            }
        }
        init_ascend_config(vllm_config)
        return vllm_config

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_configure_consumes_validated_ascend_compilation_config(
        self, mock_platform
    ):
        vllm_config = VllmConfig()
        vllm_config.additional_config = {
            "ascend_compilation_config": {
                "fuse_norm_quant": "false",
                "fuse_qknorm_rope": "false",
                "fuse_muls_add": "false",
            }
        }
        init_ascend_config(vllm_config)

        manager = GraphFusionPassManager()
        manager.configure(vllm_config)

        self.assertFalse(manager.ascend_compilation_config.fuse_norm_quant)
        self.assertFalse(manager.ascend_compilation_config.fuse_qknorm_rope)
        self.assertFalse(manager.ascend_compilation_config.fuse_muls_add)
        self.assertEqual(manager.passes, [])

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_qknorm_rope_fusion_enabled_by_default(self, mock_platform):
        config = self._qknorm_only_config()

        with (
            patch.object(envs, "VLLM_BATCH_INVARIANT", False),
            patch(
                "vllm_ascend.compilation.passes.qknorm_rope_fusion_pass."
                "QKNormRopeFusionPass"
            ) as pass_cls,
        ):
            manager = GraphFusionPassManager()
            manager.configure(config)

        pass_cls.assert_called_once_with(config)
        self.assertEqual(manager.passes, [pass_cls.return_value])

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_batch_invariant_mode_disables_qknorm_rope_fusion(
        self, mock_platform
    ):
        config = self._qknorm_only_config()

        with (
            patch.object(envs, "VLLM_BATCH_INVARIANT", True),
            patch(
                "vllm_ascend.compilation.passes.qknorm_rope_fusion_pass."
                "QKNormRopeFusionPass"
            ) as pass_cls,
        ):
            manager = GraphFusionPassManager()
            manager.configure(config)

        pass_cls.assert_not_called()
        self.assertEqual(manager.passes, [])
