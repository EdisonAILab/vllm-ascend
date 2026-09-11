from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.sampler import AscendSampler, AscendTopKTopPSampler


class TestAscendSampler(TestBase):
    def test_init_with_raw_logprobs(self):
        sampler = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertEqual(sampler.logprobs_mode, "raw_logprobs")
        self.assertTrue(hasattr(sampler, "topk_topp_sampler"))
        self.assertIsInstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)

    @patch("vllm_ascend.sample.sampler.Sampler.apply_penalties")
    @patch("vllm_ascend.sample.sampler.is_950", return_value=True)
    @patch("vllm_ascend.sample.sampler.HAS_TRITON", True)
    def test_a5_uses_native_penalties(self, mock_is_950, mock_native):
        logits = torch.randn(2, 8)
        metadata = MagicMock()
        expected = torch.randn_like(logits)
        mock_native.return_value = expected

        actual = AscendSampler.apply_penalties(logits, metadata, [[1], [2]])

        mock_is_950.assert_called_once_with()
        mock_native.assert_called_once_with(logits, metadata, [[1], [2]])
        self.assertIs(actual, expected)
