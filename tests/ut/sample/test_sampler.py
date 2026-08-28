import os
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample import sampler as sampler_module
from vllm_ascend.sample.sampler import AscendSampler, AscendTopKTopPSampler


class TestAscendSampler(TestBase):
    def test_init_with_raw_logprobs(self):
        sampler = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertEqual(sampler.logprobs_mode, "raw_logprobs")
        self.assertTrue(hasattr(sampler, "topk_topp_sampler"))
        self.assertIsInstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)

    def test_compute_logprobs_uses_native_path_by_default(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.bfloat16)
        expected = logits.log_softmax(dim=-1, dtype=torch.float32)

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_ASCEND_TRAINING_PARITY", None)
            actual = AscendSampler.compute_logprobs(logits)

        self.assertTrue(torch.equal(actual, expected))

    def test_compute_logprobs_uses_cpu_fp32_parity_contract(self):
        torch.manual_seed(7)
        # Exercise non-contiguous input and enough rows to detect accidental
        # batching in the rowwise logdiff contract.
        logits = torch.randn(151, 17, dtype=torch.bfloat16).t()
        cpu_logits = logits.detach().cpu().contiguous().float()
        expected = torch.cat(
            [row - torch.logsumexp(row, dim=-1, keepdim=True) for row in cpu_logits.split(1, dim=0)],
            dim=0,
        )

        original_logsumexp = torch.logsumexp
        reduction_rows = []

        def recording_logsumexp(input_tensor, *args, **kwargs):
            reduction_rows.append(input_tensor.shape[0])
            return original_logsumexp(input_tensor, *args, **kwargs)

        with (
            patch.dict(os.environ, {"VLLM_ASCEND_TRAINING_PARITY": "1"}),
            patch.object(
                sampler_module.torch,
                "logsumexp",
                side_effect=recording_logsumexp,
            ),
        ):
            actual = AscendSampler.compute_logprobs(logits)

        self.assertEqual(actual.device, logits.device)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(reduction_rows, [1] * logits.shape[0])
        self.assertTrue(torch.equal(actual, expected))
