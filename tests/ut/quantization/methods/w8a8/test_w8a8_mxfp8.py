from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn as nn
from vllm.model_executor.layers.linear import RowParallelLinear

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import (
    create_mock_ascend_config,
    create_mock_vllm_config,
    create_mxfp_moe_layer,
)
from vllm_ascend.quantization.methods.w8a8 import w8a8_mxfp8
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
    AscendW8A8MXFP8DynamicFusedMoEMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)


class TestAscendW8A8MXFP8LinearMethod(TestBase):
    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config")
    def setUp(self, mock_vllm):
        mock_vllm.return_value = create_mock_vllm_config()
        self.scheme = AscendW8A8MXFP8DynamicLinearMethod()

    def test_modelopt_config_defaults_group_size(self):
        vllm_config = create_mock_vllm_config()
        vllm_config.quant_config = SimpleNamespace()
        with (
            patch(
                "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config",
                return_value=vllm_config,
            ),
            patch(
                "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_dynamic_mx_quant_scale_alg",
                return_value=0,
            ),
        ):
            scheme = AscendW8A8MXFP8DynamicLinearMethod()

        self.assertEqual(scheme.group_size, 32)

    def test_get_weight_various_input_sizes(self):
        sizes = [(128, 64), (512, 256), (1024, 512)]
        for input_size, output_size in sizes:
            result = self.scheme.get_weight(input_size, output_size, torch.bfloat16)
            self.assertEqual(result["weight"].shape, (output_size, input_size))
            self.assertEqual(result["weight"].dtype, torch.float8_e4m3fn)

    def test_get_pergroup_param_group_size_variations(self):
        group_sizes = [16, 32, 64, 128]
        for gs in group_sizes:
            self.scheme.group_size = gs
            result = self.scheme.get_pergroup_param(256, 128, torch.bfloat16)
            self.assertEqual(result["weight_scale"].shape, (128, 256 // gs))
            self.assertEqual(result["weight_scale"].dtype, torch.uint8)

    def test_process_weights_stores_original_shapes(self):
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randn(128, 256).to(torch.float8_e4m3fn), requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.randint(0, 255, (128, 8), dtype=torch.uint8), requires_grad=False)
        self.scheme.process_weights_after_loading(layer)
        self.assertTrue(hasattr(layer, "_mxfp8_original_shapes"))
        self.assertEqual(layer._mxfp8_original_shapes["weight"], (128, 256))
        self.assertTrue(layer._mxfp8_transformed)
        self.assertEqual(layer.weight_scale.shape, (4, 128, 2))
        self.assertTrue(layer.weight.data.is_contiguous())
        self.assertTrue(layer.weight_scale.data.is_contiguous())

    def test_restore_after_process_returns_original_shape(self):
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randn(128, 256).to(torch.float8_e4m3fn), requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.randint(0, 255, (128, 8), dtype=torch.uint8), requires_grad=False)
        original_weight_shape = layer.weight.shape
        original_scale_shape = layer.weight_scale.shape
        self.scheme.process_weights_after_loading(layer)
        self.scheme.restore_weights_for_rl_loading(layer)
        self.assertEqual(layer.weight.shape, original_weight_shape)
        self.assertEqual(layer.weight_scale.shape, original_scale_shape)
        self.assertFalse(layer._mxfp8_transformed)

    def test_transform_buffer_data_ptr_stable_across_reloads(self):
        # The transformed buffer is what the ACL graph captures and replays.
        # It must keep a stable data_ptr across RL weight reloads so graph
        # replay never reads stale/freed memory and produces garbled output.
        # The transformed weight/scale must also stay contiguous, since ACL
        # graph capture expects contiguous weight/scale tensors.
        layer = nn.Module()
        layer.weight = nn.Parameter(
            torch.randint(0, 255, (128, 256), dtype=torch.uint8).to(torch.float8_e4m3fn), requires_grad=False
        )
        layer.weight_scale = nn.Parameter(torch.randint(0, 255, (128, 8), dtype=torch.uint8), requires_grad=False)
        self.scheme.process_weights_after_loading(layer)
        transform_weight_ptr = layer._mxfp8_weight_buf.data_ptr()
        transform_scale_ptr = layer._mxfp8_scale_buf.data_ptr()
        # Buffers and the weight/scale views over them must be contiguous.
        self.assertTrue(layer._mxfp8_weight_buf.is_contiguous())
        self.assertTrue(layer._mxfp8_scale_buf.is_contiguous())
        self.assertTrue(layer.weight.data.is_contiguous())
        self.assertTrue(layer.weight_scale.data.is_contiguous())
        for _ in range(3):
            self.scheme.restore_weights_for_rl_loading(layer)
            # Simulate model.load_weights() writing new data via copy_.
            new_w = torch.randint(0, 255, layer.weight.shape, dtype=torch.uint8).to(torch.float8_e4m3fn)
            new_s = torch.randint(0, 255, layer.weight_scale.shape, dtype=torch.uint8)
            layer.weight.data.copy_(new_w)
            layer.weight_scale.data.copy_(new_s)
            self.scheme.process_weights_after_loading(layer)
            self.assertEqual(layer._mxfp8_weight_buf.data_ptr(), transform_weight_ptr)
            self.assertEqual(layer._mxfp8_scale_buf.data_ptr(), transform_scale_ptr)
            self.assertEqual(layer.weight.data.data_ptr(), transform_weight_ptr)
            self.assertEqual(layer.weight_scale.data.data_ptr(), transform_scale_ptr)
            # Contiguity must be preserved across reloads.
            self.assertTrue(layer._mxfp8_weight_buf.is_contiguous())
            self.assertTrue(layer._mxfp8_scale_buf.is_contiguous())
            self.assertTrue(layer.weight.data.is_contiguous())
            self.assertTrue(layer.weight_scale.data.is_contiguous())

    def test_process_weights_preserves_unaligned_tp_group_phase(self):
        layer = RowParallelLinear.__new__(RowParallelLinear)
        nn.Module.__init__(layer)
        original_weight = torch.randn(2, 528).to(torch.float8_e4m3fn)
        layer.weight = nn.Parameter(original_weight.clone(), requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.randint(0, 255, (2, 17), dtype=torch.uint8), requires_grad=False)
        layer.tp_rank = 3
        layer.input_size_per_partition = 528

        self.scheme.process_weights_after_loading(layer)

        self.assertEqual(layer.mxfp8_tp_padding, (16, 0))
        self.assertEqual(layer.weight.shape, (544, 2))
        torch.testing.assert_close(layer.weight[:16], torch.zeros(16, 2, dtype=torch.float8_e4m3fn))
        torch.testing.assert_close(layer.weight[16:], original_weight.transpose(0, 1))

    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.torch_npu")
    def test_apply(self, mock_torch_npu):
        dynamic_scale = torch.randint(0, 255, (32, 8), dtype=torch.uint8)
        mock_torch_npu.npu_dynamic_mx_quant.return_value = (
            torch.randint(0, 255, (32, 256), dtype=torch.uint8),
            dynamic_scale,
        )
        mock_torch_npu.npu_quant_matmul.return_value = torch.randn(32, 128, dtype=torch.float16)
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randn(256, 128).to(torch.float8_e4m3fn), requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.randint(0, 255, (4, 128, 2), dtype=torch.uint8), requires_grad=False)
        x = torch.randn(32, 1, 256, dtype=torch.float16)
        bias = torch.randn(128, dtype=torch.float16)
        output = self.scheme.apply(layer, x, bias)
        self.assertEqual(output.shape, (32, 1, 128))
        dynamic_quant_kwargs = mock_torch_npu.npu_dynamic_mx_quant.call_args.kwargs
        self.assertEqual(dynamic_quant_kwargs["scale_alg"], self.scheme.dynamic_mx_quant_scale_alg)
        call_kwargs = mock_torch_npu.npu_quant_matmul.call_args.kwargs
        self.assertEqual(call_kwargs["bias"].dtype, torch.float32)
        self.assertEqual(call_kwargs["group_sizes"], [1, 1, self.scheme.group_size])
        self.assertEqual(call_kwargs["output_dtype"], torch.float16)

    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.torch_npu")
    def test_apply_pads_activation_with_weight_group_phase(self, mock_torch_npu):
        mock_torch_npu.npu_dynamic_mx_quant.return_value = (
            torch.empty(2, 544, dtype=torch.float8_e4m3fn),
            torch.empty(2, 17),
        )
        mock_torch_npu.npu_quant_matmul.return_value = torch.empty(2, 4)
        layer = nn.Module()
        layer.mxfp8_tp_padding = (16, 0)
        layer.weight = nn.Parameter(torch.empty(544, 4, dtype=torch.float8_e4m3fn), requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.empty(9, 4, 2, dtype=torch.uint8), requires_grad=False)
        x = torch.randn(2, 528)

        self.scheme.apply(layer, x)

        padded_x = mock_torch_npu.npu_dynamic_mx_quant.call_args.args[0]
        self.assertEqual(padded_x.shape, (2, 544))
        torch.testing.assert_close(padded_x[:, :16], torch.zeros(2, 16))
        torch.testing.assert_close(padded_x[:, 16:], x)

    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.torch_npu")
    def test_apply_uses_batch_invariant_decomposition_when_enabled(self, mock_torch_npu):
        dynamic_scale = torch.randint(0, 255, (4, 2), dtype=torch.uint8)
        quantized = torch.randint(0, 255, (4, 64), dtype=torch.uint8)
        expected = torch.randn(4, 32, dtype=torch.bfloat16)
        mock_torch_npu.npu_dynamic_mx_quant.return_value = (
            quantized,
            dynamic_scale,
        )
        layer = nn.Module()
        layer.weight = nn.Parameter(
            torch.randn(64, 32).to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.weight_scale = nn.Parameter(
            torch.randint(0, 255, (2, 32, 2), dtype=torch.uint8),
            requires_grad=False,
        )

        with (
            patch.object(w8a8_mxfp8, "_DENSE_BI_DECOMPOSE", True),
            patch.object(
                w8a8_mxfp8,
                "_dense_bi_matmul",
                return_value=expected,
            ) as dense_bi_matmul,
        ):
            output = self.scheme.apply(
                layer,
                torch.randn(4, 64, dtype=torch.bfloat16),
            )

        assert output is expected
        dense_bi_matmul.assert_called_once()
        mock_torch_npu.npu_quant_matmul.assert_not_called()


class TestAscendW8A8MXFP8MoEMethod(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 256

    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_ascend_config")
    def setUp(self, mock_ascend, mock_vllm):
        mock_vllm.return_value = create_mock_vllm_config()
        mock_ascend.return_value = create_mock_ascend_config()
        self.scheme = AscendW8A8MXFP8DynamicFusedMoEMethod()

    def test_modelopt_config_defaults_group_size(self):
        vllm_config = create_mock_vllm_config()
        vllm_config.quant_config = SimpleNamespace()
        vllm_config.use_v2_model_runner = True
        with (
            patch(
                "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config",
                return_value=vllm_config,
            ),
            patch(
                "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_ascend_config",
                return_value=create_mock_ascend_config(),
            ),
        ):
            scheme = AscendW8A8MXFP8DynamicFusedMoEMethod()

        self.assertEqual(scheme.group_size, 32)

    def test_get_weight_various_expert_counts(self):
        for num_experts in [4, 8, 16]:
            result = self.scheme.get_weight(num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16)
            self.assertEqual(result["w13_weight"].shape[0], num_experts)
            self.assertEqual(result["w2_weight"].dtype, torch.float8_e4m3fn)

    def test_get_dynamic_quant_param_dtype_uint8(self):
        result = self.scheme.get_dynamic_quant_param(
            self.num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16
        )
        self.assertEqual(result["w13_weight_scale"].shape, (8, 512, 4))
        self.assertEqual(result["w2_weight_scale"].dtype, torch.uint8)

    def test_process_weights_stores_original_shapes(self):
        layer = create_mxfp_moe_layer(
            num_experts=self.num_experts, hidden_size=self.hidden_size, intermediate_size=self.intermediate_size
        )
        original_shape = layer.w13_weight.shape
        self.scheme.process_weights_after_loading(layer)
        self.assertTrue(hasattr(layer, "_mxfp8_original_shapes"))
        self.assertIn("w13_weight", layer._mxfp8_original_shapes)
        self.assertEqual(layer.w13_weight.shape, (original_shape[0], original_shape[2], original_shape[1]))
        self.assertFalse(layer.w13_weight.data.is_contiguous())
        self.assertFalse(layer.w2_weight.data.is_contiguous())
        self.assertFalse(layer.w13_weight_scale.data.is_contiguous())
        self.assertFalse(layer.w2_weight_scale.data.is_contiguous())

        weight_views = self.scheme.get_eplb_weight_views(layer)
        self.assertTrue(self.scheme.supports_eplb)
        self.assertEqual(len(weight_views), 4)
        for source, weight_view in zip(
            [layer.w13_weight, layer.w2_weight, layer.w13_weight_scale, layer.w2_weight_scale],
            weight_views,
        ):
            self.assertTrue(weight_view.is_contiguous())
            self.assertEqual(weight_view.shape[0], self.num_experts)
            self.assertEqual(weight_view.untyped_storage().data_ptr(), source.untyped_storage().data_ptr())

    def test_restore_weights_for_rl_loading(self):
        layer = create_mxfp_moe_layer(
            num_experts=self.num_experts, hidden_size=self.hidden_size, intermediate_size=self.intermediate_size
        )
        original_w13_shape = layer.w13_weight.shape
        self.scheme.process_weights_after_loading(layer)
        self.assertNotEqual(layer.w13_weight.shape, original_w13_shape)
        self.scheme.restore_weights_for_rl_loading(layer)
        self.assertEqual(layer.w13_weight.shape, original_w13_shape)

    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8._EXTRA_CTX")
    def test_apply_full_params(self, mock_ctx):
        tokens = 4
        layer = create_mxfp_moe_layer(
            num_experts=self.num_experts, hidden_size=self.hidden_size, intermediate_size=self.intermediate_size
        )
        self.scheme.process_weights_after_loading(layer)
        layer.swiglu_limit = 1000000
        x = torch.randn(tokens, self.hidden_size, dtype=torch.bfloat16)
        topk_weights = torch.randn(tokens, 2)
        topk_ids = torch.randint(0, self.num_experts, (tokens, 2))
        layer.activation = "silu"
        layer.ascend_pertoken_scale = torch.randn(tokens)
        layer.apply_router_weight_on_input = True
        layer.ascend_expert_map = None
        layer.global_redundant_expert_num = 0
        layer.log2phy = None
        layer.ascend_mc2_mask = None
        layer.swiglu_alpha = 1.0
        layer.swiglu_beta = 0.0
        mock_comm = Mock()
        mock_comm.fused_experts.return_value = torch.randn(tokens, self.hidden_size)
        mock_ctx.moe_comm_method = mock_comm
        mock_ctx.moe_comm_type = Mock()
        self.scheme.apply(
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )
        mock_comm.fused_experts.assert_called_once()


class TestMXFP8BatchInvariantHelpers(TestBase):
    def test_e8m0_scale_decode(self):
        scale = torch.tensor([126, 127, 128], dtype=torch.uint8)
        torch.testing.assert_close(
            w8a8_mxfp8._e8m0_to_f32(scale),
            torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32),
        )

    def test_group_offsets_accept_counts_and_cumulative_offsets(self):
        expected = [0, 2, 5, 6]
        self.assertEqual(
            w8a8_mxfp8._group_offsets(torch.tensor([2, 3, 1]), 0, 6),
            expected,
        )
        self.assertEqual(
            w8a8_mxfp8._group_offsets(torch.tensor([2, 5, 6]), 0, 6),
            expected,
        )

    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.torch_npu")
    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8._NATIVE_NPU_GROUPED_MATMUL")
    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8._grouped_weights_bf16_native")
    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8._dequant_activation")
    def test_graph_gmm1_keeps_group_list_on_device(self, mock_dequant, mock_weights, mock_gmm, mock_torch_npu):
        x = torch.zeros(4, 8, dtype=torch.uint8)
        group_list = torch.tensor([0, 2, 4], dtype=torch.int64)
        values_bf16 = torch.randn(4, 8, dtype=torch.bfloat16)
        weights_bf16 = torch.randn(3, 8, 16, dtype=torch.bfloat16)
        hidden = torch.randn(4, 16, dtype=torch.bfloat16)
        mock_dequant.return_value = values_bf16
        mock_weights.return_value = weights_bf16
        mock_gmm.return_value = [hidden]
        mock_torch_npu.npu_swiglu.return_value = hidden[:, :8]
        expected = (torch.empty(4, 8), torch.empty(4, 1))
        mock_torch_npu.npu_dynamic_mx_quant.return_value = expected

        result = w8a8_mxfp8._grouped_gmm1_graph_bi(
            x=x,
            weights=torch.empty(3, 8, 16),
            scales=torch.empty(3, 1, 16),
            x_scale=torch.empty(4, 1),
            group_list=group_list,
        )

        self.assertIs(result, expected)
        self.assertIs(mock_gmm.call_args.kwargs["group_list"], group_list)
        self.assertEqual(mock_gmm.call_args.kwargs["group_list_type"], 0)

    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8._NATIVE_NPU_GROUPED_MATMUL")
    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8._grouped_weights_bf16_native")
    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8._dequant_activation")
    def test_graph_gmm2_preserves_count_group_list(self, mock_dequant, mock_weights, mock_gmm):
        group_list = torch.tensor([1, 0, 3], dtype=torch.int64)
        expected = [torch.randn(4, 8, dtype=torch.bfloat16)]
        mock_dequant.return_value = torch.randn(4, 16, dtype=torch.bfloat16)
        mock_weights.return_value = torch.randn(3, 16, 8, dtype=torch.bfloat16)
        mock_gmm.return_value = expected

        result = w8a8_mxfp8._grouped_gmm2_graph_bi(
            values=torch.empty(4, 16),
            weights=torch.empty(3, 16, 8),
            scales=torch.empty(3, 1, 8),
            token_scale=torch.empty(4, 1),
            bias=None,
            group_list=group_list,
            group_list_type=1,
            output_dtype=torch.bfloat16,
        )

        self.assertIs(result, expected)
        self.assertIs(mock_gmm.call_args.kwargs["group_list"], group_list)
        self.assertEqual(mock_gmm.call_args.kwargs["group_list_type"], 1)

    def test_grouped_scale_transform_and_restore_are_lossless(self):
        method = object.__new__(AscendW8A8MXFP8DynamicFusedMoEMethod)
        for scale_groups in (3, 4):
            layer = SimpleNamespace(
                w13_weight=SimpleNamespace(data=torch.arange(24).reshape(2, 4, 3)),
                w2_weight=SimpleNamespace(data=torch.arange(24).reshape(2, 4, 3)),
                w13_weight_scale=SimpleNamespace(data=torch.arange(8 * scale_groups).reshape(2, 4, scale_groups)),
                w2_weight_scale=SimpleNamespace(data=torch.arange(8 * scale_groups).reshape(2, 4, scale_groups)),
            )
            original = {
                name: getattr(layer, name).data.clone()
                for name in (
                    "w13_weight",
                    "w2_weight",
                    "w13_weight_scale",
                    "w2_weight_scale",
                )
            }

            with patch.object(w8a8_mxfp8, "_GROUPED_BI_DECOMPOSE", True):
                method.process_weights_after_loading(layer)
                method.restore_weights_for_rl_loading(layer)

            for name, value in original.items():
                self.assertTrue(torch.equal(getattr(layer, name).data, value))
