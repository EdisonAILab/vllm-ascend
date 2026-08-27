#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

"""Ascend compatibility fixes for vLLM's layerwise reload path."""

from collections.abc import Callable

import torch

from vllm.model_executor.model_loader.reload import layerwise as layerwise_reload
from vllm.model_executor.model_loader.reload import utils as reload_utils
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo


def _get_layer_size(layer: torch.nn.Module) -> int:
    """Count only tensors that can be populated by a state-dict loader."""
    return sum(
        tensor.numel()
        for name, tensor in layerwise_reload.get_layer_tensors(layer).items()
        if name not in layerwise_reload.SKIP_TENSORS
        and name not in layer._non_persistent_buffers_set
    )


def _wrap_parameters_weight_loader(layer: torch.nn.Module) -> None:
    """Wrap loadable tensors while accepting callable loader objects."""
    for name, tensor in layerwise_reload.get_layer_tensors(layer).items():
        if name in layerwise_reload.SKIP_TENSORS:
            continue
        loader = layerwise_reload._get_weight_loader(tensor)
        if getattr(loader, "__name__", None) != "online_process_loader":
            tensor.weight_loader = layerwise_reload.make_online_process_loader(layer, name)


def _get_original_loader(tensor: torch.Tensor) -> Callable:
    """Return a loader after removing any layerwise wrappers."""
    loader = layerwise_reload._get_weight_loader(tensor)
    while getattr(loader, "__name__", None) == "online_process_loader":
        loader = loader.__wrapped__
    return loader


_restore_layer_on_meta = layerwise_reload.restore_layer_on_meta


def _restore_layer_on_meta_with_buffer_metadata(
    layer: torch.nn.Module,
    info: LayerReloadingInfo,
) -> None:
    _restore_layer_on_meta(layer, info)
    layer._non_persistent_buffers_set.update(info.kernel_non_persistent_buffers)


reload_utils.get_layer_size = _get_layer_size
layerwise_reload.get_layer_size = _get_layer_size
layerwise_reload._wrap_parameters_weight_loader = _wrap_parameters_weight_loader
layerwise_reload._get_original_loader = _get_original_loader
layerwise_reload.restore_layer_on_meta = _restore_layer_on_meta_with_buffer_metadata
