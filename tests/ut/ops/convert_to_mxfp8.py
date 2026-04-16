"""Convert a BF16 HuggingFace model to Ascend W8A8_MXFP8 format.

This script:
1. Loads a BF16 model from HuggingFace format
2. Quantizes all linear layer weights to FP8 E4M3 with microscaling scales
3. Saves the quantized weights + scales in safetensors format
4. Creates the hf_quant_config.json that vllm-ascend expects

Usage:
    python convert_to_mxfp8.py --input /path/to/bf16/model --output /path/to/mxfp8/model

Requirements: torch, torch_npu (with NPU device), safetensors, transformers
"""
import argparse
import json
import os
import shutil

import torch
import torch_npu
from safetensors.torch import save_file, load_file


GROUP_SIZE = 32


def quantize_weight(weight_bf16, group_size=GROUP_SIZE):
    """Quantize a BF16 weight tensor to MXFP8 format.

    Args:
        weight_bf16: [out_features, in_features] BF16 tensor

    Returns:
        weight_fp8: [out_features, in_features] float8_e4m3fn
        weight_scale: [out_features, in_features // group_size // 2, 2] uint8 (E8M0)
    """
    weight_npu = weight_bf16.to("npu")
    weight_fp8, weight_scale = torch_npu.npu_dynamic_mx_quant(
        weight_npu, dst_type=torch.float8_e4m3fn
    )
    # Return as CPU tensors for saving
    return weight_fp8.cpu(), weight_scale.cpu()


def convert_model(input_path, output_path):
    print("=" * 60)
    print("Convert BF16 model to Ascend W8A8_MXFP8")
    print("Input:  {}".format(input_path))
    print("Output: {}".format(output_path))
    print("=" * 60)

    os.makedirs(output_path, exist_ok=True)

    # Copy non-weight files
    for fname in os.listdir(input_path):
        if fname.endswith(".safetensors"):
            continue
        src = os.path.join(input_path, fname)
        dst = os.path.join(output_path, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print("Copied: {}".format(fname))

    # Load all weight files
    weight_files = [f for f in os.listdir(input_path) if f.endswith(".safetensors")]
    print("\nProcessing {} weight file(s)...".format(len(weight_files)))

    quant_description = {}
    all_tensors = {}

    for wf in sorted(weight_files):
        print("\n--- {} ---".format(wf))
        tensors = load_file(os.path.join(input_path, wf))

        for name, tensor in tensors.items():
            # Quantize linear layer weights (2D tensors with .weight suffix)
            if name.endswith(".weight") and tensor.dim() == 2:
                out_feat, in_feat = tensor.shape
                # Only quantize if in_features is divisible by group_size
                if in_feat % GROUP_SIZE == 0 and in_feat >= GROUP_SIZE:
                    print("  Quantizing: {} [{}, {}]".format(name, out_feat, in_feat))
                    fp8_weight, fp8_scale = quantize_weight(tensor.to(torch.bfloat16))

                    # Store quantized weight (as uint8 view for safetensors compatibility)
                    all_tensors[name] = fp8_weight.view(torch.uint8)
                    scale_name = name.replace(".weight", ".weight_scale")
                    all_tensors[scale_name] = fp8_scale

                    # Record in quant_description
                    layer_prefix = name.rsplit(".weight", 1)[0]
                    quant_description[layer_prefix] = "W8A8_MXFP8"
                else:
                    print("  Skip (not quantizable): {} [{}, {}]".format(name, out_feat, in_feat))
                    all_tensors[name] = tensor
            else:
                all_tensors[name] = tensor

    # Save quantized weights
    output_file = os.path.join(output_path, "model.safetensors")
    print("\nSaving quantized weights to {}...".format(output_file))
    save_file(all_tensors, output_file)

    # Create hf_quant_config.json
    hf_quant_config = {
        "quant_method": "",
        "quant_type": "W8A8_MXFP8",
        "group_size": GROUP_SIZE,
        "quant_description": quant_description,
    }
    config_path = os.path.join(output_path, "hf_quant_config.json")
    with open(config_path, "w") as f:
        json.dump(hf_quant_config, f, indent=2)
    print("Saved: hf_quant_config.json")

    # Update config.json to remove any existing quantization_config
    config_json_path = os.path.join(output_path, "config.json")
    if os.path.exists(config_json_path):
        with open(config_json_path) as f:
            config = json.load(f)
        config.pop("quantization_config", None)
        with open(config_json_path, "w") as f:
            json.dump(config, f, indent=2)
        print("Updated: config.json (removed quantization_config)")

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("Quantized layers: {}".format(len(quant_description)))
    print("Output: {}".format(output_path))
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to BF16 model")
    parser.add_argument("--output", required=True, help="Path to save MXFP8 model")
    args = parser.parse_args()
    convert_model(args.input, args.output)
