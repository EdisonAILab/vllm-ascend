# Kimi K3 W4A8 training-inference parity

This branch contains a correctness profile for localizing numerical
differences between Megatron Core `core_r0.18.0` and vLLM `0.26.0` with
vLLM-Ascend on Ascend 950DT. It targets Kimi K3 checkpoints with MXFP4 E2M1
routed-expert weights, E8M0 weight scales, and dynamic MXFP8 E4M3FN
activations.

Most of the profile is opt-in and is a correctness reference for eager TP1
execution, not a serving-performance mode. Two W4A8 SiTU MoE contracts are
automatic: routed BF16 rows are quantized with `npu_dynamic_mx_quant`, and
routing weights are consumed by SiTU before GMM2. Other activations and
quantization types retain their existing dispatch and combine paths. The
validated reduced model has 8 layers, hidden size 1,024, 6 KDA layers, 2 MLA
layers, and top-2 routing over 8 experts.

## Validated profile

Use the following settings with `enforce_eager=True`:

```bash
export VLLM_ASCEND_KIMI_REFERENCE_SHORT_CONV=0
export VLLM_ASCEND_KIMI_UNFUSED_SHORT_CONV_ACTIVATION=1
export VLLM_ASCEND_KIMI_NATIVE_STATE_OPS=1
export VLLM_ASCEND_KIMI_KDA_NATIVE_NORM_GATE=1
export VLLM_ASCEND_KIMI_SITU_MIN_ROWS=2
export VLLM_ASCEND_KIMI_GATE_LOWER_BOUND=-5.0
export VLLM_ASCEND_KIMI_REFERENCE_ATTN_RES=1
export VLLM_ASCEND_KIMI_VECTORIZED_ATTN_RES=1
export VLLM_ASCEND_KIMI_NATIVE_ATTN_RES=1
export VLLM_ASCEND_KIMI_REFERENCE_ROUTER_FP32=1
export VLLM_ASCEND_KIMI_REFERENCE_ROUTING=0
export VLLM_ASCEND_KIMI_REFERENCE_ROUTED_RMS_NORM=0
export VLLM_ASCEND_KIMI_DECOMPOSED_ROUTED_RMS_NORM=1
export VLLM_ASCEND_KIMI_REFERENCE_MLA_RMS_NORM=1
export VLLM_ASCEND_KIMI_DECOMPOSED_MLA_RMS_NORM=0
export VLLM_ASCEND_KIMI_REFERENCE_MLA_DECODE=1
export VLLM_ASCEND_KIMI_CONCAT_SHORT_MLA_ROPE=1
export VLLM_ASCEND_NATIVE_SLOT_MAPPING=1
export VLLM_ASCEND_SKIP_UNUSED_PENALTY_WARMUP=1
```

The accepted profile keeps production MoE routing, production KDA recurrence,
and the production causal convolution/cache update, but runs SiLU separately.
Dense SiTU pads a one-row decode calculation to two rows and slices the result
back when `VLLM_ASCEND_KIMI_SITU_MIN_ROWS=2`. This pins the NPU elementwise
kernel geometry: without it, an independent forced-prefix case first differed
at row 17 because one BF16 SiTU element changed between one- and two-request
decode batches. Inputs and the preceding W4A8 gate/up GEMM were byte-identical.
It keeps explicit FP32 RMSNorm reductions, router arithmetic, MLA decode, and
AttnRes reductions where the optimized kernels did not satisfy the byte-exact
contract. For W4A8 SiTU, the normal dispatch boundary quantizes the sorted
BF16 tokens with `npu_dynamic_mx_quant`, and routing weights are applied before
the second dynamic MXFP8 quantization and grouped GEMM. The fused routing
quantizer's E4M3 modes 3 and 17 use a different E8M0 scale-rounding policy and
were rejected by the serialized-logprob gate.

No KDA core selector is required. The production AscendC KDA operator passed
the 1,024-row byte-exact gate with both diagnostic KDA environment variables
unset. `VLLM_ASCEND_KIMI_NATIVE_KDA_CORE=1` selects the explicit Python
recurrence used during localization; it is not the production path.

For deliberately reduced Kimi checkpoints, the fixed-shape MLAPO prolog is
disabled automatically when `kv_lora_rank` is not 512. The standard MLA
preprocess path pads the reduced RoPE slice to the 64-wide CANN contract and
uses the supported `BNSD` layout. This makes the production fused-attention
kernel runnable for diagnosis, but that kernel did not pass the byte-exact
gate; keep `VLLM_ASCEND_KIMI_REFERENCE_MLA_DECODE=1` in correctness runs.

The decomposed routed-expert RMSNorm passed the complete 1,024-row gate. It
uses the native NPU reduction and applies the learned weight only after the
normalized value reaches BF16, matching the Megatron boundary. The same
decomposition is not interchangeable at the narrower MLA Q/KV ranks: it
passed 8 rows but differed at decode call 883. Keep
`VLLM_ASCEND_KIMI_DECOMPOSED_MLA_RMS_NORM=0` and the explicit MLA reference
reduction enabled for byte-exact runs. The fully fused native RMSNorm path was
rejected at the 8-row gate for both sites.

## Result

With prompt token IDs 1 through 32, one prefill plus 1,023 real cached-decode
calls matched the Megatron reference for all 1,024 complete FP32 native
logprob rows. The comparison covered 167,772,160 values and 671,088,640 bytes
with zero differing bytes. Native logprobs were captured immediately after
each engine's own `logits.log_softmax(dim=-1, dtype=torch.float32)` call.
