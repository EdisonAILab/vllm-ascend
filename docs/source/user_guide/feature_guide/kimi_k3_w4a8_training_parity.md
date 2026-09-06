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
export VLLM_ASCEND_KIMI_REFERENCE_KDA_CORE=0
export VLLM_ASCEND_KIMI_NATIVE_KDA_CORE=1
export VLLM_ASCEND_KIMI_NATIVE_STATE_OPS=1
export VLLM_ASCEND_KIMI_KDA_NATIVE_NORM_GATE=1
export VLLM_ASCEND_KIMI_GATE_LOWER_BOUND=-5.0
export VLLM_ASCEND_KIMI_REFERENCE_ATTN_RES=1
export VLLM_ASCEND_KIMI_VECTORIZED_ATTN_RES=1
export VLLM_ASCEND_KIMI_NATIVE_ATTN_RES=1
export VLLM_ASCEND_KIMI_REFERENCE_ROUTER_FP32=1
export VLLM_ASCEND_KIMI_REFERENCE_ROUTING=0
export VLLM_ASCEND_KIMI_REFERENCE_ROUTED_RMS_NORM=1
export VLLM_ASCEND_KIMI_DECOMPOSED_ROUTED_RMS_NORM=0
export VLLM_ASCEND_KIMI_REFERENCE_MLA_RMS_NORM=1
export VLLM_ASCEND_KIMI_DECOMPOSED_MLA_RMS_NORM=0
export VLLM_ASCEND_KIMI_REFERENCE_MLA_DECODE=1
export VLLM_ASCEND_KIMI_CONCAT_SHORT_MLA_ROPE=1
export VLLM_ASCEND_NATIVE_SLOT_MAPPING=1
export VLLM_ASCEND_SKIP_UNUSED_PENALTY_WARMUP=1
```

The accepted profile keeps production MoE routing and the production causal
convolution/cache update, but runs SiLU separately. It keeps explicit FP32
RMSNorm reductions, router arithmetic, KDA recurrence, MLA decode, and AttnRes
reductions where the optimized kernels did not satisfy the byte-exact
contract. For W4A8 SiTU, the normal dispatch boundary quantizes the sorted
BF16 tokens with `npu_dynamic_mx_quant`, and routing weights are applied before
the second dynamic MXFP8 quantization and grouped GEMM. The fused routing
quantizer's E4M3 modes 3 and 17 use a different E8M0 scale-rounding policy and
were rejected by the serialized-logprob gate.

The rejected decomposed native RMSNorm variants remain available only for
diagnosis. They passed a short gate but differed at decode step 883, so both
`VLLM_ASCEND_KIMI_DECOMPOSED_*_RMS_NORM` variables must remain disabled for
byte-exact runs.

## Result

With prompt token IDs 1 through 32, one prefill plus 1,023 real cached-decode
calls matched the Megatron reference for all 1,024 complete FP32 native
logprob rows. The comparison covered 167,772,160 values and 671,088,640 bytes
with zero differing bytes. Native logprobs were captured immediately after
each engine's own `logits.log_softmax(dim=-1, dtype=torch.float32)` call.
