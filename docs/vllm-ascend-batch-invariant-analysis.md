# vllm-ascend Batch-Invariant 算子分析

## 1. 什么是 Batch-Invariant

> 参考博客：[Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

**Batch-invariant（批次不变性）** 是指计算内核无论处理的 batch size 如何变化，都能产生**完全相同的数值结果**。

### 为什么重要

LLM 推理的不确定性主要来源于：

1. **负载动态变化** → batch size 随时间不确定地变化
2. **浮点非结合性** → `(a + b) + c ≠ a + (b + c)` 在浮点数下成立
3. **并行策略依赖 batch size** → 不同 batch size 触发不同的并行化策略，改变归约顺序

```python
# 示例：相同数学运算因 batch size 不同产生不同结果
out1 = torch.mm(a[:1], b)      # 单元素 batch
out2 = torch.mm(a, b)[:1]      # 从完整 batch 中提取
# out1 ≠ out2  （数值差异由归约顺序不同引起）
```

### 两种并行策略对比

| 策略 | Batch-Invariant | 描述 |
|------|:-:|------|
| **Data-Parallel** | ✅ | 每个 batch 元素的归约在单个 compute core 内完成，与其他 batch 元素无关 |
| **Split-Reduction** | ❌ | 归约被拆分到多个 core，拆分方式依赖 batch size |

## 2. vllm-ascend 中的 Batch-Invariant 算子

vllm-ascend 提供两套实现后端：

- **Triton 实现**：`vllm_ascend/ops/triton/batch_invariant/` 目录
- **AscendC 实现**：`batch_invariant_ops` C 扩展包（优先使用）

通过环境变量 `VLLM_BATCH_INVARIANT=1` 一键启用。

### 2.1 MatMul 矩阵乘法

**源码**：`vllm_ascend/ops/triton/batch_invariant/matmul.py`

| 函数 | 注册的 aten 算子 | 说明 |
|------|-----------------|------|
| `mm_batch_invariant` | `aten::mm` | 2D × 2D 矩阵乘 |
| `bmm_batch_invariant` | `aten::bmm` | 批量矩阵乘（逐 batch 调用 persistent kernel） |
| `addmm_batch_invariant` | `aten::addmm` | 矩阵乘 + bias（kernel 内融合） |
| `matmul_batch_invariant` | `aten::matmul` | 通用矩阵乘（支持 2D/3D/4D） |
| `linear_batch_invariant` | `aten::linear` | 线性层 `x @ W^T + bias` |

**核心设计**：

- 使用 **persistent kernel**，固定 BLOCK 分块沿 M/N 维度并行
- K 维度归约在**单个 core 内完成**，不做 split-K
- `linear_persistent_kernel` 使用**固定 1D grid size**（= 设备 vectorcore 数 / 2），不随 batch 变化
- 累加在 float32 精度下进行：`acc += tl.dot(x_chunk, y_chunk, allow_tf32=False)`

### 2.2 RMSNorm

**源码**：`vllm_ascend/ops/triton/batch_invariant/rmsnorm.py`

| 函数 | 说明 |
|------|------|
| `rms_norm_batch_invariant` | RMS 归一化：`y = x / sqrt(mean(x²) + eps) * weight` |

**核心设计**：

- Grid size 固定为硬件 vectorcore 数量（常数），不随 batch 变化
- 每行的 sum-of-squares 归约在**单个 program 内独立完成**
- 中间计算提升到 float32 精度

### 2.3 Add + RMSNorm

**源码**：`vllm_ascend/batch_invariant.py:add_rms_norm`

| 函数 | 说明 |
|------|------|
| `add_rms_norm` | 将 fused `npu_add_rms_norm` 拆分为 `add` + `rms_norm` 两步 |

**原因**：代码注释明确说明 *"AclnnAddRmsNorm can't ensure batch invariant"*，fused 算子的内部归约策略依赖输入形状，因此拆分为：

1. 元素级加法（天然 batch-invariant）
2. 独立的 `npu_rms_norm`

### 2.4 Softmax

**源码**：`vllm_ascend/ops/triton/batch_invariant/softmax.py`

| 函数 | 注册的 aten 算子 | 说明 |
|------|-----------------|------|
| `softmax_batch_invariant` | `aten::softmax`, `aten::_softmax` | 确定性 softmax 实现 |

**核心设计**：

```python
# 分步计算，避免框架内部优化改变归约顺序
input_max = torch.amax(input_, dim=dim, keepdim=True)
input_ = input_ - input_max
exp_x = torch.exp(input_)
sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)  # 依赖 batch-invariant 的 sum
return exp_x / sum_exp_x
```

### 2.5 Mean 均值

**源码**：`vllm_ascend/ops/triton/batch_invariant/mean.py`

| 函数 | 说明 |
|------|------|
| `mean_batch_invariant` | 支持单维度和多维度均值 |

**核心设计**：

- Triton kernel 中 `grid = (M * K,)`，每个 program 负责一个输出元素
- 归约维度 N 上的累加在**单个 program 内独立完成**
- 多维度均值转换为 `sum / count` 形式

### 2.6 Reduce Sum

**源码**：`vllm_ascend/batch_invariant.py:reduce_sum`

| 函数 | 说明 |
|------|------|
| `reduce_sum` | 调用 AscendC 实现 `npu_reduce_sum_batch_invariant` |

替换 `torch.sum`，同时处理 CPU tensor 回退到原生实现。

### 2.7 Attention（注意力）

**源码**：`vllm_ascend/batch_invariant.py` (AscendC 实现)

| 函数 | 说明 |
|------|------|
| `npu_fused_infer_attention_score_batch_invariant` | 替换 `torch_npu.npu_fused_infer_attention_score` |

通过 monkey-patch 直接替换为 AscendC 的 batch-invariant 版本。

## 3. 环境级保障措施

### 3.1 环境变量

`override_envs_for_invariance()` 设置以下环境变量：

| 环境变量 | 值 | 作用 |
|---------|---|------|
| `VLLM_ASCEND_ENABLE_NZ` | `0` | 禁用 NZ format（引入非确定性） |
| `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` | `0` | 禁用 fused matmul+allreduce |
| `HCCL_DETERMINISTIC` | `strict` | HCCL 通信确定性模式 |
| `LCCL_DETERMINISTIC` | `1` | LCCL 通信确定性模式 |

### 3.2 兼容性调整

| 调整项 | 位置 | 说明 |
|-------|------|------|
| 禁用 custom ops | `vllm_ascend/utils.py` | 未实现 batch-invariant 的自定义算子被禁用 |
| 禁用 async exponential sampling | `vllm_ascend/ascend_config.py` | 异步采样与 batch-invariant 不兼容 |
| TopKTopP 采样器回退 | `vllm_ascend/sample/sampler.py` | 回退到 vLLM 原生实现 |

## 4. 总结表

| 算子类别 | Triton 实现 | AscendC 实现 | 核心策略 |
|---------|:-----------:|:------------:|---------|
| **mm / matmul / bmm / addmm** | ✅ | ✅ | 固定分块，K 维单 core 归约 |
| **linear** | ✅ | ✅（间接） | 固定 grid size persistent kernel |
| **RMSNorm** | ✅ | ✅（`npu_rms_norm`） | 行内独立归约，固定 grid |
| **Add + RMSNorm** | — | — | 拆分为 add + rms_norm 两步 |
| **Softmax** | ✅ | — | 分步计算，依赖 batch-invariant sum |
| **Mean** | ✅ | — | 每输出元素独立归约 |
| **Reduce Sum** | — | ✅ | AscendC 专用实现 |
| **Attention Score** | — | ✅ | AscendC 专用实现 |

## 5. Triton vs AscendC 实现分布与优先级

### 5.1 按实现后端分类

#### 仅 Triton 实现

| 算子 | 源码位置 | 说明 |
|------|---------|------|
| **Softmax** | `ops/triton/batch_invariant/softmax.py` | 纯 PyTorch 分步实现 |
| **Mean** | `ops/triton/batch_invariant/mean.py` | Triton kernel，单 program 归约 |
| **RMSNorm** | `ops/triton/batch_invariant/rmsnorm.py` | Triton kernel，固定 grid |
| **Linear** (`x @ W^T`) | `ops/triton/batch_invariant/matmul.py` | 固定 1D grid persistent kernel |
| **AddMM** (`a @ b + bias`) | `ops/triton/batch_invariant/matmul.py` | kernel 内融合 bias |
| **BMM** (批量矩阵乘) | `ops/triton/batch_invariant/matmul.py` | 逐 batch 调用 persistent kernel |

#### 仅 AscendC 实现

| 算子 | 注册方式 | 说明 |
|------|---------|------|
| **Reduce Sum** | `torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant` | 替换 `torch.sum` |
| **Attention Score** | `torch.ops.batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant` | 替换 `torch_npu.npu_fused_infer_attention_score` |

#### Triton 和 AscendC 都有实现（AscendC 优先）

| 算子 | Triton 实现 | AscendC 实现 |
|------|------------|-------------|
| **MM** (`a @ b`) | `mm_batch_invariant` | `npu_mm_batch_invariant` |
| **Matmul** (通用) | `matmul_batch_invariant` | `npu_matmul_batch_invariant` |

#### 非算子级替换

| 算子 | 实现方式 | 说明 |
|------|---------|------|
| **Add + RMSNorm** | 拆分为 `torch.add` + `torch_npu.npu_rms_norm` | 非新算子，而是将 fused 算子拆开以保证确定性 |

### 5.2 优先级逻辑

核心调度逻辑在 `batch_invariant.py:enable_batch_invariant_mode()` 中：

```python
# 第一步：Triton 独占算子（无论 AscendC 是否可用都注册）
if HAS_TRITON:
    _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::bmm", bmm_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::softmax", softmax_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::_softmax", softmax_batch_invariant, "NPU")

# 第二步：AscendC 优先路径
if HAS_ASCENDC_BATCH_INVARIANT:
    # mm, matmul, sum 用 AscendC
    # attention 通过 monkey-patch 替换
    # add_rms_norm 拆分为 add + rms_norm
    # torch.sum 直接 patch
elif HAS_TRITON:
    # mm, matmul, linear 用 Triton 兜底
```

**关键行为差异**：

| 场景 | mm / matmul | linear | addmm / bmm / softmax | sum | attention |
|------|:-----------:|:------:|:---------------------:|:---:|:---------:|
| **有 AscendC + 有 Triton** | AscendC | 不注册（间接走 AscendC matmul） | Triton | AscendC | AscendC |
| **无 AscendC + 有 Triton** | Triton | Triton | Triton | 原生 torch.sum | 原生 torch_npu |
| **无 AscendC + 无 Triton** | — | — | — | — | — |

**为什么有 AscendC 时不注册 Triton 版 linear？**

`aten::linear` 内部会调用 `aten::matmul`，当 matmul 已被替换为 AscendC 版本时，linear 自然间接走 AscendC 路径，无需单独注册。而 Triton 版 `linear_persistent_kernel` 是一个独立的 `x @ W^T` kernel，仅在没有 AscendC 时作为兜底方案。

## 6. 与博客对比

| 博客提及的算子 | vllm-ascend 覆盖情况 | 备注 |
|--------------|:-------------------:|------|
| MatMul | ✅ | Triton + AscendC 双实现 |
| RMSNorm | ✅ | Triton kernel + fused 拆分 |
| Attention | ✅ | AscendC 实现 |
| Softmax | ✅ | 博客未单独列出，vllm-ascend 额外覆盖 |
| Mean | ✅ | 博客未单独列出，vllm-ascend 额外覆盖 |
| Reduce Sum | ✅ | 博客未单独列出，vllm-ascend 额外覆盖 |
| Scatter (scatter_add) | ❌ | 博客提及为固有非确定性算子，vllm-ascend 未实现 |

vllm-ascend 的 batch-invariant 实现完整覆盖了博客中提到的核心算子，并额外增加了 Softmax、Mean、Reduce Sum 的 batch-invariant 实现。性能方面，AscendC 实现优先于 Triton 实现（如 matmul 在有 AscendC 时不再注册 Triton 版 linear，因为 linear 内部会调用已被替换的 matmul）。

## 7. MXFP8 Batch-Invariant 实现注意事项

现有的 batch-invariant 算子主要针对 BF16 精度设计。若要扩展到 MXFP8（Microscaling FP8）精度，需要关注以下关键问题。

### 7.1 MXFP8 当前计算流程

MXFP8 推理的核心流程（见 `vllm_ascend/quantization/methods/w8a8_mxfp8.py`）：

```
BF16 activation
    │
    ▼ npu_dynamic_mx_quant (动态量化)
FP8(E4M3FN) activation + FP8(E8M0FNU) per-group scale
    │
    ▼ npu_quant_matmul / npu_grouped_matmul (量化矩阵乘)
BF16 output
```

涉及的关键 torch_npu 算子：

| 算子 | 功能 | 调用位置 |
|------|------|---------|
| `npu_dynamic_mx_quant` | 动态 microscaling 量化：BF16 → FP8 + scale | `w8a8_mxfp8.py:78`, `device_op.py:296` |
| `npu_quant_matmul` | 量化矩阵乘（Linear 层） | `w8a8_mxfp8.py:84` |
| `npu_grouped_matmul` | 分组量化矩阵乘（MoE 层） | `moe_mlp.py` 多处 |
| `npu_grouped_matmul_swiglu_quant_v2` | 分组矩阵乘 + SwiGLU + 量化融合 | `device_op.py:322` |

### 7.2 需要新增/适配的 Batch-Invariant 算子

#### (1) 量化算子：`npu_dynamic_mx_quant`

**问题**：动态量化需要计算 per-group scale（每组取 max 再量化），scale 的计算涉及归约操作（group 内求 max/absmax），若并行策略依赖 batch size，会导致 scale 值不同，进而影响后续所有计算。

**注意事项**：
- 确保 per-group absmax 归约在**单个 core 内完成**，不跨 core split
- group_size（默认 32）维度上的归约顺序必须固定
- 量化 rounding 策略（RNE / stochastic）必须确定性

#### (2) 量化矩阵乘：`npu_quant_matmul`

**问题**：与 BF16 matmul 不同，量化 matmul 内部有额外步骤：
1. FP8 × FP8 乘法（累加到更高精度）
2. 反量化（乘以 scale）
3. 最终输出转换

**注意事项**：
- 现有 Triton batch-invariant matmul 将输入 `.to(tl.float32)` 后再做 `tl.dot`，MXFP8 需要保持 FP8 输入但控制累加精度
- `tl.dot` 的 `allow_tf32=False` 在 FP8 场景下的语义需要确认
- scale 的 broadcast 和乘法顺序必须固定
- dequant 阶段的归约顺序必须与 batch size 无关

#### (3) 分组矩阵乘：`npu_grouped_matmul`

**问题**：MoE 场景下，token 被路由到不同 expert，每个 expert 处理的 token 数量**天然依赖 batch size**，这是 MXFP8 batch-invariance 最大的挑战。

**注意事项**：
- `group_list`（每个 expert 分到的 token 数）随 batch 变化，直接影响 `npu_grouped_matmul` 内部的并行策略
- 需要确保即使 group_list 不同，同一个 token 在同一个 expert 上的计算结果**完全一致**
- 可能需要为每个 group 独立调用 batch-invariant matmul，而非使用 fused grouped matmul

#### (4) 融合算子：`npu_grouped_matmul_swiglu_quant_v2`

**问题**：将 GMM + SwiGLU + 再量化融合为一个算子，融合算子的内部实现很可能不保证 batch-invariance（类似 `npu_add_rms_norm` 的情况）。

**注意事项**：
- 可能需要像 `add_rms_norm` 一样，将融合算子拆分为独立步骤
- 拆分为：`grouped_matmul` → `swiglu` → `dynamic_mx_quant`，每步分别保证 batch-invariance
- 会有性能损失，但这是确保确定性的必要代价

### 7.3 精度相关的特殊挑战

#### Scale 计算的确定性

MXFP8 的 microscaling 引入了额外的归约操作（per-group absmax），这在 BF16 中不存在：

```python
# MXFP8 量化伪代码
for each group of 32 elements:
    scale = max(abs(group))              # 归约操作 — 必须 batch-invariant
    quantized = round(group / scale)     # rounding — 必须确定性
```

- **group 划分方式**必须与 batch size 无关（当前按 hidden_dim 分组，group_size=32，天然与 batch 无关）
- 但若 batch 维度影响了 tensor 的内存布局，可能间接影响 group 划分

#### 累加精度链路

BF16 batch-invariant matmul 的累加路径：
```
BF16 → cast to FP32 → FP32 dot → FP32 accumulate → cast back to BF16
```

MXFP8 需要的累加路径：
```
FP8 × FP8 → FP32 accumulate → × scale_a × scale_b (dequant) → FP32 → cast to BF16
```

**关键差异**：dequant 阶段 scale 的乘法顺序和精度必须固定。如果 `scale_a` 和 `scale_b` 的广播/乘法在不同 batch size 下使用不同的顺序，结果会不同。

#### Rounding 模式

FP8 量化的 rounding（舍入）必须使用确定性模式（如 Round-to-Nearest-Even），不能使用 stochastic rounding。需要确认 `npu_dynamic_mx_quant` 内部使用的 rounding 策略。

### 7.4 实现路径建议

```
                    ┌──────────────────────────────────────┐
                    │  优先级 1: torch_npu 原生算子验证      │
                    │  验证 npu_dynamic_mx_quant /          │
                    │  npu_quant_matmul 本身是否已           │
                    │  batch-invariant                      │
                    └──────────────┬───────────────────────┘
                                   │
                      ┌────────────┴────────────┐
                      │ 是                       │ 否
                      ▼                          ▼
              ┌───────────────┐    ┌──────────────────────────────┐
              │ 直接注册为     │    │  优先级 2: 拆分融合算子         │
              │ batch-invariant│    │  npu_grouped_matmul_swiglu    │
              │ 实现          │    │  _quant_v2 → 拆为独立步骤      │
              └───────────────┘    └──────────────┬───────────────┘
                                                   │
                                                   ▼
                                   ┌──────────────────────────────┐
                                   │  优先级 3: Triton 自定义实现    │
                                   │  FP8 persistent matmul       │
                                   │  + 确定性 dequant             │
                                   └──────────────┬───────────────┘
                                                   │
                                                   ▼
                                   ┌──────────────────────────────┐
                                   │  优先级 4: AscendC 定制实现    │
                                   │  npu_quant_matmul_batch       │
                                   │  _invariant 等                │
                                   └──────────────────────────────┘
```

### 7.5 对比总结：BF16 vs MXFP8 Batch-Invariant

| 维度 | BF16 | MXFP8 |
|------|------|-------|
| **核心算子** | mm / matmul / linear | npu_quant_matmul / npu_grouped_matmul |
| **额外算子** | — | npu_dynamic_mx_quant（量化）|
| **归约来源** | K 维 dot product | K 维 dot product + per-group absmax (scale) |
| **融合算子风险** | npu_add_rms_norm | npu_grouped_matmul_swiglu_quant_v2 |
| **精度链路** | BF16 → FP32 → BF16 | FP8 → FP32 (+ scale dequant) → BF16 |
| **MoE 特殊性** | group_list 影响并行 | group_list 影响并行 + scale 依赖 token 数据 |
| **Triton kernel 改造** | `.to(tl.float32)` + `tl.dot` | 需要 FP8 原生 `tl.dot` + scale 处理 |
| **预期性能损失** | ~38%（参考博客） | 预计 > 38%（额外的 scale 计算 + 融合算子拆分） |

## 8. MXFP8 Batch-Invariant 验证结果

> 以下结果在 Ascend A5 (内测版, Ascend910_9589) + CANN 8.5.0 + torch_npu 2.7.1 上验证。

### 8.1 验证结论

**所有 MXFP8 核心算子均已原生支持 batch-invariant**，无需自定义 Triton/AscendC 实现。
对应 Section 7.4 实现路径的**优先级 1**：直接注册为 batch-invariant 实现。

| 算子 | Batch-Invariant | 测试范围 | 说明 |
|------|:-:|------|------|
| `npu_dynamic_mx_quant` | ✅ | M=[1-2048], K=[4096-14336] | per-token 量化，天然与 batch 无关 |
| `npu_quant_matmul` | ✅ | M=[1-2048], K=[4096-14336], N=[2048-18432] | K 维归约顺序不受 M 影响 |
| `npu_grouped_matmul` | ✅ | 不同 group_list, M=[16-512] | 同 token 同 expert 结果一致 |
| `npu_grouped_matmul_swiglu_quant_v2` | ✅ | expert token 数量 1-32, 总 M=16-512 | 融合算子内部确定性 |

### 8.2 详细测试用例

#### npu_dynamic_mx_quant
- 对 64 行输入，分别以 BS=1/2/4/8/16/32/64 chunk 处理后拼接，与完整 batch 结果逐 bit 一致
- 逐行处理 vs 完整 batch：quantized tensor 和 scale 均完全一致

#### npu_quant_matmul
- 模型级维度 (DeepSeek-like): K=7168, N=18432, M=256 — 各种 chunk size 均一致
- 奇数 chunk (BS=7, 13, 37) 也完全一致，排除对齐偶发一致的可能

#### npu_grouped_matmul (MoE GMM2)
- 不同 group_list `[16,32,48,64]` vs `[8,24,48,64]`：共享 expert 区域逐 bit 一致
- 与逐 expert `npu_quant_matmul` 结果完全一致

#### npu_grouped_matmul_swiglu_quant_v2 (MoE GMM1 + SwiGLU)
- Expert0 获取固定 8 tokens，其他 expert 分配从 `[8,8,8,40]` 到 `[52,2,2]` 变化 → 全部 MATCH
- Expert0 token 数量从 1 到 32 变化，token[0] 输出不变 → 全部 MATCH
- 总 M 从 16 到 512 变化，token[0] 输出不变 → 全部 MATCH
- 相同输入连续 5 次调用 → 完全确定性

### 8.3 集成方式

在 `batch_invariant.py` 的 `enable_batch_invariant_mode()` 中，通过 monkey-patch 注册：

```python
# MXFP8 operators are verified batch-invariant on Ascend A5.
torch_npu.npu_dynamic_mx_quant = npu_dynamic_mx_quant_batch_invariant
torch_npu.npu_quant_matmul = npu_quant_matmul_batch_invariant
```

包装函数位于 `vllm_ascend/ops/triton/batch_invariant/mxfp8_quant_matmul.py`，
当前为直接透传（passthrough），若未来硬件/固件行为变化，可在此处插入固定 chunk 处理等保障措施。

## 9. 全算子 Batch-Invariant 验证总结

> 测试环境：Ascend A5 (内测版, Ascend910_9589) + CANN 8.5.0 + torch_npu 2.7.1
>
> 测试方法：对同一输入，以不同 batch size 分 chunk 处理后拼接，与完整 batch 结果逐 bit 比较。

### 9.1 本次实验验证的算子

#### MXFP8 量化算子

| 算子 | BI 结果 | 实现层面 | 测试规模 |
|------|:-:|:-:|------|
| `npu_dynamic_mx_quant` | ✅ | NPU 原生 | M=[1-2048], K=[4096-14336], BS=1/2/4/8/16/32/64 |
| `npu_quant_matmul` | ✅ | NPU 原生 | M=[1-2048], K=[4096-14336], N=[2048-18432], BS=1/7/13/32/37/64/128 |
| `npu_grouped_matmul` | ✅ | NPU 原生 | 不同 group_list, M=[16-512], 4 experts |
| `npu_grouped_matmul_swiglu_quant_v2` | ✅ | NPU 原生 | expert token=[1-32], 总 M=[16-512], 确定性 5 次重复 |

#### Norm 算子

| 算子 | BI 结果 | 实现层面 | 测试规模 |
|------|:-:|:-:|------|
| `npu_rms_norm` | ✅ | NPU 原生 | M=256, hidden=[2048-14336], BS=1/3/7/16/32/64/128, 逐行验证 |
| `npu_add_rms_norm`（融合） | ✅ | NPU 原生 | M=128, hidden=4096, BS=1/7/16/32/64 |
| split `add` + `npu_rms_norm` | ✅ | PyTorch 拆分 | 同上 |

**发现**：`npu_add_rms_norm` 在当前 A5 固件上实际已经是 batch-invariant 的，但代码中的拆分方案作为跨固件版本的安全保障仍有价值。

### 9.2 项目已有的 Batch-Invariant 实现（非本次验证）

| 算子 | 实现层面 | 核心策略 |
|------|:-:|------|
| mm / matmul / bmm / addmm | Triton persistent kernel / AscendC | 固定分块，K 维单 core 归约 |
| linear | Triton persistent kernel | 固定 1D grid size |
| RMSNorm | Triton kernel | 行内独立归约，固定 grid |
| Softmax | PyTorch 分步计算 | `amax → sub → exp → sum → div` |
| Mean | Triton kernel | 每输出元素独立归约 |
| Reduce Sum | AscendC | NPU 专用实现 |
| Attention Score | AscendC | NPU 专用实现 |

### 9.3 与博客对比

> 参考博客：[Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

| 博客提及 | 项目已有 BI 实现 | 本次验证 NPU 原生 BI | 状态 |
|---------|:-:|:-:|:-:|
| MatMul | ✅ Triton + AscendC | — | 已覆盖 |
| RMSNorm | ✅ Triton + 拆分 | ✅ `npu_rms_norm` | 已覆盖 |
| Attention | ✅ AscendC | — | 已覆盖 |
| Scatter (`scatter_add`) | ❌ | ❌ | **未覆盖** |

博客未提及，项目已覆盖：

| 算子 | 实现层面 |
|------|:-:|
| Softmax | PyTorch 分步 |
| Mean | Triton |
| Reduce Sum | AscendC |

博客未提及，本次新增验证（MXFP8）：

| 算子 | 结果 |
|------|:-:|
| `npu_dynamic_mx_quant` | ✅ NPU 原生 BI |
| `npu_quant_matmul` | ✅ NPU 原生 BI |
| `npu_grouped_matmul` | ✅ NPU 原生 BI |
| `npu_grouped_matmul_swiglu_quant_v2` | ✅ NPU 原生 BI |

### 9.4 唯一缺口：scatter_add

`scatter_add` 是博客中标记的**固有非确定性算子**，项目未实现 batch-invariant 替代方案。

**非确定性根源**：多个值按动态索引累加到同一位置时，累加顺序由硬件调度决定，无法提前固定。与 matmul 的 K 维归约不同，scatter_add 的冲突位置是**数据依赖的**。

**在 LLM 推理中的位置**：MoE token combine 阶段，多个 expert 的输出按原始 token 位置累加（top-k routing 下同一 token 可能有多个 expert 结果）。

**可能的解决方案**（参考博客）：
1. **排序后归约**：先按目标索引排序，再顺序累加（确定性但需额外排序开销）
2. **串行化**：逐元素处理（确定性但极慢）
3. **避免使用**：重构 token combine 逻辑，用 gather + 加权求和替代 scatter_add

### 9.5 实现层面总结

| 层面 | 算子 | 说明 |
|------|------|------|
| **NPU 原生（已验证 BI）** | MXFP8 全套 + rms_norm + add_rms_norm | 直接使用，passthrough wrapper 注册 |
| **Triton** | matmul / linear / rmsnorm / mean | persistent kernel，固定 grid/block |
| **AscendC** | mm / matmul / sum / attention | C 扩展包，优先于 Triton |
| **PyTorch** | softmax / add+rms_norm 拆分 | 分步计算避免框架优化 |

### 9.6 极端维度 Batch-Invariant 验证

为排除硬件在大矩阵下因 split-K tiling 或并行策略变化导致的非确定性，对所有算子进行了极端维度压力测试。

#### npu_dynamic_mx_quant (39/39 PASS)

| 测试类别 | 维度范围 | 结果 |
|---------|---------|:---:|
| 超大 K | K=16384, 32768, **65536** | ✅ |
| 超大 M | M=1024, 4096, **8192** | ✅ |
| 非对齐 | K=4128(129×32), M=63(奇数) | ✅ |
| DeepSeek V3 | K=7168, K=18432 | ✅ |

#### npu_quant_matmul (64/64 PASS)

| 测试类别 | 维度范围 | 结果 |
|---------|---------|:---:|
| 超大 K | K=16384, 32768, **65536** (2048 个 group) | ✅ |
| 超大 M | M=1024, 4096, **8192** | ✅ |
| 非对齐 | K=4128, K=4064, N=2049, N=1023, M=63, M=1 | ✅ |
| 极端比例 | M=4096/K=128(tall), K=4096/N=32768(wide) | ✅ |
| DeepSeek V3 | gate_up(K=7168,N=18432), down(K=9216,N=7168), qkv(K=7168,N=1536) | ✅ |

#### npu_grouped_matmul (15/15 PASS)

| 测试类别 | 维度范围 | 结果 |
|---------|---------|:---:|
| 大维度 | K=7168/N=2048, K=16384/N=4096 | ✅ |
| 多 expert | 8 experts, M=1024 | ✅ |
| 大 batch | M=2048 | ✅ |
| DeepSeek V3 | down_proj K=9216/N=7168 | ✅ |

#### npu_grouped_matmul_swiglu_quant_v2 (全部 PASS)

| 测试类别 | 维度范围 | 结果 |
|---------|---------|:---:|
| group_list 变化 | uniform / skewed / extreme(expert0=1 token) | ✅ |
| 总 M 变化 | M=64 vs M=128，共享 token 逐 bit 一致 | ✅ |
| 大维度 | K=7168/GU=18432(DSv3), K=14336/GU=8192 | ✅ |
| 多 expert | 8 experts, M=256 | ✅ |

#### npu_rms_norm (31/31 PASS)

| 测试类别 | 维度范围 | 结果 |
|---------|---------|:---:|
| 超大 hidden | hidden=16384, 32768, **65536** | ✅ |
| 超大 M | M=4096, **8192** | ✅ |
| 非对齐 | hidden=4097(奇数), M=63 | ✅ |
| DeepSeek V3 | hidden=7168 | ✅ |

#### npu_add_rms_norm (23/23 PASS)

| 测试类别 | 维度范围 | 结果 |
|---------|---------|:---:|
| 超大 hidden | hidden=16384, 32768 | ✅ |
| 超大 M | M=4096, **8192** | ✅ |
| DeepSeek V3 | hidden=7168 | ✅ |
| 非对齐 | M=63 | ✅ |

#### 结论

即使在 K=65536（2048 个 microscaling group，对 split-K tiling 压力极大）和 M=8192（大 batch）下，所有算子的结果仍然**逐 bit 一致**。这强烈支持 NPU 的 MXFP8 matmul 实现采用了**固定归约路径**（K 维不做跨 core 拆分，或拆分方式不依赖 M 维），因此天然保证 batch-invariance。

### 9.7 测试脚本索引

| 脚本 | 测试内容 | 位置 |
|------|---------|------|
| `test_mxfp8_batch_invariant.py` | MXFP8 原生算子 BI 验证 | `tests/ut/ops/` |
| `test_mxfp8_integration.py` | MXFP8 BI wrapper 集成测试 (21/21) | `tests/ut/ops/` |
| `test_swiglu_quant_v2.py` | MoE 融合算子 BI 验证 | `tests/ut/ops/` |
| `test_rmsnorm_bi.py` | RMSNorm / AddRMSNorm BI 验证 (22/22) | `tests/ut/ops/` |
| `test_extreme.py` | npu_quant_matmul 极端维度 (64/64) | `tests/ut/ops/` |
| `test_extreme_all.py` | 全算子极端维度压力测试 | `tests/ut/ops/` |

## 10. 新增 Triton BI 算子实现与基准测试

针对原 vllm-ascend 中**未实现**的 6 个 batch-invariant 算子，分析其在 Qwen 模型推理中的实际位置，编写参考实现并在 A5 上做性能 benchmark。

### 10.1 算子覆盖情况

| 算子 | 原项目是否有 BI 实现 | 本次工作 |
|------|:-:|------|
| `_logsoftmax_batch_invariant` | ❌ | 实现 + 测试 |
| `topk_softmax_batch_invariant` | ❌ | 实现 + 测试 |
| `moe_gating_batch_invariant` | ❌ | 实现 + 测试 |
| `silu_and_mul_batch_invariant` | ❌ | 实现 + 测试 |
| `rotary_embedding_batch_invariant` | ❌ | 实现 + 测试 |
| `all_reduce_batch_invariant` | ❌ | 跳过（已由 `HCCL_DETERMINISTIC=strict` 控制） |

### 10.2 BI 实现策略

所有 6 个算子均**不需要新写 Triton kernel** — 通过组合现有 BI 原语即可保证 BI：

| 算子 | BI 实现思路 |
|------|------|
| `silu_and_mul` | 元素级 `F.silu(x[...,:H]) * x[...,H:]`，无任何归约 → 天然 BI |
| `rotary_embedding` | 元素级 `x*cos + rotate_half(x)*sin`，per-token 独立 → 天然 BI |
| `_logsoftmax` | 拆为 `amax → sub → exp → sum → log → sub`，所有归约沿 vocab 维度 |
| `topk_softmax` | BI softmax + topk（topk 是确定性元素比较） |
| `moe_gating` | linear (BI matmul) + topk_softmax (BI) |

### 10.3 基础 BI 测试结果（合成数据，BF16）

A5 服务器上验证（M=256，不同 chunk size 拼接后逐 bit 比较）：

| 算子 | BI 验证 | vs Native (max_diff) | 性能 (M=256) |
|------|:-:|------|:-:|
| `silu_and_mul` | ✅ | 6.25e-2 (bf16 内) | **1.88x** 慢 |
| `rotary_embedding` | ✅ | 6.25e-2 | **4.34x** |
| `logsoftmax` | ✅ | 6.25e-2 | **4.45x** |
| `topk_softmax` | ✅ | 3.91e-3 | **3.14x** |
| `moe_gating` | ✅ | 3.91e-3 | **2.36x** |

性能开销来自把 fused kernel 拆成多次 PyTorch 操作（多次 kernel launch）。

### 10.4 性能随 M 变化趋势

| M | silu | rotary | logsoftmax | topk_softmax | moe_gating |
|---|:-:|:-:|:-:|:-:|:-:|
| 16 | 1.73x | 4.06x | 5.91x | 3.17x | 2.38x |
| 256 | 1.88x | 4.34x | 4.45x | 3.14x | 2.36x |
| 4096 | 3.01x | 4.98x | 3.87x | 3.18x | 2.73x |

### 10.5 MXFP8 端到端流水线 BI 验证

将这些 BI 算子接在 MXFP8 matmul 之后，验证完整链路 BI（合成数据）：

| 流水线 | 模型场景 | 结果 |
|--------|---------|:-:|
| MXFP8 Linear → `silu_and_mul` | FFN gate_up_proj → SwiGLU | ✅ |
| MXFP8 Linear → `rotary_embedding` | Q/K projection → RoPE | ✅ |
| MXFP8 Linear → `logsoftmax` | lm_head → log probabilities | ✅ |
| MXFP8 Linear → `topk_softmax` | MoE gate (quantized) → expert selection | ✅ |
| BF16 gate → `topk_softmax` | 实际 MoE 模式（gate 通常保留 BF16） | ✅ |

## 11. Qwen3 真实计算路径分析

通过阅读 vllm 源码，确认 Qwen3 在 NPU 上推理的实际 dispatch 路径（不依赖具体模型文件，纯静态分析）。

### 11.1 关键代码位置

| 阶段 | 实际算子 | 调用位置 |
|------|---------|---------|
| Linear (qkv_proj/gate_up_proj/down_proj/lm_head) | `npu_quant_matmul` | `vllm_ascend/quantization/methods/w8a8_mxfp8.py:84` |
| RoPE | `torch_npu._npu_rotary_embedding` (或 Triton fallback) | `vllm_ascend/ops/rotary_embedding.py:189` |
| SiluAndMul | `forward_native`（PyTorch 原生） | `vllm/.../activation.py:140`（NPU 无 forward_npu） |
| MoE Gating | `torch_npu.npu_moe_gating_top_k` | `vllm_ascend/device/device_op.py:248` |
| MoE GMM1+SwiGLU | `npu_grouped_matmul_swiglu_quant_v2` | `vllm_ascend/device/device_op.py:322` |
| MoE GMM2 | `npu_grouped_matmul` | `vllm_ascend/device/device_op.py:433` |
| RMSNorm | `npu_rms_norm` | `vllm_ascend/batch_invariant.py:62` |

### 11.2 真实算子 BI 测试

直接调用上述实际 NPU 算子验证 BI（合成数据，A5）：

| 算子 | 测试结果 |
|------|:-:|
| `torch_npu._npu_rotary_embedding` | ✅ PASS（chunk size 1/7/32/64/128 全部一致） |
| `torch_npu.npu_moe_gating_top_k` (softmax+topk) | ✅ PASS |
| `torch_npu.npu_moe_gating_top_k` (sigmoid+grouped) | ⚠️ NPU 报错跳过 |

### 11.3 关键发现

**Qwen3 推理路径中所有关键算子在 NPU 上都已经原生 batch-invariant**：

- ✅ MXFP8 量化算子（已验证）
- ✅ RMSNorm（已验证）
- ✅ RoPE（本次验证）
- ✅ MoE Gating（本次验证）
- ✅ SiluAndMul（PyTorch native，元素级，天然 BI）

理论上 Qwen3 在 vllm-ascend + 本仓库 `enable_batch_invariant_mode()` 下进行 MXFP8 推理应当**端到端 batch-invariant**。

### 11.4 端到端推理验证状态

⚠️ **未完成**。受服务器环境限制（`acl`/NNAL 等 CANN 组件缺失），真实 vllm 端到端推理测试一直未跑通。所有结论基于：

1. **算子级 BI 测试**（合成数据）— 184+ 个测试用例全部通过
2. **MXFP8 上下游链路 BI 测试**（MXFP8 matmul → BI op 流水线）— 5/5 通过
3. **真实 NPU 算子 BI 测试**（直接调用 `torch_npu._npu_rotary_embedding` 等）— 全部通过
4. **vllm 源码静态分析** — 确认 Qwen3 dispatch 路径

剩余的端到端 vllm 推理（用 GSM8K 等数据集对比不同 batch size 下 token 级输出）需要在装好完整 CANN（含 pyACL/NNAL）的环境上运行。

### 11.5 测试脚本索引（新增）

| 脚本 | 测试内容 |
|------|---------|
| `test_new_bi_ops.py` | 6 个新算子 PyTorch 拆分实现的 BI 验证 + 性能 benchmark |
| `test_new_bi_ops_mxfp8.py` | MXFP8 → BI op 流水线 (5/5 PASS) |
| `test_real_npu_ops_bi.py` | 真实 NPU 算子 BI (`_npu_rotary_embedding`, `npu_moe_gating_top_k`) |
| `convert_to_mxfp8.py` | BF16 模型 → vllm-ascend W8A8_MXFP8 格式离线转换工具 |
| `test_inference_bi.py` | vllm 端到端推理 BI 测试（环境就绪后可直接运行） |
| `test_triton_bi_kernels.py` | 5 个原生 Triton kernel 的 BI/正确性/性能测试 |

## 12. 原生 Triton BI Kernel 实现

按照现有 `rmsnorm.py` 的设计模式（fixed grid + per-row 独立处理 + 行内 reduction），为之前的 PyTorch 拆分实现编写真正的 Triton kernel。

### 12.1 实现的 Kernel

| Kernel 文件 | 算子 | 模式 |
|------------|------|------|
| `silu_and_mul.py` | `silu_and_mul_batch_invariant` | 元素级（Per-row, BLOCK_SIZE 列循环） |
| `logsoftmax.py` | `logsoftmax_batch_invariant` | Per-row 三遍扫描（max → sum_exp → output） |
| `rotary_embedding.py` | `rotary_embedding_batch_invariant` | Per-token，每个 program 处理多个 token 的 Q/K |
| `topk_softmax.py` | `topk_softmax_batch_invariant` | Per-row softmax + repeated argmax |
| `moe_gating.py` | `moe_gating_batch_invariant` | 复用 `linear_persistent` (现有) + `topk_softmax` |

### 12.2 设计原则（沿用 rmsnorm.py 模板）

```python
def kernel(...):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    rows_per_program = (n_rows + n_programs - 1) // n_programs
    start_row = pid * rows_per_program
    end_row = tl.minimum(start_row + rows_per_program, n_rows)
    for row_idx in range(start_row, end_row):
        # 行内独立计算（reductions 只沿最后一维）
```

**Grid 大小固定**：`grid = (min(n_rows, num_vectorcore),)`，与 batch size 无关。
**每行独立处理**：rows 在 program 之间静态分配，row i 的输出只依赖 row i 的输入。

### 12.3 编译验证状态

⚠️ **这些 Triton kernel 在当前 A5 服务器上无法编译**。

错误：`bishengir-compile` 报告 `Unknown command line argument '-cce-vf-aa-between-iters=true'`，是 triton-ascend 版本（3.4.0.dev*）与服务器上 CANN-1223 自带的 `bisheng` 编译器版本不匹配。

尝试过 triton-ascend 的 3 个版本（dev2026010422 / dev2026011116 / dev2026032222）均失败。

### 12.4 设计正确性证据（不依赖编译）

虽然无法在此服务器上编译运行，但有以下证据支持设计正确性：

1. **沿用现有可工作的模板**：`rmsnorm.py`（已在项目中工作）使用同样的 fixed grid + per-row 模式
2. **PyTorch 等价实现已验证**：`test_new_bi_ops.py` 用与 Triton kernel 数学等价的 PyTorch 拆分实现，**全部通过 BI 验证**（chunk size 1/7/32/64/128 逐 bit 一致）
3. **BI 来自结构而非数值**：BI 性质由 kernel 的并行结构保证（per-row 独立）— 任何遵循此结构的实现（无论 Triton/PyTorch/AscendC）都是 BI 的

### 12.5 如何在适配环境下验证

在装有匹配版本 triton-ascend + bisheng 编译器的环境上：

```bash
source ~/miniconda3/bin/activate <env_with_compatible_triton>
export PATH=/path/to/bisheng/bin:$PATH
export TRITON_ASCEND_ARCH=Ascend910_9589
cd vllm-ascend
python tests/ut/ops/test_triton_bi_kernels.py
```

预期：5 个 kernel 全部通过 BI + 正确性测试，性能因 fused kernel 通常优于 PyTorch 拆分版本。
