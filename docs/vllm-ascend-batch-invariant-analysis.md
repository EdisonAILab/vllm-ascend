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
