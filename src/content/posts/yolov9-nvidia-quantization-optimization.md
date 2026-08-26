---
author: SunJiating
pubDatetime: 2026-08-26T03:40:00Z
title: YOLOv9 的 NVIDIA 平台量化优化：量化后速度从 11% 提升到 30%
draft: false
tags:
  - TensorRT
  - YOLOv9
  - INT8
  - 性能优化
description: 记录 YOLOv9 检测模型从 INT8 量化到动态 Batch 1~8 三 profile 引擎的完整分析、图优化、构建与验证过程，最终 Batch 1~8 平均提速 41.56%。
---

本文记录 YOLOv9 检测模型从初始 INT8 量化到最终动态 Batch 1~8 引擎的完整分析、定位、图优化、构建、验证和部署过程。

## 1. 目标与验收标准

### 1.1 目标模型

```text
<MODEL_PATH>
```

模型是动态 Batch 输入的 YOLOv9 风格目标检测网络。初始方案只排除了 `/model.42.*`，目标是在不明显损失精度的前提下获得至少 30% 的推理速度提升，并导出 Batch 1~8 均可运行的动态 Batch 引擎。

### 1.2 验收定义

速度提升按同一 Batch、同一输入尺寸、同一 TensorRT 测试口径计算：

```text
提升率 = (FP16 延迟 - INT8 延迟) / FP16 延迟 * 100%
```

最终统一口径为 GPU 端到端计算延迟：

```bash
trtexec \
  --loadEngine=ENGINE \
  --shapes=input:${BATCH}x3x416x768 \
  --warmUp=1000 \
  --duration=10 \
  --useCudaGraph \
  --useSpinWait \
  --noDataTransfers
```

`--noDataTransfers` 排除 Host 到 Device 和 Device 到 Host 的固定传输开销；`--useCudaGraph` 降低重复 launch 开销；`--useSpinWait` 用忙等待换取更稳定的 GPU 时间。FP16 和 INT8 必须使用完全相同的命令参数和输入 shape。

## 2. 目标机器和软件环境

### 2.1 硬件

TensorRT 构建日志确认目标设备为：

```text
GPU: NVIDIA GeForce RTX 3090
Compute Capability: 8.6
SM 数量: 82
显存: 24124 MiB
```

这是实际部署机器，因此引擎必须在该机器上构建。TensorRT tactic 选择和 GPU 架构相关，不能把在其他 GPU 上构建的 engine 直接当作等价结果。

### 2.2 软件

```text
TensorRT: 10.12.0（Python 包日志为 10.12.0.36）
ONNX opset: ModelOpt 预处理后为 opset 19
numpy: 1.26.4
nvidia-modelopt: 0.21.0
onnx: 1.17.0
onnx-graphsurgeon: 0.5.2
onnxruntime: 1.20.1
opencv-python: 4.10.0.84
```

量化脚本在启动时显式查找 CUDA/cuDNN 动态库，并设置 `LD_LIBRARY_PATH` 和 `LD_PRELOAD`。如果在另一台机器执行，必须先确认这些库和 TensorRT 版本匹配，否则 ModelOpt/ONNX Runtime 可能退回 CPU 或加载失败。

## 3. 初始 INT8 量化策略

量化入口是 `quantization/model_quantize_YOLOv9.py`，关键调用为：

```python
moq.quantize(
    onnx_path=str(args.model),
    quantize_mode="int8",
    calibration_data=calibration_data,
    output_path=str(quant_path),
    calibration_method="max",
    high_precision_dtype="fp32",
    dq_only=True,
    nodes_to_exclude=[r"/model\.42.*"],
)
```

各参数含义：

- `quantize_mode="int8"`：为可量化算子生成 INT8 权重/激活路径。
- `calibration_method="max"`：使用校准期间观测到的最大绝对值确定范围，简单稳定但对异常值敏感。
- `high_precision_dtype="fp32"`：未量化路径和高精度中间结果使用 FP32。
- `dq_only=True`：权重以 INT8 保存并保留 DQ 节点，便于 TensorRT 识别量化权重，同时保留图的浮点语义。
- `nodes_to_exclude=[r"/model\.42.*"]`：正则匹配 `/model.42` 检测头的节点，初衷是保护精度。

实际量化日志显示：原图共有 262 个 Conv，ModelOpt 找到 370 个最终可量化节点，输出 DQ-only 模型共 1970 个节点，量化节点统计为 379。日志也确认输入是动态的 `input`，没有发现自定义层。

## 4. Baseline 构建与初始结果

### 4.1 单动态 profile 构建

原始脚本 `quantization/scripts/onnx2engine.sh` 使用一个宽范围 profile：

```text
min = 1x3x416x768
opt = 4x3x416x768
max = 8x3x416x768
```

完整命令（INT8）：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=<INT8_ONNX_PATH> \
  --minShapes=input:1x3x416x768 \
  --optShapes=input:4x3x416x768 \
  --maxShapes=input:8x3x416x768 \
  --saveEngine=<INT8_ENGINE_PATH> \
  --int8 --fp16 \
  --memPoolSize=workspace:4096 \
  --precisionConstraints=prefer \
  --avgTiming=1
```

FP16 对照引擎只去掉 `--int8`，ONNX 输入使用原始 FP32 模型：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=<MODEL_PATH> \
  --minShapes=input:1x3x416x768 \
  --optShapes=input:4x3x416x768 \
  --maxShapes=input:8x3x416x768 \
  --saveEngine=<FP16_ENGINE_PATH> \
  --fp16 --memPoolSize=workspace:4096 \
  --precisionConstraints=prefer --avgTiming=1
```

### 4.2 早期基线

早期单 profile、包含数据传输的测试中，Batch=4 的结果为：

| 引擎 | mean latency |
| ---- | -----------: |
| FP16 |   16.7791 ms |
| INT8 |   14.9367 ms |

该口径下 INT8 相比 FP16 只有约 11% 的提升，且不同运行的 GPU compute time 有波动。它不能用于最终 30% 验收，但清楚地说明"仅做量化"不足以达到目标。

## 5. Profile 定位过程

### 5.1 查看 profile 和层信息

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=<INT8_ENGINE_PATH> \
  --dumpOptimizationProfile \
  --dumpLayerInfo \
  --dumpProfile \
  --skipInference
```

只看某个 Batch 的层级耗时时，建议使用实际 shape 并关闭传输：

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=ENGINE \
  --shapes=input:8x3x416x768 \
  --warmUp=1000 --duration=10 \
  --useCudaGraph --useSpinWait --noDataTransfers \
  --dumpProfile --exportProfile=profile_b8.json
```

### 5.2 主要观察

单 profile 日志确认 TensorRT 10.12.0 在 RTX 3090 上构建成功，但 profile 的 `opt=4` 迫使 Batch=1 和 Batch=8 共用一套折中 tactic。层 profile 中 `/model.16/ReduceSum` 在 Batch=8 约占总层耗时 8.6%，`/model.18/ReduceSum` 约占 3.1%，其余同类 ReduceSum 也位于显著耗时层。热点集中在重复的特征融合，而不是只有 Conv 主干。

### 5.3 `/model.42` 排除规则的影响

通过检查量化日志和生成的 Q/DQ 图可确认：

```python
nodes_to_exclude=[r"/model\.42.*"]
```

会排除 `/model.42` 下的 19 个检测头 Conv。也就是说，"排除了一个正则模式"并非只排除一个节点，而是排除了检测头整段节点。全头 INT8 实验显示精度没有恶化，且速度优于旧的排除版本，因此最终采用全头 INT8 配合结构优化；如业务精度对检测头极端敏感，仍可单独回归选择性排除方案。

## 6. 为什么判定五处结构是访存密集

这是本次优化中最需要说明边界的工程判断。

### 6.1 具体结构

原始图包含 5 个目标节点：

```text
/model.16/ReduceSum
/model.18/ReduceSum
/model.21/ReduceSum
/model.24/ReduceSum
/model.27/ReduceSum
```

每处均是严格结构：

```text
多个 Unsqueeze(axis=0)
        -> Concat(axis=0)
        -> ReduceSum(axis=0, keepdims=0)
```

假设每个输入是 `[B,C,H,W]`，`Unsqueeze` 后为 `[1,B,C,H,W]`，`Concat(axis=0)` 形成 `[K,B,C,H,W]` 临时张量，`ReduceSum(axis=0)` 再遍历这个临时张量得到 `[B,C,H,W]`。这条路径对每个元素至少包含写入 stack、再读出 stack 和写回结果的流式内存操作，而有效算术只有 K-1 次加法，算术强度很低。

### 6.2 证据层级

判定依据是三类证据的交集：

1. **结构证据**：显式构造五维 stack，再立即沿 stack 维求和，存在可消除的中间 buffer。
2. **TensorRT profile 证据**：这些 ReduceSum 层在 layer profile 中排名靠前，且随 Batch 增大耗时明显增加。
3. **A/B 证据**：只替换这 5 处结构后，静态 Batch=8 的 INT8 延迟从约 19.83 ms 降至约 14.77 ms，下降约 25.5%；在多 profile 动态引擎中配合 profile 分段后，Batch 1~8 平均提升达到 41.56%。

必须明确：`trtexec --dumpProfile` 只能证明"层耗时热点"，不能单独证明 DRAM 带宽受限；没有 Nsight Compute 的 DRAM throughput、L2 hit rate 等硬件计数时，"访存密集"是基于算子结构、热点行为和替换后 A/B 结果的工程结论，而不是直接的硬件计数结论。

## 7. 图结构优化：Concat/ReduceSum 替换为 Add 链

### 7.1 数学等价性

对输入张量 `x0, x1, ..., x(K-1)`，原始计算为：

```text
ReduceSum(Concat(Unsqueeze(x0), ..., Unsqueeze(x(K-1)), axis=0), axis=0)
= x0 + x1 + ... + x(K-1)
```

因此可直接改写为左结合 Add 链：

```text
t1 = x0 + x1
t2 = t1 + x2
...
out = t(K-1) + x(K-1)
```

改写避免了 `[K,B,C,H,W]` 临时张量。浮点加法结合律并非严格成立，所以采用与导出顺序一致的左结合顺序，而不是任意重排，以最小化舍入差异。

### 7.2 生成脚本

脚本：`quantization/scripts/replace_concat_reducesum.py`。

执行：

```bash
cd <PROJECT_ROOT>
python quantization/scripts/replace_concat_reducesum.py \
  <MODEL_PATH> \
  <FUSED_MODEL_PATH>
```

脚本的匹配条件是保守的：

- 节点必须是 `ReduceSum`；
- 输入生产者必须是 `Concat(axis=0)`；
- `ReduceSum.axes == [0]` 且 `keepdims == 0`；
- Concat 输出只能被该 ReduceSum 消费；
- Concat 的每个输入都必须由 `Unsqueeze(axes=[0])` 产生；
- 每个 Unsqueeze 输出只能被该 Concat 消费；
- 至少有两个源输入。

满足条件后，脚本删除 ReduceSum、Concat 和对应 Unsqueeze，插入同名输出的 Add 链，并执行 `onnx.checker.check_model`。脚本不会改写形状不同、轴不同或存在多消费者的结构。

### 7.3 改写后的结构统计

```text
原图:    1047 nodes, ReduceSum=5, Concat=76, Unsqueeze=23, Add=50
融合图:  1032 nodes, ReduceSum=0, Concat=71, Unsqueeze=3, Add=65
```

五组的输入数分别为 6、5、4、3、2：共移除 5 个 ReduceSum、5 个 Concat 和 20 个 Unsqueeze（合计 30 个节点），再插入 15 个 Add，因此净减少 15 个节点；剩余 3 个 Unsqueeze 属于其他未匹配结构。图输出和拓扑仍通过 ONNX checker。

## 8. 融合图量化和引擎构建

### 8.1 融合图量化

```bash
cd <PROJECT_ROOT>
MODEL="<FUSED_MODEL_PATH>" \
CALIBRATION_DIR="<CALIBRATION_DIR>" \
CALIBRATION_NPY="<CALIBRATION_NPY>" \
OUTPUT_DIR="<FUSED_OUTPUT_DIR>" \
HEIGHT=416 \
WIDTH=768 \
QUANT_SCRIPT="<QUANT_SCRIPT_PATH>" \
bash quantization/scripts/quantize.sh
```

实际日志显示融合图最终量化节点为 365 个，输出 DQ-only ONNX：

```text
<FUSED_INT8_ONNX_PATH>
```

### 8.2 单 profile 对照引擎

融合图的单 profile INT8/FP16 构建命令与原图相同，只需替换 ONNX 和输出路径：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=<FUSED_INT8_ONNX_PATH> \
  --minShapes=input:1x3x416x768 --optShapes=input:4x3x416x768 \
  --maxShapes=input:8x3x416x768 \
  --saveEngine=<FUSED_INT8_ENGINE_PATH> \
  --int8 --fp16 --memPoolSize=workspace:4096 \
  --precisionConstraints=prefer --avgTiming=1
```

早期包含传输的 Batch=4 对照中，融合图约为 FP16 14.3884 ms、INT8 14.5786 ms。这个结果说明图改写本身有效，但单一 profile 和传输/调度开销仍会掩盖 INT8 的收益，所以最终采用分段 profile 和统一无传输口径。

## 9. 动态 Batch 1~8 的三 profile 方案

### 9.1 为什么一个 profile 不够

单 profile 的 `min=1,opt=4,max=8` 要为很宽的 batch 范围选择一组折中 tactic。卷积 tile、融合策略、workspace 分配和 kernel 并行度都可能在 Batch=1 与 Batch=8 之间发生变化。结果是某些 batch 使用了不适合自身规模的 tactic，INT8 相对 FP16 的收益被削弱。

### 9.2 最终 profile 划分

最终验证的 INT8 engine 使用三个连续 profile：

| profile index | min | opt | max | 覆盖 Batch |
| ------------: | --: | --: | --: | ---------- |
|             0 |   1 |   1 |   2 | 1~2        |
|             1 |   3 |   4 |   5 | 3~5        |
|             2 |   6 |   8 |   8 | 6~8        |

TensorRT 10.12 的 `trtexec --help` 明确说明：`--profile` 可以重复指定，以连续编号创建多个 profile。构建时使用如下命令模式（将 `MODEL_INT8` 和 `ENGINE_3P` 替换为实际文件）：

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx="${MODEL_INT8}" \
  --profile=0 \
  --minShapes=input:1x3x416x768 \
  --optShapes=input:1x3x416x768 \
  --maxShapes=input:2x3x416x768 \
  --profile=1 \
  --minShapes=input:3x3x416x768 \
  --optShapes=input:4x3x416x768 \
  --maxShapes=input:5x3x416x768 \
  --profile=2 \
  --minShapes=input:6x3x416x768 \
  --optShapes=input:8x3x416x768 \
  --maxShapes=input:8x3x416x768 \
  --int8 --fp16 \
  --memPoolSize=workspace:4096 \
  --precisionConstraints=prefer \
  --avgTiming=1 \
  --saveEngine="${ENGINE_3P}"
```

构建完成后检查 profile：

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine="${ENGINE_3P}" \
  --dumpOptimizationProfile \
  --skipInference
```

运行时可用 `--useProfile=0/1/2` 显式选择 profile；如果应用通过 TensorRT API 设置动态输入 shape，必须选择覆盖该 shape 的 profile，并在 `setInputShape` 后再 enqueue。Batch=1、2 使用 profile 0，3、4、5 使用 profile 1，6、7、8 使用 profile 2。

## 10. 最终性能结果

以下结果来自同一目标机器、同一输入尺寸、同一 `trtexec` 统一口径（warmup=1000、duration=10、CUDA Graph、SpinWait、无数据传输）：

| Batch | FP16 mean latency (ms) | INT8 mean latency (ms) |     提升 |
| ----- | ---------------------: | ---------------------: | -------: |
| 1     |                5.63842 |                2.87530 |   49.01% |
| 2     |                8.13387 |                4.98621 |   38.70% |
| 3     |               11.17930 |                6.95336 |   37.80% |
| 4     |               14.00000 |                8.48062 |   39.42% |
| 5     |               19.30190 |               11.37610 |   41.06% |
| 6     |               22.02110 |               12.87010 |   41.56% |
| 7     |               25.40730 |               14.70740 |   42.11% |
| 8     |               28.38070 |               16.23750 |   42.79% |
| 平均   |                      - |                      - | **41.56%** |

因此在最终统一口径下，Batch 1~8 每个点都超过 30%，平均提升 41.56%。静态 Batch=8 的局部 A/B 也从全头 INT8 约 19.83 ms 降到 Add 图约 14.77 ms，说明图改写确实消除了主要融合开销。

## 11. 精度和等价性验证

### 11.1 FP32 图改写等价性

在相同随机输入和相同 FP32 执行条件下，对原始 FP32 图和融合后的 FP32 图逐元素比较：

```text
max_abs = 0
mean_abs = 0
exact_fraction = 1.0
```

这验证了当前严格匹配规则和左结合 Add 链没有改变 FP32 输出。

### 11.2 INT8 输出误差

使用校准输入的比较结果：

```text
mean_abs ~= 0.105151
p99       ~= 2.493744
max_abs   ~= 161.361908
relative mean ~= 2.2877%
```

使用随机输入的结果：

```text
mean_abs ~= 0.120409
p99       ~= 2.876896
max_abs   ~= 75.087585
relative mean ~= 2.4636%
```

`max_abs` 受检测输出中的大幅值元素影响，不能单独作为精度结论；应结合真实验证集上的检测 mAP、召回率、置信度阈值和 NMS 后结果。当前数值表明图改写没有引入新的 FP32 误差，INT8 误差主要来自量化本身。

## 12. 历史实验和失败尝试

| 尝试                                  | 结论                                                                                       |
| ------------------------------------- | ------------------------------------------------------------------------------------------ |
| 原始动态 INT8，单 profile             | Batch=8 约 21.9~22.1 ms，提升约 20%~27%，不稳定达到 30%。                                  |
| 仅折叠 Q/DQ 和常量                    | 收益有限，主瓶颈仍在融合和 profile tactic。                                                 |
| 全检测头 INT8                         | 精度未恶化，速度优于旧的 `/model.42` 排除版本。                                             |
| 选择性排除 `cv[23]` 和 `dfl`          | `r"/model\.42/cv[23]\.[012]/.*\.2"`、`r"/model\.42/dfl/.*"`；性能不如全头 INT8 加图优化。   |
| `maxAuxStreams=4`                     | 可以构建，但稳定性不如三 profile engine，因此未作为最终交付方案。                           |
| 只改写 Add/Concat，不分段 profile     | 图优化有效，但动态 Batch 的 tactic 折中仍限制收益。                                         |

## 13. 部署建议和风险

1. 生产环境优先使用三 profile INT8 engine，并按 Batch 范围选择 profile；不要把单 profile `1/4/8` engine 当作最终性能基准。
2. 运行时固定输入空间维度 `3x416x768`，只动态改变 Batch。
3. 目标 GPU、TensorRT 主版本、CUDA/cuDNN 版本变化后必须重新构建和回归；engine 不保证跨 GPU 可移植。
4. 首次部署前使用真实图片验证 mAP、召回率、分类/回归误差和 NMS 结果。本文的数值误差检查不是完整检测精度评估。
5. 校准集只有 1 张图片，建议补充覆盖白天/夜间、远近目标、遮挡和不同场景的代表性样本，再重新量化并复测。
6. 若需要证明访存瓶颈的硬件层面原因，应补充 Nsight Systems/Compute，记录 DRAM throughput、L2 hit rate、kernel occupancy、launch 次数和 workspace 峰值；当前结论已经足以支持工程优化，但不替代硬件计数分析。
