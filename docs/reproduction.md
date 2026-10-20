# 独立使用与结果复现

仓库目录下的命令只读本仓库模型、输入和结果，不依赖相邻软件项目。
所有新计算写入显式指定的runs/子目录，已确认的results/保留原始记录。

## 论文取数

Python标准库即可执行，完整检查不运行仿真：
```bash
python scripts/verify.py
python scripts/verify.py --full
python scripts/report.py --output runs/paper_metrics.csv
```

完整检查会解压读取SAIF信号活动和报告，并验证原始SHA-256。
已有CSV直接位于results/paper_metrics.csv。

## 数值参考与RTL回放

使用Python 3.11及requirements.txt依赖；仅回放和整数单元测试需要NumPy，
模型量化另需PyTorch和Sinabs。CPU模型加载使用本仓库hardware/quantization/model.py，
与软件侧模型定义保持相同数学实现。

```bash
python -m unittest discover -s tests -v
source /opt/amd/2025.2.1/Vivado/settings64.sh
python scripts/replay.py --variant l1_l2_stateless --samples 0 --output-dir runs/rtl_l1_l2
python scripts/quantize.py --variant l1_l2_stateless --output-dir runs/quant_l1_l2
python -m hardware.quantization.pack \
  --parameters models/l1_l2_stateless/parameters.npz --output-dir runs/weight_image
```

replay默认覆盖16输入；--samples可以指定已有验证索引0～15。
NPZ参考数组生成输入与逐层.mem文件时逐个检查原始校验值，不进行模型推理。
整理后的运行核验记录见[verification.md](verification.md)。

## 四档双任务24用户量化

确认后的192模型结果在`results/quantization/state_depth/`，含输入快照及逐人结果。
以下命令只核对保存的预测、输入校验值和汇总，不重新推理：

```bash
python scripts/verify_state_depth_quantization.py results/quantization/state_depth
```

完整重新计算141份原待补模型时，使用Python 3.11和本仓库依赖，输出到新的runs目录。
已有L1～L2双任务48份及Pursuit user10其余3份按源文件校验值复用。
`--models-root`指定原始训练结果根，`--cache-root`指定正式eye0缓存根；准备阶段保存独立输入副本。

```bash
python scripts/quantize_state_depth.py prepare \
  --models-root /data/xuzx/result/ebv-eye \
  --cache-root /data/xuzx/data/ebv-eye/cache/formal_eye0 \
  --output-dir runs/state_depth_quantization
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
  python scripts/quantize_state_depth.py evaluate \
  --output-dir runs/state_depth_quantization --workers 8 --threads 2
python scripts/quantize_state_depth.py summarize --output-dir runs/state_depth_quantization
python scripts/verify_state_depth_quantization.py runs/state_depth_quantization --check-original-inputs
```

2026-10-19的实际执行位置为carbonite CPU。每份新模型先检查一个样本的61个逐层整数参考数组，
再计算完整test；浮点指标保留原L40S/Exodus训练报告，Saccade采用连续xy读出。

### Pursuit L1～L4无状态的24用户量化

完整确认结果在`results/quantization/l1_l4/`。只核验已保存结果：

```bash
python scripts/verify_state_depth_quantization.py results/quantization/l1_l4
```

重新计算时显式选择Pursuit和第五档，其余数值规则不变：

```bash
python scripts/quantize_state_depth.py prepare \
  --tasks pursuit --variants l1_l2_l3_l4_stateless \
  --models-root /data/xuzx/result/ebv-eye \
  --cache-root /data/xuzx/data/ebv-eye/cache/formal_eye0 \
  --output-dir runs/pursuit_l1_l4_quantization
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
  python scripts/quantize_state_depth.py evaluate \
  --output-dir runs/pursuit_l1_l4_quantization --workers 8 --threads 2
python scripts/quantize_state_depth.py summarize --output-dir runs/pursuit_l1_l4_quantization
python scripts/verify_state_depth_quantization.py runs/pursuit_l1_l4_quantization --check-original-inputs
```

### Saccade旧连续输出头L1～L4无状态的24用户量化

确认后的完整结果在`results/quantization/saccade_l1_l4/`，可直接核验：

```bash
python scripts/verify_state_depth_quantization.py results/quantization/saccade_l1_l4
```

选择相同第五档和Saccade任务，使用原连续xy输出头：

```bash
python scripts/quantize_state_depth.py prepare \
  --tasks saccade --variants l1_l2_l3_l4_stateless \
  --models-root /data/xuzx/result/ebv-eye \
  --cache-root /data/xuzx/data/ebv-eye/cache/formal_eye0 \
  --output-dir runs/saccade_l1_l4_quantization
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
  python scripts/quantize_state_depth.py evaluate \
  --output-dir runs/saccade_l1_l4_quantization --workers 8 --threads 2
python scripts/quantize_state_depth.py summarize --output-dir runs/saccade_l1_l4_quantization
python scripts/verify_state_depth_quantization.py runs/saccade_l1_l4_quantization --check-original-inputs
```

## 综合和布局布线

```bash
mkdir -p runs/synthesis/full_state
vivado -mode batch -nojournal -log runs/synthesis/full_state/vivado.log \
  -source hardware/vivado/synthesize.tcl -tclargs runs/synthesis/full_state 0
mkdir -p runs/implementation/full_state
vivado -mode batch -nojournal -log runs/implementation/full_state/vivado.log \
  -source hardware/vivado/implement.tcl \
  -tclargs runs/synthesis/full_state runs/implementation/full_state
vivado -mode batch -nojournal -source hardware/vivado/audit.tcl \
  -tclargs runs/implementation/full_state
```

深度参数0/1/2/3/4对应全状态、L1、L1～L2、L1～L3、L1～L4；每组使用自己的输出目录。
已有最终DCP和全部原报告在results/，写论文无需执行这些步骤。

新增第五档同样可以直接重放，不需要重新量化或创建参考输入：

```bash
python scripts/replay.py --variant l1_l2_l3_l4_stateless --samples 0 --output-dir runs/rtl_l1_l4
```

原四档测量使用的RTL原文按校验值保存在`results/provenance/measured_sources.zip`及
`four_depth_sources.zip`中；当前参数化实现增加深度4。旧DCP、原报告与旧测量哈希保持不变，
核验分别对照旧源码快照和新增实现源码。

## 两任务核心时间与存储核验

以下只核对保存的输入/参数布局、192模型配置和64次Saccade回放记录，不重新推理或仿真：

```bash
python scripts/verify_task_equivalence.py --cases results/task_equivalence/cases \
  --replay results/task_equivalence/replay.json --output runs/task_equivalence_check.json
```

若需重放某档Saccade，先复制其完整参考文件并解压mem，再运行同一个测试台：

```bash
cp -a results/task_equivalence/cases/saccade_l1_l2_stateless runs/saccade_l1_l2_case
tar -xzf runs/saccade_l1_l2_case/sample_memories.tar.gz -C runs/saccade_l1_l2_case
python scripts/replay_cases.py --cases runs/saccade_l1_l2_case --output-dir runs/saccade_rtl_check
```

## 功耗与活动

### 五档默认活动估算

直接读取五份既有布线检查点，恢复Vivado默认活动设置；不运行样本仿真、综合或布线。
计算时钟160 MHz、结温25°C、典型工艺、标称电压。普通输入默认高电平概率0.5、
每周期翻转率12.5%；复位由工具默认置为无效。

```bash
source /opt/amd/2025.2.1/Vivado/settings64.sh
python scripts/power_vectorless.py --output-dir runs/vectorless_power_check --workers 2
python scripts/verify_vectorless_power.py runs/vectorless_power_check
```

已确认结果位于`results/power_vectorless/`。核对已保存的结果，无需启动Vivado：

```bash
python scripts/verify_vectorless_power.py results/power_vectorless
```

同一深度的Pursuit/Saccade连续坐标网络共用一份运行时加载权重的电路，
默认活动分析使用同一份报告。原始文本/XML、活动来源、操作条件、输入检查点校验值、
执行日志及当时的脚本均保存。工具的Low可信度标记与内部控制信号活动提示一同保留。

### 已有真实输入活动估算

直接使用已保存的活动重算功耗：
```bash
python scripts/power.py --variant l1_l2_stateless --output-dir runs/power_l1_l2
```

只有需要重新记录波形时才执行：
```bash
python scripts/activity.py --variant l1_l2_stateless --samples 0 \
  --output-dir runs/activity_l1_l2
```

该命令使用已保存的布线后功能电路，不包含逐线SDF延时。结果统计条件为
160 MHz、25°C结温、典型工艺与标称电压，覆盖推理窗口。
