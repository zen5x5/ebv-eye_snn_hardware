# EBV-Eye SNN Hardware

## 1. 网络结构

![400 事件输入、八层卷积、两层脉冲全连接与连续坐标读出的网络结构](docs/hardware/figures/network_architecture.svg)

本项目用脉冲神经网络（SNN）把眼部事件直接转换为连续屏幕注视坐标。每个样本包含 400 个事件，空间表示为 2×96×128；网络共有 96,792 个参数。

同一样本用于两个逻辑时间步 t0、t1。L1/L2 的脉冲输出供两步共用，L3 至 H2 保存第一步的膜电位，参与第二步计算。最后对 H2 两步输出取平均，再通过 12→2 线性层得到坐标。每个新样本开始时重置状态。

图中尺寸依次为“通道数×高×宽”。归一化参数在硬件推理前合入权重与常量。逐层计算、数据存储与计算单元安排见[当前硬件设计](docs/hardware/current_design.md)。

## 2. 精度：软件浮点与硬件定点对比

以下均为 test 集的平均角误差，越小越好。浮点列使用训练后的软件模型；定点列使用与 RTL 电路一致的整数运算规则，权重为 16 位。

| 评价范围 | 软件浮点 SNN | 16 位权重定点 SNN | 定点相对浮点变化 |
|---|---:|---:|---:|
| 当前硬件候选：Pursuit user17 / seed2022，1,364 个样本 | 0.489147° | 0.488592° | -0.000555° |
| Pursuit：24 用户，seed2021，25,716 个样本 | 0.823945° | 0.825481° | +0.001536° |
| Saccade：24 用户，seed2021，28,389 个样本 | 1.325509° | 1.326037° | +0.000528° |

每行的浮点、定点使用相同模型和相同 test 样本。多用户结果先逐用户计算均值，再对 24 用户等权平均；Pursuit 是平滑追踪，Saccade 是快速跳转注视。当前 FPGA 验证选用第一行模型。

定点后的精度变化很小。整网 RTL 功能仿真进一步核对了一个真实样本的 159,408 个中间值和输出值，零失配。

来源：[当前候选量化结果](evidence/current/quantization/pursuit/user17/summary.json)、[48 模型量化汇总](evidence/formal_quantization/results.json)、[量化方法](docs/hardware/06_formal_l40s_quantization.md)、[整网仿真记录](evidence/current/rtl_replay/export_report.json)。

## 3. FPGA 实现与性能

目标板卡为 AUP-ZU3，器件为 XCZU3EG。当前已完成 RTL 电路功能仿真和加速器核心布局布线；板级接口、上板推理及功率测量待完成。硬件证据日期为 2026-10-16，首页整理日期为 2026-10-17。

### 处理速度与能量

| 指标 | 当前结果 | 计算或验证依据 |
|---|---:|---|
| 单样本周期 | 104,005 拍 | 完整 RTL 功能仿真 |
| 核心时钟 | 160 MHz | 布线后核心时序通过 |
| 单样本核心处理时间 | 0.650 ms | 104,005 拍 / 160 MHz |
| 连续串行处理能力 | 约 1,538 样本/秒 | 160 MHz / 104,005 拍 |
| 片上功率 | 1.835 W，工具估算 | Vivado 默认信号活动估计 |
| 单样本能量 | 1.193 mJ，工具估算 | 估算功率 × 核心处理时间 |

时间覆盖权重已装载后的加速器核心；相机采集、事件成窗与排序、处理器传输的时间在板级集成时统计。功率和能量目前采用工具估算值。

### 器件资源占用

| 资源 | 本设计使用 | XCZU3EG 总量 | 占用比例 |
|---|---:|---:|---:|
| LUT：逻辑运算单元 | 61,657 | 70,560 | 87.38% |
| FF：单比特寄存器 | 24,835 | 141,120 | 17.60% |
| DSP：专用算术单元 | 308 | 360 | 85.56% |
| BRAM：片上块存储，按 36 Kb 等效计 | 151 | 216 | 69.91% |

BRAM 实际使用 124 个 36 Kb 块和 54 个 18 Kb 块，折合 124＋54/2＝151 个 36 Kb 块。LUT 包括用于逻辑、分布式存储和移位寄存器的全部占用。

来源：[布线后资源](evidence/current/vivado/reports/post_route_utilization.rpt)、[器件容量与功率估算](evidence/current/vivado/reports/post_route_power_vectorless_160mhz.rpt)、[160 MHz 时序](evidence/current/vivado/reports/post_route_timing_160mhz.rpt)。

## 4. 与其他眼动追踪工作的性能对比

### FPGA 实现

| 工作与平台 | 输出 | 处理时间 | 吞吐或更新率 | 功率 | 每次能量 |
|---|---|---:|---:|---:|---:|
| 本项目，XCZU3EG | 连续屏幕坐标 | 0.650 ms，核心周期换算 | 约 1,538 次/秒，换算 | 1.835 W，工具估算 | 1.193 mJ，估算 |
| [SEE-D，2024，ZCU102＋Arm](https://arxiv.org/html/2404.14279v1#S4.T2) | 眼中心 | 0.70 ms | 约 1,429 次/秒，换算 | 3.86 W | 原文 2.29 mJ |
| [SEE-C，2024，同一论文](https://arxiv.org/html/2404.14279v1#S4.T2) | 眼中心 | 0.60 ms | 约 1,667 次/秒，换算 | 3.86 W | 原文 1.88 mJ |
| [Khaki，2023，PYNQ-Z1](https://doi.org/10.1109/ACCESS.2023.3342839) | 瞳孔中心、反光点、瞳孔尺寸 | 1.8 ms，含处理器传输 | 最高 500 FPS | 247 mW，工具面板报告 | 494 µJ，功率/帧率换算 |
| [Kowalski，2021，Kintex-7＋PC](https://doi.org/10.1364/BOE.433766) | 瞳孔中心 | 0.6～2.1 ms，含相机与 CPU | 三种配置为 575 / 1,250 / 5,400 FPS | 未报告 | 未报告 |

SEE-C/D 使用稀疏卷积网络；其延迟包含 Arm 上的后续网络。原文能量列与 SCNN 部分的计算时间对应，完整统计关系见[对比报告](docs/hardware/comparison.md)。

### 专用芯片与 SNN 硬件

ASIC 表示专门制造的芯片。下表包含制造后的芯片测试、可穿戴原型和芯片设计阶段的工具评估。

| 工作与平台 | 输出 | 处理时间或更新率 | 功率 | 每次能量 | 验证阶段 |
|---|---|---|---:|---:|---|
| [Tan，2025，40 nm ASIC](https://doi.org/10.1109/TCSI.2025.3553497) | 连续屏幕坐标 | 处理小于 1 ms；1,252 FPS | 15.2 mW | 原文 12.7 µJ | 芯片实测，常规卷积网络 |
| [NeuroEye，2025，40 nm ASIC](https://doi.org/10.1109/ISCAS56072.2025.11043272) | SNN 眼动计算 | 12,200 FPS | 54.59 mW | 4.47 µJ | 芯片布局布线后评估 |
| [Retina，2024，Speck](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/html/Bonazzi_Retina__Low-Power_Eye_Tracking_with_Event_Camera_and_Spiking_CVPRW_2024_paper.html) | 瞳孔位置 | 固定/动态窗延迟 5.57 / 8.01 ms | 2.89 / 4.80 mW | 单位核对见下文 | 芯片上传感器与 SNN 处理器实测 |
| [Paredes-Valles，2025，双 Speck2f＋微控制器](https://arxiv.org/html/2511.20175v1) | 双眼瞳孔中心 | 100 Hz 更新 | 每颗 Speck2f <5 mW；微控制器另约 13.6 mW | 未列统一系统能量 | 可穿戴原型实测 |

Retina 原表能量单位与功率、延迟相差 1,000 倍；按其功率×延迟换算，固定/动态窗分别为 16.10 / 38.45 µJ。该测试以 100 ms 间隔隔离输入响应。NeuroEye 的 12,200 FPS 表示吞吐量，摘要未给单次输入到输出延迟。

从这些结果看，当前核心已达到亚毫秒处理；LUT 和 DSP 占用较高，能量优化是后续重点。详细的精度、资源、处理范围和原文核读记录见[完整对比报告](docs/hardware/comparison.md)。

<details>
<summary>更多 FPGA 实现与 SNN 研究：其余 9 篇</summary>

| 工作 | 内容与已报告结果 |
|---|---|
| Ando 2012，Cyclone II | 传统人眼定位；约 8 FPS、118.9 ms；Quartus 估算约 375 mW |
| Yang 2018，Kintex-7 | 视线追踪电路原型；200 MHz、超过 380 Hz，摘要报告误差不超过 1.5° |
| Yu 2021，单 FPGA | 640×480、80 FPS 的视线方向分类；分类模块 0.52 µs，识别率 92% |
| Wu、Han 2024，Zynq-7035 | 人脸、头部姿态与眼图的视线估计；32 FPS，原文平均功率 6.4 W |
| Jiang 2024，SNN | RTX 3090 软件实验；专用硬件部署列为后续工作 |
| GazeSCRNN 2025，SNN | EV-Eye 视线方向预测；平均角误差 6.034°，未报告部署后的硬件性能 |
| Hueber 2025，SNN | 瞳孔中心预测；按 SENeCA 模型预测 3.9～4.9 mW、约 3 ms 延迟、1 kHz 更新 |
| Sui 2026，SNN＋位置更新 | GPU/CPU 算法时间 3.35 ms；按另一套硬件成本估算系统功率 102.2 mW |
| E-SpikePT 2026，ANN/SNN 混合 | 瞳孔中心预测；49 µJ 为运算能量估算，未计存储访问、数据搬运和脉冲路由 |

全部 16 篇外部工作的论文链接、网络类型与核读范围集中在[对比报告](docs/hardware/comparison.md)。SEE-C/D 计为同一篇论文。

</details>

## 5. 硬件如何执行这个网络

![L1/L2 前端、共享深层引擎、全连接与坐标读出的硬件数据通路](docs/hardware/figures/current_dataflow.svg)

L1/L2 边接收事件边计算；L3～L8 共用卷积引擎；H1/H2 共用全连接引擎。权重和中间数据保存在 FPGA 片上，各层按顺序使用计算单元。

- 计算、寄存器和存储细节：[当前设计与逐层数据流](docs/hardware/current_design.md)。
- 数值、仿真和工具报告：[机器证据](evidence/README.md)、[验证记录](VALIDATION.md)。
- 设计演变：[早期实现](docs/history/pre_vivado_20261013.md)、[Vivado 优化过程](docs/history/vivado_development_20261015.md)。

## 6. 复现当前整网

<details>
<summary>展开复现命令与实现入口</summary>

在仓库根目录，启用 Vivado 2025.2.1 后运行：

```bash
source /opt/amd/2025.2.1/Vivado/settings64.sh
python -B scripts/verify_evidence.py
python -B scripts/replay_current.py --output-dir build/replay
```

回放入口只使用 Python 标准库和 Vivado 自带的 xvlog、xelab、xsim。它检查随仓库提供的真实样本、参数和逐层标准结果，预期得到 104,005 拍、159,408 个值零失配。输出目录使用新的空路径。

重新执行核心综合、布局和布线：

```bash
bash code/hardware/vivado/run_sparse_ooc.sh build/route160 sparse_lm_a
```

该入口默认 6.250 ns，即 160 MHz。详细参数见 [Vivado 入口](code/hardware/vivado/README.md)。已归档的时序证据来自 200 MHz 目标下布线后，在相同布线结果上覆盖为 160 MHz 的时序复核。

检查仓库文件：

```bash
sha256sum -c MANIFEST.sha256
```

</details>

## 7. 仓库目录

<details>
<summary>展开源码、证据和报告目录</summary>

| 位置 | 内容 |
|---|---|
| code/hardware/rtl/ | 当前硬件描述、单元测试和历史基线 |
| code/hardware/vivado/ | 器件相关综合、实现与时序脚本 |
| code/hardware/*.py | 定点参考和数值验证工具 |
| code/pursuit/ | 与硬件语义对应的模型代码 |
| scripts/ | 当前 XSim 回放和资产检查 |
| evidence/current/ | user17 标准向量、权重、量化、仿真、布线报告 |
| evidence/formal_quantization/ | 48 模型正式量化结果 |
| evidence/rtl_three_route/ | user15 早期三路线对照 |
| docs/hardware/ | 当前设计和外部比较 |
| docs/history/ | 两篇保留的历史阶段记录 |

早期 user15 完整版本可从 Git 提交 3fef30e 恢复。

</details>
