# 与 Tan 和 FPGA 眼动追踪工作的硬件性能比较

核读日期：2026-10-16。每行给出处理范围、原文数据与换算方式。µJ 是微焦耳，mJ 是毫焦耳；1 mJ = 1,000 µJ。

## 速度与能量

| 工作 | 平台、输出 | 处理时间 | 吞吐或更新率 | 功率 | 每次能量 |
|---|---|---:|---:|---:|---:|
| 当前实现 | XCZU3EG；400 事件→连续屏幕坐标 | 0.650 ms，核心周期换算 | 约 1,538 次/s，串行换算 | 1.835 W，Vivado 估算 | 1.193 mJ，估算功率×核心时间 |
| Tan 2025 | UMC 40 nm ASIC；连续屏幕坐标 | 原文小于 1 ms；1252 FPS 对应间隔 0.799 ms | 原文 1,252 FPS | 原文 15.2 mW | 原文 12.7 µJ |
| SEE-D 2024 | ZCU102 FPGA＋Arm；眼中心 | 原文 0.70 ms，SCNN 0.59＋GRU/FC 0.11 | 约 1,429/s，由总时间换算 | 原文 3.86 W | 原文 2.29 mJ |
| SEE-C 2024 | 同一论文的较快模型；眼中心 | 原文 0.60 ms，0.49＋0.11 | 约 1,667/s，由总时间换算 | 原文 3.86 W | 原文 1.88 mJ |
| Khaki 2023 | PYNQ-Z1；瞳孔中心、角膜反光点及瞳孔尺寸 | 原文 1.8 ms，含 PS 传输 | 原文最高 500 FPS | 原文 247 mW，控制面板数值 | 494 µJ，由功率/500 FPS 换算 |
| Kowalski 2021 | Kintex-7 325T＋PC CPU；瞳孔中心 | 原文 0.6～2.1 ms，含相机读出、传输和 CPU 拟合 | 原文演示 575、1,250、5,400 FPS 三种配置 | 原文未报告电路功率 | 原文未报告 |

Tan 数据来自芯片测试章节的 Fig.13、Fig.14 和 Table III；SEE 数据来自 Table 2；Khaki 数据来自 IV-B；Kowalski 数据来自 Table 3 与人眼演示。各项原文见下方来源。

## 资源和时钟

| 工作 | 时钟 | LUT | FF | DSP/计算单元 | 片上存储 |
|---|---:|---:|---:|---:|---|
| 当前实现 | 核心时序通过 160 MHz | 61,657 | 24,835 | 308 DSP | 124 RAMB36＋54 RAMB18，合计 151 RAMB36 等效 |
| Tan | 200 MHz | — | — | 144 MAC | 252.19 KB SRAM；核心面积 2.52 mm² |
| SEE-D | 该文未列时钟 | 约 130K | 约 90K | 1,606 DSP | BRAM 1,092，原文计数单位 |
| SEE-C | 该文未列时钟 | 约 135K | 约 94K | 1,880 DSP | BRAM 1,001，原文计数单位 |
| Khaki | 100 MHz | 9,584 | 11,750 | 11 DSP | 40 BRAM；Table 1 的器件总量为 140 |
| Kowalski | 处理 100 MHz，像素重排 110 MHz | 未列 | 未列 | 未列 | FPGA DRAM 和局部缓冲，未列总量 |

BRAM 的比较保留论文使用的计数方式。SEE 表中未说明 18/36 Kb 单位，当前实现显式给出两类物理块数量。

## 精度与处理对象

| 工作 | 输入与评价对象 | 精度 |
|---|---|---|
| 当前 user17 候选 | 400 事件，2×96×128，Pursuit 单用户 test | 定点平均角误差 0.488592° |
| 当前正式多用户结果 | Pursuit/Saccade 各 24 用户，逐用户训练 | 定点等权均值 0.825481°/1.326037° |
| Tan | 500 事件窗、每 200 新事件触发；2×120×220；EBV-Eye | 正文跨用户平均 1.15°，最佳 0.91° |
| SEE-D / SEE-C | AIS2024 事件眼中心定位 | 平均像素距离 3.71 / 4.05 px |
| Khaki | 176×120 灰度眼图；与软件结果对照 | 横/纵最大偏差 ±1 px |
| Kowalski | 灰度瞳孔图，三种 ROI/相机配置 | 模型瞳孔中心精密度 0.5～1.5 µm |

角度误差、像素定位误差和模型瞳孔精密度分别衡量屏幕注视估计、图像位置及位置重复性；表中保留原评价量。

## 怎样读这些数字

1. **速度位置。** 当前核心的 0.650 ms 位于 SEE-C 的 0.60 ms 和 SEE-D 的 0.70 ms 之间，属于亚毫秒计算。SEE 包含 Arm 上的后续网络；当前核心的采集、排序和板级传输需在系统集成后补上时间。Tan 给出包含 ROI 与推理的芯片处理。
2. **能量位置。** 当前 1.193 mJ 是用工具功率乘核心时间得到的估算，约为 Tan 12.7 µJ 的 94 倍。这个估算显示，能量是当前需要重点改进的指标。相对于 SEE，当前估算值处于较低的毫焦耳量级；板上测量将给出正式结果。
3. **资源位置。** 当前使用约 SEE-D 的 47.4% LUT、19.2% DSP。Khaki 的传统形态学处理仅用 9,584 LUT、11 DSP，模型任务和计算量与神经网络回归不同。
4. **完成阶段。** 当前证据为完整 RTL 回放和器件核心布线。Tan 已流片并测试，三篇 FPGA 工作已在系统上运行。接下来需要完成当前设计的板级接口与实际输入回放。

### 能量统计的具体处理

SEE-D 的 2.29 mJ 接近 3.86 W × 0.59 ms；按总网络 0.70 ms 乘相同功率，算得 2.70 mJ。SEE-C 也有相同现象。这里保留论文能量列，并给出这个算术关系，正式同范围比较时应统一 SCNN 或完整网络的时间。

Khaki 的 494 µJ 使用 247 mW / 500 FPS 换算；用其 1.8 ms 延迟相乘得到 444.6 µJ。前者表示按报告帧率持续处理时的平均能量分摊。

Tan 的正文 Fig.17 前明确报告全用户平均 1.15°、最佳 0.91°，摘要和 Table III 使用 0.91°。本表按正文分别列出平均与最佳。Fig.13 给出 15.2 mW 和 12.7 µJ，按论文原数值保留。

## 原始来源

- Tan et al., *Toward Efficient Eye Tracking in AR/VR Devices: A Near-Eye DVS-Based Processor for Real-Time Gaze Estimation*, IEEE TCSI 2025。[DOI](https://doi.org/10.1109/TCSI.2025.3553497)。核读正式 PDF 印刷页 5757 的 Fig.13/14、V-C，以及页 5758 的 Table III；本仓库提供出处，全文留在原项目文献目录。
- Zhang et al., *Co-designing a Sub-millisecond Latency Event-based Eye Tracking System with Submanifold Sparse CNN*, CVPRW 2024。[作者全文与 Table 2](https://arxiv.org/html/2404.14279v1#S4.T2)。
- Khaki et al., *A Resource-Efficient Multi-Function Embedded Eye Tracker System Implemented on FPGA*, IEEE Access 2023。[作者机构正式全文](https://scholar.gist.ac.kr/bitstream/local/9829/2/43745.pdf)，[DOI](https://doi.org/10.1109/ACCESS.2023.3342839)。资源和功率取印刷页 142936。
- Kowalski et al., *Hybrid FPGA-CPU pupil tracker*, Biomedical Optics Express 2021。[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8548015/)，[DOI](https://doi.org/10.1364/BOE.433766)。相机、处理时钟取 2.4/2.5，延迟取 Table 3。
- 本项目：[整网回放](../../evidence/current/rtl_replay/report.json)、[资源报告](../../evidence/current/vivado/reports/post_route_utilization.rpt)、[功率估算](../../evidence/current/vivado/reports/post_route_power_vectorless_160mhz.rpt)。
