# FPGA、专用芯片与 SNN 眼动追踪性能比较

更新日期：2026-10-17。当前实现的硬件结果沿用 2026-10-16 已归档证据；本次更新扩展外部文献、核对性能数字及其统计范围。

本报告收录 16 篇独立外部工作：7 篇含 FPGA 实现或原型，8 篇 SNN 眼动研究，另列 Tan 的专用芯片。SEE-C 和 SEE-D 是同一篇论文的两个模型，计为一篇。这个数量表示本报告的收录范围。

FPGA 是可以重新配置电路的器件；ASIC 是专门制造的芯片。SNN 使用脉冲神经元，ANN 使用常规神经网络运算。事件相机可以配合这两类网络及传统图像算法。SEE 的 SCNN 指稀疏卷积网络；Tan 使用常规卷积网络。

吞吐量表示每秒能完成多少次处理；延迟表示一次输入到对应输出的时间。µJ 是微焦耳，mJ 是毫焦耳；1 mJ = 1,000 µJ。下表分别注明原文数值、算术换算、实测和估算。

## 1. FPGA 实现：速度与能量

| 工作 | 平台、输出 | 处理时间 | 吞吐或更新率 | 功率 | 每次能量 |
|---|---|---:|---:|---:|---:|
| 当前实现 | XCZU3EG；400 事件→连续屏幕坐标 | 0.650 ms，核心周期换算 | 约 1,538 次/s，串行换算 | 1.835 W，Vivado 估算 | 1.193 mJ，估算功率×核心时间 |
| SEE-D 2024 | ZCU102 FPGA＋Arm；眼中心 | 原文 0.70 ms，SCNN 0.59＋GRU/FC 0.11 | 约 1,429/s，由总时间换算 | 原文 3.86 W | 原文 2.29 mJ |
| SEE-C 2024 | 同一论文的较快模型；眼中心 | 原文 0.60 ms，0.49＋0.11 | 约 1,667/s，由总时间换算 | 原文 3.86 W | 原文 1.88 mJ |
| Khaki 2023 | PYNQ-Z1；瞳孔中心、角膜反光点及瞳孔尺寸 | 原文 1.8 ms，含处理器传输 | 原文最高 500 FPS | 原文 247 mW，控制面板数值 | 494 µJ，由功率/500 FPS 换算 |
| Kowalski 2021 | Kintex-7 325T＋PC CPU；瞳孔中心 | 原文 0.6～2.1 ms，含相机读出、传输和 CPU 拟合 | 原文演示 575、1,250、5,400 FPS 三种配置 | 原文未报告电路功率 | 原文未报告 |
| Ando 2012 | Cyclone II 2C70；传统算法定位人眼 | Table 1 总计 118.9 ms，含采集和格式整理 | 原文约 8 FPS | Table 2 为 375.45 mW，Quartus 工具估算 | 本报告不另行换算 |
| Yang 2018 | Kintex-7 上的视线追踪电路原型 | 已核摘要未给输入到输出延迟 | 原文超过 380 Hz | 已核摘要未给 | 已核摘要未给 |
| Yu 2021 | 单 FPGA；人眼特征与视线方向分类 | 原文分类模块识别时间 0.52 µs | 原文 640×480、80 FPS | 已核摘要未给 | 已核摘要未给 |
| Wu、Han 2024 | Zynq-7035＋Arm；结合人脸、头部姿态和眼图估计视线 | 已核摘要未给系统延迟 | 原文 32 FPS | 原文平均 6.4 W，供电测量范围待全文核对 | 本报告不另行换算 |

SEE 数据来自 Table 2；Khaki 数据来自 IV-B；Kowalski 数据来自 Table 3 与人眼演示；Ando 数据来自 3.2、Table 1/2。Yang、Yu、Wu 的本轮新增数值来自论文摘要。各项原文见第 8 节来源。

Yu 的 0.52 µs 覆盖视线方向分类模块，其相机输入为 80 FPS。两项分别记录。Kowalski 的采集与处理可以重叠，表中的延迟和帧率分别来自各配置的实际报告。

## 2. FPGA 实现：资源和时钟

LUT 是实现逻辑运算的小单元；FF 是保存一位数据的寄存器；DSP 是专用算术单元；BRAM 是片上块存储。它们共同描述 FPGA 为这套电路占用了多少器件资源。

| 工作 | 时钟 | LUT | FF | DSP/计算单元 | 片上存储 |
|---|---:|---:|---:|---:|---|
| 当前实现 | 核心时序通过 160 MHz | 61,657 | 24,835 | 308 DSP | 124 RAMB36＋54 RAMB18，合计 151 RAMB36 等效 |
| SEE-D | 该文未列时钟 | 约 130K | 约 90K | 1,606 DSP | BRAM 1,092，原文计数单位 |
| SEE-C | 该文未列时钟 | 约 135K | 约 94K | 1,880 DSP | BRAM 1,001，原文计数单位 |
| Khaki | 100 MHz | 9,584 | 11,750 | 11 DSP | 40 BRAM；Table 1 的器件总量为 140 |
| Kowalski | 处理 100 MHz，像素重排 110 MHz | 未列 | 未列 | 未列 | FPGA DRAM 和局部缓冲，未列总量 |
| Ando | 27 MHz | 原文合计 8,878 logic cells，保留该计数 | 未单列 | 28 个乘法器 | 片上 SRAM 53,184 bit；另有板上 SDRAM/SSRAM |
| Yang | 200 MHz | 已核摘要未给 | 已核摘要未给 | 已核摘要称电路未调用 DSP | 已核摘要称未使用外部存储 |
| Yu | 100 MHz | 已核摘要未给 | 已核摘要未给 | 已核摘要未给 | 已核摘要未给 |
| Wu、Han | 已核摘要未给 | 已核摘要未给 | 已核摘要未给 | 已核摘要未给 | 已核摘要未给 |

BRAM 的比较保留论文使用的计数方式。SEE 表中未说明 18/36 Kb 单位，当前实现显式给出两类物理块数量。Ando 的 logic cells 是另一代器件的资源计数，本表按原名称保留。

## 3. Tan：已制造并测试的专用芯片

| 平台与输出 | 处理时间 | 吞吐量 | 功率 | 每次能量 | 计算与存储 |
|---|---|---:|---:|---:|---|
| UMC 40 nm ASIC；连续屏幕坐标 | 原文小于 1 ms；1,252 FPS 对应输出间隔约 0.799 ms | 1,252 FPS | 15.2 mW | 12.7 µJ | 200 MHz；144 个乘加单元；252.19 KB SRAM；核心面积 2.52 mm² |

数值取芯片测试章节的 Fig.13/14 和 Table III。处理范围包含事件区域选择与网络推理。测试系统使用 ZCU102 连接芯片，视线计算由定制芯片完成。

## 4. SNN 眼动追踪：8 篇独立工作

以下按网络实际输出、运行平台和验证阶段列出。Groenen、Paredes-Valles、Hueber 三项按本次核读的 2025 年 arXiv v1 记录。

| 工作 | SNN 完成什么 | 运行平台与证据 | 延迟、吞吐或更新 | 功率与能量 |
|---|---|---|---|---|
| Retina，Bonazzi 2024 | 事件→瞳孔位置 | Speck 专用脉冲芯片实测 | 固定 3 ms 窗：延迟 5.57 ms；动态 300 事件窗：8.01 ms | 固定/动态分别 2.89/4.80 mW，覆盖芯片上的传感器与处理器；能量单位核对见第 7 节 |
| Jiang 2024 | 脉冲特征提取与眼睛区域追踪 | RTX 3090 软件实验；未部署专用推理硬件 | 原文最细时间分辨率 0.081 ms，表示事件输出时间粒度；硬件推理延迟未测 | 未给专用电路功率和能量 |
| GazeSCRNN，Groenen 2025 | 事件→视线方向 | EV-Eye 上的软件模型实验；专用硬件部署列为后续工作 | 未报告部署后的延迟和吞吐量 | 未报告部署后的功率和能量 |
| NeuroEye，Zheng 2025，ISCAS | 事件输入与 SNN 眼动计算，各层接力处理脉冲 | 40 nm 定制芯片设计，完成布局布线后的评估；本轮核对 IEEE 摘要 | 12,200 FPS；摘要未给单次输入到输出延迟 | 54.59 mW；4.47 µJ/次，均为布局布线后评估 |
| Paredes-Valles 2025 | 连续双眼瞳孔中心预测 | 两颗 Speck2f＋nRF52840 微控制器，可穿戴原型实测 | 100 Hz 坐标更新；正文未给单次计算延迟 | 每颗 Speck2f 平均低于 5 mW；Fig.6 另列微控制器平均约 13.6 mW |
| Hueber 2025 | 将已有眼动网络改为 SNN，预测瞳孔中心 | 软件模型验证；效率按 SENeCA 芯片模型预测 | 预测延迟约 3 ms、1 kHz 更新 | 预测 3.9～4.9 mW |
| Sui 2026，EAAI | SNN 分割瞳孔，配合位置更新与个人校准输出注视点 | GPU/CPU 测算法时间；按 Loihi 单位运算成本及传感器数据估算能耗 | 原文期望处理时间 3.35 ms，约 300 Hz | 系统估算 102.2 mW；一次 SNN 分割估算 0.023 J |
| E-SpikePT，Kang 2026，Neurocomputing | ANN/SNN 混合网络预测瞳孔中心 | 软件实验；用 45 nm 单位运算成本估算能量 | 实际硬件延迟和功率测量列为后续工作 | 0.049 mJ＝49 µJ，为运算能量估算 |

Retina 的测量分解为传感器数字/模拟电源、SNN 逻辑、RAM 和 I/O。Paredes-Valles 的小于 5 mW 是 Speck2f 单颗设备功率，微控制器在原文图中另行统计；本报告分别保留这两部分。

NeuroEye 摘要明确给出跨层脉冲流水处理，适合作为当前设计的重点结构对照。其完整输入规模、网络精度、面积和逐层延迟仍需取得全文核读；本报告先收录已经核对的摘要数字。

本轮核对到的 SNN 硬件主要采用专用脉冲芯片或定制芯片设计。SNN、眼动任务、FPGA 加速电路三者同时满足的已发表实现，本报告尚未核实到可列入的独立外部工作。

## 5. 精度与处理对象

| 工作 | 输入与评价对象 | 精度 |
|---|---|---|
| 当前 user17 候选 | 400 事件，2×96×128，Pursuit 单用户 test | 定点平均角误差 0.488592° |
| 当前正式多用户结果 | Pursuit/Saccade 各 24 用户，逐用户训练 | 定点等权均值 0.825481°/1.326037° |
| Tan | 500 事件窗、每 200 新事件触发；2×120×220；EBV-Eye | 正文跨用户平均 1.15°，最佳 0.91° |
| SEE-D / SEE-C | AIS2024 事件眼中心定位 | 平均像素距离 3.71 / 4.05 px |
| Khaki | 176×120 灰度眼图；与软件结果对照 | 横/纵最大偏差 ±1 px |
| Kowalski | 灰度瞳孔图，三种 ROI/相机配置 | 模型瞳孔中心精密度 0.5～1.5 µm |
| Yang | 面向 VR 的视线追踪原型 | 摘要报告误差不超过 1.5° |
| Yu | 视线方向分类 | 摘要报告平均识别率 92% |
| Retina | Ini-30；64×64 输入；validation | 平均瞳孔位置误差 3.24 px |
| GazeSCRNN | EV-Eye 视线方向预测 | 原文平均角误差 6.034°；训练和评价协议见原文 |
| Hueber | 3ET+ 瞳孔位置预测 | 原文平均像素距离约 3.7～4.1 px |

角度误差、像素定位误差、方向分类准确率和模型瞳孔精密度分别衡量不同输出的质量；表中保留原评价量。user17 的 0.488592°属于单用户候选，多用户结果单独列出。GazeSCRNN、Tan 和当前项目各自的输入、切分和用户汇总方式沿用各自实验定义。

## 6. 当前设计的位置与优先对照

1. **速度位置。** 当前核心的 0.650 ms 位于 SEE-C 的 0.60 ms 和 SEE-D 的 0.70 ms 之间，属于亚毫秒计算。SEE 包含 Arm 上的后续网络；当前核心的采集、排序和板级传输需在系统集成后补上时间。SNN 对照还应加入 Retina 的芯片实测延迟与 NeuroEye 的处理吞吐量。
2. **能量位置。** 当前 1.192807 mJ 是用 Vivado 工具功率乘核心时间得到的估算。对应 Tan 12.7 µJ 和 NeuroEye 4.47 µJ 的数值比约为 94 和 267；三个数字分别来自 FPGA 工具估算、40 nm 芯片测试和 40 nm 芯片布局布线后评估。当前 FPGA 实现的能量量级值得重点优化，板上活动与电源测量将提供当前设计的实测值。
3. **资源位置。** 当前使用约 SEE-D 的 47.4% LUT、19.2% DSP。Khaki 的传统形态学处理仅用 9,584 LUT、11 DSP，模型任务和计算量与神经网络回归不同。
4. **结构对照。** SEE 用于比较 FPGA 上稀疏计算的代价；NeuroEye 用于研究各层之间如何连续传递脉冲；Retina 与 Paredes-Valles 用于研究低功耗 SNN 部署及坐标输出；GazeSCRNN 和 Sui 用于比较视线预测任务与神经元状态的组织。当前实现的浅层流式处理、深层共享计算单元及中间数据保存方式详见[当前设计](current_design.md)。
5. **完成阶段。** 当前证据为完整 RTL 功能回放和器件核心布线。Tan 有制造后的芯片测试，Retina 和 Paredes-Valles 有专用芯片运行，NeuroEye 的摘要结果为布局布线后评估。当前设计下一步是板级接口、实际输入回放与功率测量。

## 7. 性能数字的逐项核对

### 当前实现

104,005 拍 / 160,000,000 拍每秒 = 0.65003125 ms。

1.835 W × 0.65003125 ms = 1.19280734375 mJ。

这是当前样本在权重已经装载完成后经过加速器核心的时间和能量估算。周期来自完整 RTL 回放，频率来自已布线核心时序检查，功率来自缺少真实输入活动记录的 Vivado 估算。原始证据列于第 8 节。

### SEE、Khaki 与 Tan

SEE-D 的 2.29 mJ 接近 3.86 W × 0.59 ms；按总网络 0.70 ms 乘相同功率，算得 2.70 mJ。SEE-C 也有相同现象。这里保留论文能量列，并给出这个算术关系，正式同范围比较时应统一 SCNN 或完整网络的时间。

Khaki 的 494 µJ 使用 247 mW / 500 FPS 换算；用其 1.8 ms 延迟相乘得到 444.6 µJ。前者表示按报告帧率持续处理时的平均能量分摊。

Tan 的正文 Fig.17 前明确报告全用户平均 1.15°、最佳 0.91°，摘要和 Table III 使用 0.91°。本表按正文分别列出平均与最佳。Fig.13 给出 15.2 mW 和 12.7 µJ，按论文原数值保留。

### Retina：测量范围与能量单位

原文 §5.2.4 将输入事件窗间隔设为 100 ms，以分离芯片响应。Table 9 给出固定/动态窗功率 2.89/4.80 mW、延迟 5.57/8.01 ms。测量范围覆盖 Speck 芯片上的传感器与处理器各电源项；主机和开发板总供电未列入这些分项。

同表能量列标成 mJ，数值为 16.10/38.40；功率乘延迟的单位换算为：

- 固定窗：2.89 mW × 5.57 ms = 16.0973 µJ。
- 动态窗：4.80 mW × 8.01 ms = 38.448 µJ。

原表能量单位与同表功率、延迟相差 1,000 倍。本报告写出原始数值与算术核对结果，能量比较使用时保留其隔离输入测量条件。

### NeuroEye：吞吐量与单次延迟

54.59 mW / 12,200 次每秒 ≈ 4.475 µJ/次，与摘要的 4.47 µJ/次一致。12,200 FPS 对应相邻输出间隔约 0.08197 ms。其单个样本从输入到输出需要多长时间，仍需全文中的流水线延迟数据。

### Sui：分别记录算法运行速度和能耗估算

正文 p14 给出 RTX 3090 GPU 与 Xeon Gold 6130 CPU 平台的算法时间统计，期望总时间为 3.35 ms。p14～15 按 Loihi 单位突触运算与神经元更新的参考能耗估算，Table 5 给出一次 SNN 分割约 0.023 J。模式出现比例为异常 2.44%、SNN 分割 3.57%、形态学更新 93.98%。原文最后报告算法功率 82.2 mW，加传感器 20 mW 得到 102.2 mW。

本报告保留这些分别报告的量及其运行条件；0.023 J 对应一次分割网络计算，102.2 mW 对应整套模式切换系统的估算。

### E-SpikePT 与 Hueber：计算依据

E-SpikePT §4.3 使用 45 nm 工艺下每次加法累加 0.9 pJ、每次乘加 4.6 pJ 计算能量。正文 p8 说明该估算未计存储访问、数据搬运和脉冲路由。49 µJ 按这套运算成本模型记录。

Hueber 的效率预测使用 SENeCA 能耗模型；约 3 ms 延迟按三个连续脉冲层的流水处理估计。原文 §IV-B、结论均将其标为预测，实际部署列为后续工作。

## 8. 原始来源与核读范围

### 已保留的硬件对照

- Tan et al., *Toward Efficient Eye Tracking in AR/VR Devices: A Near-Eye DVS-Based Processor for Real-Time Gaze Estimation*, IEEE TCSI 2025。[DOI](https://doi.org/10.1109/TCSI.2025.3553497)。核读正式 PDF 印刷页 5757 的 Fig.13/14、V-C，以及页 5758 的 Table III；本仓库提供出处，全文留在原项目文献目录。
- Zhang et al., *Co-designing a Sub-millisecond Latency Event-based Eye Tracking System with Submanifold Sparse CNN*, CVPRW 2024。[作者全文与 Table 2](https://arxiv.org/html/2404.14279v1#S4.T2)。
- Khaki et al., *A Resource-Efficient Multi-Function Embedded Eye Tracker System Implemented on FPGA*, IEEE Access 2023。[作者机构正式全文](https://scholar.gist.ac.kr/bitstream/local/9829/2/43745.pdf)，[DOI](https://doi.org/10.1109/ACCESS.2023.3342839)。资源和功率取印刷页 142936。
- Kowalski et al., *Hybrid FPGA-CPU pupil tracker*, Biomedical Optics Express 2021。[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8548015/)，[DOI](https://doi.org/10.1364/BOE.433766)。相机、处理时钟取 2.4/2.5，延迟取 Table 3。

### 新增 FPGA 对照

- Ando, Moshnyaga, Hashimoto, *A Low-Power FPGA Implementation of Eye Tracking*, ICASSP 2012。[原文 PDF](https://dihana.cps.unizar.es/proceedings/ICASSP/2012/pdfs/0001573.pdf)，[DOI](https://doi.org/10.1109/ICASSP.2012.6288193)。核对原文 3.2 与 Table 1/2 的周期、资源和 Quartus 功率估算。
- Yang et al., *30-2: Implementation of a Real-time Eye Gaze Tracking Solution for ASIC Based on VR Display*, SID 2018。[DOI](https://doi.org/10.1002/sdtp.12579)，[作者论文摘要](https://www.researchgate.net/publication/325490914_30-2_Implementation_of_a_Real-time_Eye_Gaze_Tracking_Solution_for_ASIC_Based_on_VR_Display)。本次取得摘要，记录 Kintex-7 原型、200 MHz、超过 380 Hz 和不超过 1.5°；资源及功率待全文核读。
- Yu et al., *High-speed Gaze Detection Using a Single FPGA for Driver Assistance Systems*, Journal of Real-Time Image Processing 18, 681–690, 2021，2020 年在线发表。[出版社摘要](https://link.springer.com/article/10.1007/s11554-020-01004-8)。核对 640×480、80 FPS、100 MHz、分类模块 0.52 µs 与 92% 识别率；资源和功率待全文核读。
- Wu and Han, *Design of an FPGA-accelerated Real-time Gaze Estimation System*, Journal of Physics: Conference Series 2897, 012015, 2024。[DOI](https://doi.org/10.1088/1742-6596/2897/1/012015)，[作者公开论文](https://www.researchgate.net/publication/386448988_Design_of_an_FPGA-accelerated_real-time_gaze_estimation_system)。本次新增数值核对摘要中的 Zynq-7035、32 FPS 与平均 6.4 W；功率测量范围及资源待全文核读。

### 新增 SNN 对照

- Bonazzi et al., *Retina: Low-Power Eye Tracking with Event Camera and Spiking Hardware*, CVPR Workshops 2024。[正式全文](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/papers/Bonazzi_Retina__Low-Power_Eye_Tracking_with_Event_Camera_and_Spiking_CVPRW_2024_paper.pdf)。核读 §5.2.4、Table 9/10，包括测量电源分项和能量单位。
- Jiang et al., *Eye Tracking Based on Event Camera and Spiking Neural Network*, Electronics 13(14), 2879, 2024。[正式全文](https://doi.org/10.3390/electronics13142879)。核对 §4.1 的 RTX 3090 实验平台及结论中的后续硬件部署。
- Groenen et al., *GazeSCRNN: Event-based Near-eye Gaze Tracking Using a Spiking Neural Network*, arXiv:2503.16012v1, 2025。[全文](https://arxiv.org/html/2503.16012v1)。核对摘要的 6.034°及讨论中的后续延迟、能耗硬件评价。
- Zheng et al., *NeuroEye: A 54.59mW, 12200FPS Event-Driven Near-Sensor Eye-Tracking Processor with Pipelined Spatial-Temporal Spike-Streaming*, ISCAS 2025。[IEEE 摘要](https://doi.org/10.1109/ISCAS56072.2025.11043272)。核对摘要所列 40 nm、54.59 mW、12,200 FPS、4.47 µJ/次及布局布线后评估阶段；全文待取得。
- Paredes-Valles et al., *Realizing Fully-Integrated, Low-Power, Event-Based Pupil Tracking with Neuromorphic Hardware*, arXiv:2511.20175v1, 2025。[全文](https://arxiv.org/html/2511.20175v1)。核读 §2.1、§4.3、Fig.6 及结论，区分 Speck2f 和微控制器功率。
- Hueber et al., *Neuromorphic Eye Tracking for Low-Latency Pupil Detection*, arXiv:2512.09969v1, 2025。[全文](https://arxiv.org/html/2512.09969v1)。核读 Table III、§IV-B 与结论中的 SENeCA 模型和延迟预测。
- Sui et al., *Event-based Low-power Spiking Gaze Estimation*, Engineering Applications of Artificial Intelligence 165, 113359, 2026。[出版社入口](https://doi.org/10.1016/j.engappai.2025.113359)。核读本地正式全文 p14～15 的速度平台、§4.5 与 Table 4/5；全文保留在父项目文献目录。
- Kang et al., *E-SpikePT: Energy-efficient Event-based Eye Tracking with Spiking Neural Networks and Knowledge Distillation*, Neurocomputing 681, 133433, 2026。[出版社入口](https://doi.org/10.1016/j.neucom.2026.133433)。核读本地正式全文 §4.3 与 p8，核对运算成本、49 µJ 和实际硬件部署状态。

### 本项目机器证据

- [整网回放](../../evidence/current/rtl_replay/report.json)、[资源报告](../../evidence/current/vivado/reports/post_route_utilization.rpt)、[功率估算](../../evidence/current/vivado/reports/post_route_power_vectorless_160mhz.rpt)、[当前指标](../../evidence/current/metrics.json)。本次文献更新保留这些数据和 RTL 原样。
