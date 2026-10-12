# Pursuit SNN FPGA加速器设计文献审计

## 结论

本项目现有三套可逐值对照的整网RTL。既有`Dense-TM`是306-DSP稠密raster
正确性基线；`Sparse-TM`只把L1/L2替换为事件驱动；选定的`Sparse-LM-A`
再将深层每个权重tile的`t0/t1`相邻执行。两套稀疏顶层中，L1用72个LUT
add lane，L2用48个DSP count-MAC；L3～L8仍共享8个输出PE、每PE完整3×3核
和4个输入通道的288-DSP核心，加上H1/H2的16个DSP与readout的2个DSP，
Yosys整顶层映射均为354个DSP48E2，占器件360个DSP的98.3%。它不是单个MAC
串行遍历全网，也不是为每层复制核心。

这个组织来自三类证据：Tan 2025说明眼动网络必须同时处理激活复用、权重
驻留和部分和近端保存；FireFly在同为`XCZU3EG`器件族的平台上证明大DSP阵列
可行；FireFly-S和ESDA说明稀疏收益依赖权重稀疏、激活语义、坐标路由和负载
均衡共同匹配。本模型只有L1/L2具有足够空间稀疏性，深层目标位置几乎全部被
触及且量化权重接近无零，因此采用“浅层事件驱动、深层规则共享”而不是整网
稀疏或八层空间流水。

## 与本设计直接相关的工作

| 工作 | 平台、资源与数值 | 可借鉴部分 | 公平边界 |
|---|---|---|---|
| [Tan 2025](https://doi.org/10.1109/TCSI.2025.3553497) | UMC 40 nm ASIC；W8/A20；144 MAC；252.19 KB SRAM；200 MHz、1,252 FPS，约0.799 ms；流片实测15.2 mW、12.7 µJ/次估计 | 3×3核展开、激活广播、权重驻留、寄存器部分和和卷积—Pool流式衔接 | 500事件/pitch 200、`2×120×220`、约91k参数和11.5M MAC；量化平均/最佳误差1.15/0.91°。ASIC面积、功耗和协议不能同本项目FPGA条件值直接判胜负 |
| [SEE/ESDA-eye 2024](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/html/Zhang_Co-designing_a_Sub-millisecond_Latency_Event-based_Eye_Tracking_System_with_Submanifold_CVPRW_2024_paper.html) | ZCU102 FPGA＋Arm；W8/A8；SEE-D为178k参数、3.71 px，SCNN/GRU＋FC分别0.59/0.11 ms，总0.70 ms；3.86 W、2.29 mJ/次；1,606 DSP、1,092 BRAM、130k LUT | 事件坐标、稀疏卷积和异构系统的真实FPGA速度参照 | 输出为pupil/eye center，GRU/FC运行在CPU；submanifold语义、器件规模和功耗范围与本项目不同 |
| [NeuroEye 2025](https://doi.org/10.1109/ISCAS56072.2025.11043272) | 40 nm CMOS post-layout；全脉冲、跨层空间—时间流；12,200 FPS、54.59 mW、4.47 µJ/frame | 展示逐层复制与空间—时间流水能达到的SNN吞吐上界 | 不是本项目的共享核心映射，公开摘要未给出可与屏幕gaze直接对齐的误差协议 |
| [JaneEye 2026](https://arxiv.org/abs/2510.01213) | 12 nm ASIC post-layout；W8/A16；17.6k参数；pupil误差2.45 px；400 MHz、0.50 ms、2,000 FPS、18.9 µJ/frame | 64-PE权重驻留、输出驻留和激活零跳过 | 输出为pupil center，模型为ConvJANET而非SNN直接gaze |
| [Event-Gaze 2026](https://doi.org/10.1109/VLSITechnologyandCir65830.2026.11577258) | 28 nm实测SoC；事件图构建＋GNN眼动＋Transformer gaze；1.12°、441 µs、12.88 mW | 最新直接gaze专用芯片的延迟和精度参照 | 模型、数据协议和工艺与本项目不同；公开摘要未给完整资源与操作口径 |
| [FireFly 2023](https://doi.org/10.1109/TVLSI.2023.3279349) | Ultra96v2、同为`XCZU3EG`器件族；INT8/binary；15k LUT、288 DSP、162 BRAM；300 MHz；CIFAR/DVS时延1.035/3.546 ms；Vivado估计2.55 W且不含PS | 同器件族的大DSP阵列、权重驻留及状态/激活带宽设计 | 论文未给足信息确认相同package/speed grade；其DSP打包多个binary×INT8突触，且端点是分类，不能比较peak GOP/s/DSP |
| [FireFly v2](https://arxiv.org/abs/2309.16158) | Ultra96v2；23k LUT、256 DSP、103 BRAM；逻辑/DSP 300/600 MHz；DVS-Gesture 1.989 ms；Vivado估计总SoC 6.2 W | 输出驻留、时间步内循环、双时钟和多维参数化 | 时间步局部化改变状态RAM组织；本项目方案A只借鉴相邻时间复用，仍按自己的S16/U8算术和真实周期报告 |
| [FireFly-S](https://doi.org/10.1109/TCSI.2024.3496554) | XCZU5EV；4 bit、85%～95%权重稀疏；DVS-Gesture用55,234 LUT、98,731 FF、11 BRAM、0 DSP；333 MHz、3,683 FPS；Vivado估计1.821 W | bitmap双侧稀疏检测、逐层空间流水和负载均衡 | 高权重稀疏、低位宽和每层复制是零DSP与吞吐的前提；本项目S16权重几乎无零，不能套用这些结果 |
| [ESDA](https://doi.org/10.1145/3626202.3637558) | ZCU102/XCZU9EG；8 bit submanifold sparse DNN；1,532 DSP、848 BRAM；187 MHz、0.66 ms、1,526 FPS；PMBus实测PL 1.58 W | 稀疏表示、坐标路由、层间流水及板级测量方法 | 不是SNN，器件大得多；submanifold语义保持原坐标集合，会改变本项目普通same-padding卷积、Pool和常量平面的语义 |
| [Cheng 2025](https://doi.org/10.1109/TCSI.2025.3560666) | ZCU104、三层全连接分类SNN；层间结构稀疏和early stop；250 MHz、3.22 GSOP/s | 工作队列和结构化稀疏可作为以后重训路线参考 | 没有卷积/Pool，分类early stop没有连续gaze的直接判据；当前路线不为硬件收益重训模型 |

外部比较可分为三组。本项目方案A为67,942.9拍均值，按冻结的200 MHz设计点
换算为0.340 ms和2,943.6 sample/s。直接screen-gaze组以Tan 2025的0.799 ms、
1,252 FPS、1.15°均值和Event-Gaze 2026的0.441 ms、1.12°为主要参照；真实
FPGA眼动组以SEE-D的0.70 ms、2.29 mJ/次为参照；SNN/眼部处理组以NeuroEye、
JaneEye和Retina说明逐层流水吞吐、ASIC效率与超低功耗Speck三种不同设计点。
FireFly系列单独用于同器件族的资源、DSP组织和时间局部化比较。论文主结果保留
本项目的周期、scheduled lane、权重读取和状态访问，同时在外部表中明确任务
输出、平台和证据类型。

## 给论文写作agent的性能比较入口

论文硬件实验应先报告同一模型的三条RTL路线，再用外部工作定位设计水平。
本项目主结果来自[稀疏前端与方案A整网RTL报告](../hardware/05_pursuit_sparse_frontend_and_layer_major_rtl.md)：

- `Dense-TM`、`Sparse-TM`、`Sparse-LM-A`平均周期为83,286.0、72,444.9、
  67,942.9拍；A相对Dense减少18.42%，相对Sparse-TM减少6.21%；
- scheduled arithmetic lane slot由14.592408M降至7.911147M，减少45.79%；
- A相对Sparse-TM把256-bit运行期权重读取由11,892降至6,060拍，减少49.04%；
- BRAM36等效由117降至80.5，减少31.20%；
- 67,942.9拍按冻结的200 MHz设计点换算为0.340 ms和2,943.6 sample/s。

外部性能表优先放Tan 2025、Event-Gaze 2026、SEE/ESDA-eye 2024、NeuroEye
2025、JaneEye 2026和Retina 2024。按延迟/服务率，本项目200 MHz设计点相对
Tan、Event-Gaze、JaneEye和SEE-D分别为2.35×、1.30×、1.47×和2.06×吞吐；
NeuroEye的12,200 FPS约为本项目的4.14×。直接screen-gaze精度可并列本项目
0.767±0.021°、Tan平均1.15°和Event-Gaze 1.12°，不计算跨数据集相对精度
提升。FireFly与FireFly v2放在架构比较：本项目/FireFly/FireFly v2分别使用
354/288/256 DSP和80.5/162/103 BRAM，说明本项目用更多DSP支持S16/U8
MultiSpike直接gaze，同时通过层内时间相邻复用控制片上块RAM。

推荐结果叙事是：同模型RTL消融给出因果清楚的计算、周期与访存收益；外部表
说明本设计达到亚毫秒级直接gaze服务率；在已核验工作集合中，本项目同时具备
事件输入、SNN、连续屏幕gaze回归和L1至readout完整RTL，不写成未经穷尽检索的
全球首次主张。定点位宽与精度取自[定点参考报告](../hardware/03_pursuit_fixed_point_and_parallel_core.md)。

## 对当前dataflow的约束

稀疏前端固定为：

```text
400 sorted unit events
  -> L1 add-only scatter, 9 taps x 8 outputs
  -> nonzero U8 L1 packets
  -> L2 count-MAC scatter, 3 taps/cycle x 16 outputs
  -> finalize the complete 16 x 24 x 32 L2 plane
```

L1/L2用三行九bank S40环形部分和，完成一行后立刻Pool并清除；L2不能只输出
被事件触及的位置，因为三个checkpoint各有2/3/4个通道会由折叠常量单独发放，
所以完整默认平面必须保留。三seedvalidation平均L1包为2,554.4，L2有效
count-MAC为364,499.3。

三套顶层把“空间稀疏”和“时间顺序”拆成两个对照：Dense-TM是原稠密raster
SL-HYB；Sparse-TM只替换L1/L2，仍用L3整层电流缓存和先`t0`后`t1`的深层
顺序；Sparse-LM-A使用同一稀疏前端，并让每个L4～L8权重tile装入一次后依次
完整扫描`t0/t1`，H1/H2也复用同一权重beat。L3输入两步相同，只卷积一次，
使用4.5 KiB tile局部电流而不是核心外部的13.5 KiB整层片上缓存。方案A仍是一个共享
288-lane核心顺序执行各层，不是跨层波前或多样本流水。

方案A保留四种驻留：前端三行部分和驻留、当前卷积权重tile驻留、两步S40
部分和分别驻留，以及`t0`膜在引擎内驻留到`t1`。完整validation的运行期
256-bit权重BRAM读取由Sparse-TM的11,892拍降至6,060拍；这正是采用相邻
两步的主要可测收益。

## PE算术选择

数学上，整数脉冲计数`n`乘权重`w`可以写成`n`次加法，硬件选择仍取决于
事件表示和实际issue数：

- L1直接消费400个单位事件，三seedvalidation平均只需28.679k次有效加法，
  因而使用72个LUT add lane；
- L2输入是U8 MultiSpike。逐单位加法平均需785.234k次，而非零count乘权重
  只需364.499k次，因此使用48个DSP lane；
- L3以后空间几乎完全被触及，继续使用规则288-DSP count-MAC核心；
- 二进制分解需要`popcount(n)`个issue，平均4.690M次。若把所有bit shift
  在一个周期内展开，会得到很大的组合移位加法网络，并没有真正省下硬件；
- 当前Yosys微核映射中，8×18计数乘法为144个DSP48E2、5,760个LUT2和
  1,440个CARRY4；逐单位加法为0 DSP、8,064个LUT2和1,440个CARRY4；
  单bit移位项版本还需14,724个不同输入数LUT及1,440个CARRY4。该结果不含
  布局布线与时序，只用于排除不合理结构。

硬件主比较采用固定宽度RTL的scheduled-lane口径。Dense-TM每样本发射
14.592408M个乘法lane slot；两条稀疏路线平均发射L1 28.800k个加法lane、
L2 367.827k个乘法lane和规则后端7.514520M个乘法lane，合计7.911147M，
减少45.79%。它包含same-padding和固定通道tile中被屏蔽的lane，只表示issue
工作量，不表示数据相关toggle或能量。为与旧报告衔接，常规算法混合账仍为
12.818616M对7.903379M、名义少38.34%；该旧账的稀疏L1/L2只计in-bounds
更新，而规则层计完整kernel，不能用作同口径硬件主比较。

## 比较优势应怎样成立

现阶段已经建立位精确RTL和数据相关周期证据，仍不能提前宣称FPGA实测优势：

1. 正确性：S16权重、S24电流、S16膜状态的三seed test为
   `0.767±0.021°`，同一CPU浮点参考为`0.760±0.016°`；所有train、
   validation和test扫描均无电流或脉冲饱和。
2. 正确性范围：三seed完整validation共检查345,080,448个定点值，稀疏
   累加、Pool后常量与L3复用均零失配；三套整网顶层在seed2022样本0各检查
   159,408个值，零失配。
3. 速度：Dense-TM、Sparse-TM、Sparse-LM-A完整validation的RTL校准周期
   均值为83,286、72,444.9、67,942.9，方案A相对基线减少18.42%。只有
   Vivado满足200 MHz后，才可分别换算0.416/0.362/0.340 ms；1,278个
   稀疏样本的分布是逐样本前端FSM精确回放加整网RTL校准后端，不是1,278次
   完整RTL仿真。
4. 资源：Sparse-TM/Sparse-LM-A的Yosys整顶层均为354个DSP；BRAM36等效
   分别为117/80.5，FF为36,941/36,124。计入LUTRAM后的物理LUT理论界为
   52,432～96,408/56,211～95,133，两者均跨过70,560容量线，Yosys不能判断
   能否放入。Vivado post-route资源与WNS仍待测。
5. 模型对象：本项目直接输出连续gaze；FireFly、ESDA和Cheng的公开端点是
   分类，Retina/Speck是pupil或框回归，不能只用FPS排成一张无条件优劣表。
6. 状态贡献：Dense-TM保留同一核心下72 KiB浅层状态删除与时间复用基线；
   方案A再单独报告权重读取、局部状态和L3 tile缓存，避免把算法状态删除与
   调度收益混成一个数字。
7. 板级证据：最终至少报告XCZU3EG的WNS/Fmax、LUT/FF/BRAM/DSP、真实周期、
   AXI回放一致性和板级功率。Yosys primitive mapping不能替代这些数字。

## 已归档全文

- [FireFly TVLSI 2023](https://doi.org/10.1109/TVLSI.2023.3279349)
- [FireFly v2 arXiv 2023](https://arxiv.org/abs/2309.16158)
- Tan、FireFly-S和ESDA全文已在既有相关文献专题中归档；本轮实现不存在必须
  由用户补下载后才能继续的文献阻塞。

## ISCAS图示层次参考（2026-10-07）

本节只用于论文架构图的信息组织，不新增硬件路线或性能比较。已完整阅读下列
作者公开稿并查看关键图页；正式会议归属由作者机构/主页与DOI核对。

| 工作与全文 | 原文怎样分图 | 对本项目的启发 |
|---|---|---|
| [Windhager ISCAS 2025](https://doi.org/10.1109/ISCAS56072.2025.11043954) | Fig.1是网络硬件总架构，Fig.2/3分别展开spike sorter/neuron core，Fig.4讲输入编码 | 总体关系与局部机制分层，控制和数据通路分色 |
| [OpenSpike，ISCAS 2023](https://doi.org/10.1109/ISCAS46773.2023.10182182) | Fig.1为core、控制和分用途存储，Fig.2/3展开发放与膜运算，Fig.4独立给版图 | 膜存储应具名，与权重/激活存储区分；后端结果不挤进总图 |
| [Bezugam EMG/Loihi，ISCAS 2023](https://doi.org/10.1109/ISCAS46773.2023.10181510) | 前期稿Fig.1讲数据/编码，Fig.2讲网络，Fig.3讲Loihi映射；结果另图 | 应用算法与硬件映射分图仍可保持同一方法叙事 |
| [Windhager AICAS 2024](https://arxiv.org/pdf/2311.14447) | Fig.1–4依次为网络、编码、层架构、neuron core | 作为由总体到局部的补充参考；该文不是ISCAS |

Bezugam的公开前期稿题名为“Low Power Neuromorphic EMG Gesture Classification”，
不是最终IEEE排版稿；作者主页将其与ISCAS 2023论文关联。这里的图号明确指所读
前期稿，不断言终稿版面完全相同。其余两篇ISCAS作者稿来源分别为
[arXiv 2501.07952](https://arxiv.org/pdf/2501.07952)和
[arXiv 2302.01015](https://arxiv.org/pdf/2302.01015)。

用户在查看初稿后明确总图用双栏、硬件图用单栏，且总图须尽可能矮，因此不再
采用“overview底部增加硬件横栏”。当前两版overview均为181.35×59.27mm，带硬件
版仅在状态示意下嵌入Buffers→PE+LIF↔Deep-state RAM，不额外增高；独立硬件图
重排为88.57×77.61mm的“上部共享计算/存储，下部两步电流复用”，而非等比缩小横图。
L3缓存接在Conv/Pool后、LIF前，第二步继续更新L3及深层膜，明确复用的是电流而非
L3脉冲。当前文件、物理尺寸检查页及重建入口在相邻Overleaf仓库
`figures/overview/`；用户已认可并设为正式版，旧图、旧规划页和旧生成入口已移除。
`build_figures.py`一次生成两版总图及单栏硬件图，真实曲线/帧和来源记录保存在
`assets/`。硬件后续计划仍由项目硬件报告记录，不混入已实现架构图。

2026-10-08完成三套整网RTL后，上述单栏图中的13.5 KiB L3整层缓存准确对应
Dense-TM/Sparse-TM基线，不代表选定的Sparse-LM-A；方案A实际使用4.5 KiB
tile局部电流、层内相邻`t0/t1`和局部`t0`膜。现有图文件未在本轮静默修改，
论文若将方案A作为最终实现，须在用户确认版式后同步更新图和Table III。
