# 2026-10-15 历史：AUP-ZU3 模型选择与 Vivado 优化过程

本篇保留开发过程及原运行位置。当前交付代码与可移植资产见 [当前设计](../hardware/current_design.md) 和 [机器证据](../../evidence/README.md)。

## 模型选择与精度

检查了服务器上 24 用户 × 3 seed = 72 个正式 Pursuit L1/L2 跨时间步无状态模型的 `summary.json`。每个 checkpoint 原本均按其用户的 validation 误差选取；这里只比较已经训练完的 checkpoint，没有再训练。按各自 test 平均角误差，前三个为 user17/seed2022 0.489147°、user10/seed2021 0.490071°、user12/seed2022 0.491813°。不同用户的 test 集并不相同，且跨用户按 test 排序是部署候选选择，不是无偏泛化评估。此前把“seed2021 中最佳”的 user10 误称为全部 72 个模型中的最佳，现已纠正。

选定 user17/seed2022，第 193 轮 checkpoint，validation 0.494873°。checkpoint SHA-256 为 `04dfaabd03ed87a2ca8068a68c08b23b2bdcb5375580ac2441f63adf7ab2cd66`。服务器在其 1,364 个 test 样本上重新运行与正式量化相同的 16 位权重、40 位中间部分和、24 位电流、16 位膜状态、8 位脉冲规则，浮点/定点误差分别为 0.489147°/0.488592°。量化参数与随后生成的硬件 golden 参数文件 SHA-256 同为 `3c34b8ae0202f7d6f77975d8e13b5ca7dd5e8361ee294fcfb4b69d36ea93ce13`。

服务器证据：

- 量化：`/data/xuzx/result/ebv-eye/analysis/formal_fixed_point_user17_seed2022_l40s_20261015/`。
- 16 个 validation 样本的逐层 golden：`/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/golden_vectors_l40s_user17_seed2022_20261015/`。
- 可装载的权重/常量镜像：`/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/hardware_image_l40s_user17_seed2022_20261015/`。
- 对应模型、量化、golden、镜像已复制到本机 `/home/xen/project/02_eb_snn_gaze_system/data/`，文件哈希与服务器一致。

## 整网 RTL 正确性

服务器的独立副本 `/datanaco/xuzx/project/ebv-eye_vivado_local_check_20261015/` 使用 user17 的权重、事件和逐层标准值重新运行完整加速器。最终逐通道 L2 版本检查 159,408 个逐层及输出数值，零失配；Verilator 静态检查通过。该样本共运行 100,880 拍，其中事件驱动前端 48,210 拍、L1 非零包 3,315 个。验证报告为该副本下的 `result_l40s_user17_seed2022_l2_serial_neuron_verified_counts/report.json`，本机副本为 `/home/xen/project/02_eb_snn_gaze_system/results/rtl_validation/rtl_result_l40s_user17_seed2022_l2_serial_neuron_verified.json`。

原模型权重由主机在推理前写入 RTL 存储器，并非综合时固化。因此同一 RTL 的 Vivado 逻辑资源与选用 user10 或 user17 权重无关；不同模型的发放活动会改变实际运行周期，必须分别用各自的 golden 仿真，不能沿用旧周期。

## 本机 Vivado 器件检验

工具为 Vivado 2025.2.1，器件 `xczu3eg-sfvc784-2-e`，计算时钟约束 5 ns（目标 200 MHz）。这是完整加速器顶层的脱离板级接口实现（OOC），不包括 Zynq PS、AXI、引脚或板级镜像；板卡不必上电。

初始 24 路 L1 版本不能布局；把 L1/L2 累加器与深层部分和改为块 RAM 后，上一版综合用 68,340 LUT、196 RAMB36 + 40 RAMB18、308 DSP，仍因缺 106 个 CLB 不能布局。仅交换 L2 池化暂存和一个累加器银行的 RAM 类型，综合为 68,245 LUT，同样不能布局，缺 58 个 CLB。随后让 L2 的 16 个神经元逐通道复用一套电路，综合降到 65,786 LUT、196 RAMB36 + 40 RAMB18、308 DSP，功能仍逐值正确，但周期增加。该版成功布线，布线后占 63,043 LUT、196 RAMB36 + 40 RAMB18、308 DSP，setup 最差余量为 -5.801 ns，hold 最差余量为 +0.043 ns。因此面积/可布线性得到证实，但没有达到 200 MHz。原批处理脚本在导出严重警告时把一个带空格的级别写成两个列表项，导致已保存布线报告后没写汇总文件；现已按 Vivado 实测语法修正，不能把该脚本错误误称布线失败。

上述布线运行目录：`/home/xen/project/02_eb_snn_gaze_system/results/vivado/20261015_sparse_lm_a_l40s_user10_l2_serial_neuron/`。目录中的 `user10` 是选型纠正前建立的运行名，RTL 是可写权重的通用结构；user17 的功能和周期由上面的独立 RTL 回放验证。按该版布线报告的第一层最慢路径，又将 L1 池化和脉冲计算分成相邻两拍；user17 的完整 RTL 再次检查 159,408 个值零失配、Verilator 静态检查通过，实测 103,207 拍，其中事件驱动前端 50,537 拍。其服务器报告为独立副本下的 `result_l40s_user17_seed2022_l1_pipeline_l2_serial_verified_counts/report.json`，本机副本为 `/home/xen/project/02_eb_snn_gaze_system/results/rtl_validation/rtl_result_l40s_user17_seed2022_l1_pipeline_l2_serial_verified.json`。

L1 拆分版的本机 Vivado 运行目录为 `/home/xen/project/02_eb_snn_gaze_system/results/vivado/20261015_sparse_lm_a_l40s_user17_l1_pipeline_l2_serial/`。完整顶层成功布局并布线，布线后为 64,423 LUT、19,481 FF、196 RAMB36 + 40 RAMB18、308 DSP。5 ns 目标下 setup 最差余量 -2.357 ns、hold 最差余量 +0.036 ns；没有未约束的时序问题或严重警告，源文件哈希在运行结束时复核未变。批处理退出码为 1 是其如实拒绝未满足 200 MHz 的结果，不是布线失败。当前最慢路径从共享卷积引擎的片上块 RAM 到该引擎中脉冲神经元的膜状态寄存器，经过 24 级逻辑，布线后数据路径延迟 7.291 ns；若继续优化时序，应先针对这条真实路径拆分或缩短组合运算，再分别重验整网数值和 Vivado 时序。不能把 103,207 拍按尚未达到的 200 MHz 换算成已实现推理时间。

以上是继续优化前的基线。它没有达到 200 MHz；下面的新版本已在同一器件的布线结果上通过 160 MHz 时序复核。当前仍无板级 bitstream、功率实测或上板推理结果，板卡保持断电即可。

## 参考 Tan 2025 后的本机结构优化

Tan 2025 的芯片让权重留在计算单元附近，复用相邻窗口的输入，并把中间和暂存在与计算顺序相配的小缓冲中，以减少反复访问大存储器。这里借鉴的是“计算节奏和中间结果存储一起设计”的原则，而不是照搬其 144 路乘法器或将 ASIC 的功耗当作 FPGA 的功耗。原文：Tan et al., *Toward Efficient Eye Tracking in AR/VR Devices: A Near-Eye DVS-Based Processor for Real-Time Gaze Estimation*, IEEE TCSI 2025，DOI 10.1109/TCSI.2025.3553497。

对本项目先前布线报告的真实最慢路径逐段核查后，做了三项保持数值规则不变的修改：

1. 在脉冲神经元中，把正常情况下“减去整数个阈值”改写为直接取低位余数；脉冲数超过 U8 上限时仍保留原来的截断和相减语义。卷积引擎在衰减膜电位与电流相加后插入一拍寄存器，下一拍再完成脉冲和复位。
2. L2 的 16 个神经元仍逐通道共用一套电路，但把“池化结果加折叠常量并钳位为 24 位电流”与“计算脉冲及膜电位”拆成相邻两拍，并在每个输出位置增加一拍排空。每通道仍为一拍吞吐，整网只增加 798 拍。
3. L2 的九个累加器银行每个只有 22 个地址、每地址 640 位；改用逻辑单元内的分布式存储，而 L1 保持原来的块存储。这样避免把很宽但很浅的 L2 数组拆占大量专用块 RAM。具体节省量以 Vivado 布线资源表为准，不按逻辑位数猜测。

完整整网回放使用同一 user17/seed2022 权重和 golden，检查 159,408 个逐层及输出值，零失配；新版本实测 104,005 拍。独立神经元等价测试覆盖 32 种小数位设置和 100,000 个输入周期；L2 正常包流及空输入复位的定向测试也通过。定向测试原本将空输入运行限制为 10,000 拍，但未修改的旧设计实测也需 15,455 拍；已把它修正为与首轮测试相同的 50,000 拍上限，新版本空输入实测 16,223 拍。这个测试修正不改变 RTL 行为。

| 同一 user17 模型、同一 XCZU3EG 器件 | 上一版 | 本轮选定版 |
| --- | ---: | ---: |
| 整网样本周期 | 103,207 | 104,005 |
| 布线后 LUT（逻辑单元） | 64,423 | 61,657 |
| 布线后 FF（寄存器） | 19,481 | 24,835 |
| 布线后 RAMB36 / RAMB18 | 196 / 40 | 124 / 54 |
| RAMB36 等效占用（器件共有 216） | 216 | 151 |
| 布线后 DSP（器件共有 360） | 308 | 308 |
| 5 ns / 200 MHz 最差 setup 余量 | -2.357 ns | -1.109 ns |
| 5 ns / 200 MHz 最差 hold 余量 | +0.036 ns | +0.040 ns |

新版本少 2,766 个 LUT（4.29%），释放 65 个 RAMB36 等效块（30.09%），代价是多 5,354 个寄存器（27.48%）和 798 拍（0.77%）。200 MHz 仍然未通过，不能把 104,005 拍除以 200 MHz 报成实际时延。布线后最慢路径仍在 L2：一拍内从权重存储读出、完成乘法与累加、写回中间结果；路径延迟 6.031 ns，其中 4.358 ns 为布线延迟。原检查没有未约束时序问题或严重警告，输入 RTL 和约束文件的 SHA 在运行前后相同；Vivado 脚本以退出码 1 如实拒绝 200 MHz，并非布线失败。

在完全相同的已布线 checkpoint 上，仅把计算时钟约束覆盖为 6.250 ns，重新运行 setup/hold 时序报告：setup 最差余量 +0.141 ns，hold 最差余量 +0.040 ns。因此 160 MHz 在这份脱离板级接口的实现上可满足时序。按整网回放 104,005 拍换算，已装载权重后的加速器核心约 0.650 ms/样本；若样本串行且可立即启动下一样本，换算约 1,538 样本/秒。不包括相机采集、400 事件形成与排序、权重初次装载、PS/AXI 传输及上板接口。160 MHz 复核不会生成板级 bitstream，也不能证明接口加入后仍满足时序。

正式源码现在位于 `hardware/rtl/`。本机完整布线 checkpoint、5 ns 与 6.250 ns 时序报告、资源报告、源码/脚本快照保存在 `/home/xen/project/02_eb_snn_gaze_system/results/vivado/20261015_sparse_lm_a_l40s_user17_l2_lutram_pipeline_160mhz/`。功能和周期的服务器检查仍在独立副本中完成，未覆盖正式训练 checkpoint 或 golden。Vivado 在 160 MHz 下、没有真实信号活动文件时给出的片上总功率估算为 1.835 W（其中动态 1.608 W、静态 0.228 W，工具标 Medium 置信度）；它只提示量级，不是板级或真实输入功耗，也不能用来与 Tan 的实测 ASIC 能量作优劣结论。
