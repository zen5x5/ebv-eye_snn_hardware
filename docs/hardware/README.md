# 硬件设计索引

本目录是从EBV-Eye主工程导出的硬件阅读材料。独立仓库中的机器结果位于
`evidence/`，目录对应关系见`evidence/README.md`；下文保留的主工程绝对路径
只用于记录原始产物来源。

2026-10-12已补双任务24用户L40S正式模型的16位量化：Pursuit连续0.825°、
Saccade连续1.326°，详见
[正式多用户量化](06_formal_l40s_quantization.md)。
该结果用于正式硬件精度汇报；以下RTL对象仍为pursuit user15 eye0的最终L1/L2跨bin无状态SNN。项目已有
位精确定点参考、两套golden vectors和硬件镜像，以及三套完整整网RTL：
“稠密扫描基线”“仅浅层事件驱动”“浅层事件驱动＋深层逐层相邻执行双时间步”。
后两者均从400个排序单位事件开始，L1用72个LUT加法lane，L2用48个DSP
count-MAC，L3以后使用共享288-lane规则核心。

三seed完整validation位精确回放共检查345,080,448个值，零失配。seed2022
validation样本0的两套事件驱动整网RTL各检查159,408个值、零失配，
周期为72,431/67,929；既有稠密基线为83,286拍。完整validation的RTL校准
平均周期为72,444.9/67,942.9，后者相对稠密基线减少18.42%。当前仍无Vivado
时序、实现后资源、功率或上板结果，200 MHz换算只能写为条件值。按同一RTL
scheduled-lane口径，两条事件驱动路线平均7.911147M，相对稠密扫描基线的
14.592408M减少45.79%；这是发射lane slot，不是切换活动或能量。仅浅层事件
驱动与最终方案的Yosys整顶层均为354个DSP，BRAM36等效为117/80.5，物理LUT
理论界为52,432～96,408/56,211～95,133；两组界均跨过器件70,560容量线，
是否能布局及达到200 MHz必须由Vivado确定。

- [Pursuit L1/L2无状态SNN：上EDA前的语义、dataflow与验证方案](01_pursuit_l1_l2_stateless_pre_eda_design.md)：推理合同、逐层存储账、12组数值范围审计、全状态/无状态公平对照、复用方案、AUP-ZU3目标和一至三周交付边界。
- [Pursuit硬件dataflow、PE与逐通道设计空间](02_pursuit_dataflow_pe_and_channel_design_space.md)：驻留层级、循环顺序、稀疏边界、逐通道负载、47个PE候选及当前实现顺序。
- [Pursuit定点参考与并行卷积核心](03_pursuit_fixed_point_and_parallel_core.md)：S16/S24/S16位精确合同、三seed定点精度、golden vectors、144/288-DSP核心、周期/存储估计及RTL微核结果。
- [Pursuit整网稠密基线RTL与Vivado前验证](04_pursuit_complete_pre_vivado_rtl.md)：完整L1～readout数据通路、三条时间dataflow逐值回放、稠密RTL周期、lean/full开源映射及其证据边界。
- [Pursuit浅层事件驱动与深层逐层双时间步复用整网RTL](05_pursuit_sparse_frontend_and_layer_major_rtl.md)：排序事件入口、L1 add-only/L2 count-MAC、L2默认平面、三种整网实现的算术/访存/周期对照、完整validation位精确证据及Vivado前边界。
- [L40S正式模型双任务24用户16位量化](06_formal_l40s_quantization.md)：48个seed2021模型完整test评估、浮点/定点精度、数值验证与可复现产物。
- [导师硬件进展汇报](07_advisor_hardware_progress.md)：集中展示最终混合dataflow、48模型定点、L1～readout完整RTL、三路线周期/访存/资源消融、逐值正确性和已有硬件定位。
- [最终硬件架构图：浅层事件驱动＋深层逐层相邻执行双时间步](figures/ebv_eye_event_driven_layer_adjacent_architecture.svg)：整网数据通路、片上权重与局部状态，以及每层内`t0/t1`相邻执行；同目录提供PDF和高清PNG。
- 数值范围工具：`code/hardware/profile_snn_numeric_ranges.py`。
- 12组机器结果：`/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/numeric_range_profile_20261006/`。
- 逐通道机器结果：`/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/channel_profile_20261006/`。
- PE设计空间：`/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/dataflow_design_space_20261006.json`。
- 三seed突触算术对照：`/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/synapse_arithmetic_comparison_20261006/`。
- 三seed全split定点结果：`/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/fixed_point_reference_20261006/`。
- RTL golden vectors：`/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/golden_vectors_seed2022_20261006/`。
- L1校准的实现前周期/存储估计：`/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/rtl_schedule_estimate_l1calibrated_20261006.json`。
- RTL仿真、lint和开源映射：`/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/rtl_microkernels_open_source_20261006/`。
- 含流水计数核心的开源检查：`/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/rtl_core_open_source_20261006/`。
- 完整L1正式golden回放与开源映射：`/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/rtl_l1_streaming_golden_20261006/`。
- 全状态golden vectors：`/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/golden_vectors_full_state_seed2022_20261006/`。
- 最终无状态/配对全状态硬件镜像：`/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/hardware_image_stateless_seed2022_20261006/`与`hardware_image_full_state_seed2022_20261006/`。
- 整网三dataflow回放、lint与Yosys映射：`/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/rtl_full_accelerator_pre_vivado_20261006/`。
- 稀疏三路线validation、整网回放与映射：`/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/rtl_sparse_three_schedule_pre_vivado_20261008/`。

当前候选 FPGA 为 RealDigital AUP-ZU3，器件
`XCZU3EG-2SFVC784E`。行为级 RTL 仿真不需要实体板卡；器件相关的综合、
布局布线和时序分析需要 Vivado 与准确 part，上板回放才需要板卡。
