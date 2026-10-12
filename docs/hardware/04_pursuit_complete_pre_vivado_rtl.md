# Pursuit整网稠密基线RTL与Vivado前验证

> 2026-10-08边界修正：本文顶层按完整空间raster调度，零脉冲仍占DSP lane与
> 周期。它保留为位精确正确性、状态RAM和时间复用的`Dense-TM`对照；已完成的
> `Sparse-TM`与`Sparse-LM-A`稀疏实现见05报告。本文其余数字只表示已验证的
> 稠密基线。

## 结论

截至2026-10-06，pursuit user15 eye0已经形成可综合的整网稠密基线。现有
实现不是全模型串行乘加器，而是：

- 一个供L1～L8共享的`8输出PE × 4输入通道 × 9卷积核点`卷积核心，即
  288条并行`U9×S16`乘法lane；
- 一个16-lane H1/H2核心和一个2-lane readout核心；
- 片上权重、输入/激活ping-pong、S40部分和、跨bin膜状态和可选L3电流缓存；
- 完整的L1～L8、H1、H2、时间聚合和readout控制器；
- FS-TM、SL-TM、SL-HYB三条可选择dataflow。

seed2022 validation样本0的整网回放已经逐值检查电流、脉冲、膜和最终两个
readout整数。三条路线均为零失配、零范围错误。这里的周期数来自完整RTL状态机，
不再是解析估计；200 MHz对应的毫秒数仍以Vivado最终满足5 ns时序为前提。

| dataflow | 模型与执行方法 | 检查值数 | RTL周期 | 若200 MHz成立 |
|---|---|---:|---:|---:|
| FS-TM | 配对全状态epoch100；两个bin均计算整网 | 270,000 | 119,791 | 0.598955 ms |
| SL-TM | 最终L1/L2无状态epoch200；两个bin均计算整网 | 270,000 | 119,791 | 0.598955 ms |
| SL-HYB | 最终模型；第二个bin复用L3 Pool后电流 | 159,408 | 83,286 | 0.416430 ms |

SL-HYB比SL-TM少36,505拍，即30.47%。FS-TM和SL-TM周期完全相同，因而
前两行能把状态容量差异与计算复用分开；第三行才加入重复输入特有的复用。

## 三条dataflow究竟做什么

FS-TM和SL-TM都按`time → layer → output-channel tile → input-channel tile →
spatial raster`执行。每个卷积tile把288个权重装到PE近端寄存器，三行输入窗口
驻留在行缓冲中，8路输出部分和留在寄存器或S40部分和RAM，完成全部输入通道后
才做折叠常量、可选MaxPool和LIF。权重整体常驻片上RAM，层间只切换地址范围。

SL-HYB利用了两个已经成立的模型条件：两个bin的输入计数图完全相同，且L1/L2
不保留跨bin膜。因此L1、L2的第二步输出严格等于第一步，L3的卷积及Pool后电流
也相同。硬件在`t0`保存13.5 KiB的L3 S24电流；到`t1`时只将它送入保留了
`t0`膜的L3 LIF，然后继续计算L4～readout。它跳过第二次L1、L2、L3卷积，
没有错误地复用L3脉冲，因为L3本身仍是有状态层。

## 冻结的算术和存取格式

推理合同为S16权重、S24归一化电流、S16膜、U8 MultiSpike、U9输入计数和
S40内部部分和。每个输出通道有独立的2的幂scale，最多15个小数位；
`alpha=0.75`实现为有符号`floor(3*v/4)`，静态参数采用round-nearest-
ties-away，部分和累积期间不饱和，只在完整dot product后检查并钳位到S24。

参数打包器已把两套模型分别转换为硬件直接可装载的镜像：

- 卷积：256-bit beat，顺序为layer、output tile、input tile、output lane、
  input lane、3×3 kernel；
- H1/H2：256-bit beat，input-major，每拍16个S16权重；
- readout：每个输入同时读取两个S16输出权重；
- 折叠常量、逐通道fraction bits和readout fraction bits分别存放。

打包器会从镜像反向重建每个张量并逐元素核对；golden manifest、镜像布局、
RTL源文件、testbench和工具输出均记录SHA-256。最终readout端口输出两个S64
整数和各自fraction bits，物理坐标换算为
`integer / 2^(fraction_bits + 1)`；其中额外的1来自两个bin求和后的除2。

## RTL组成与验证覆盖

顶层为`code/hardware/rtl/pursuit_snn_accelerator_top.sv`，其主模块包括：

- `conv3x3_layer_engine.sv`：可编程L1～L8、同步权重/参数读取、same padding、
  行缓冲、288-lane流水乘加、S40部分和、可选2×2 MaxPool和LIF；
- `linear_lif_layer_engine.sv`：16-lane H1/H2及其膜状态；
- `readout_engine.sv`：2-lane连续输出；
- `cached_current_lif_replay.sv`：L3缓存电流的第二bin LIF回放；
- `pursuit_activation_store.sv`：U9输入和8-bank激活ping-pong；
- `pursuit_conv_state_store.sv`：lean/full两种深度的8-bank S16状态RAM；
- `pursuit_l3_current_cache.sv`：8-bank S24缓存；
- 顶层调度器、参数RAM、H1/H2状态、时间求和和逐层trace接口。

验证不是只比最终坐标。完整testbench在每个有效transaction处核对所有lane的
S24电流、U8脉冲和S16膜，并核对最终两个S64 readout；L1/L2在SL模式明确
检查无状态语义，L3～H2在`t1`读取`t0`状态。各子模块也有独立正式golden回放。

| 已验证对象 | 覆盖与结果 |
|---|---|
| L1～L8共享卷积核心 | 两个bin逐层回放，共269,568个值，零失配 |
| H1/H2 | 两个bin共432个电流/脉冲/膜值，零失配 |
| readout | 两个S64整数，零失配 |
| L3电流回放 | 13,824个值，零失配 |
| 整网三路线 | 699,408个逐层/最终值，全部通过 |
| 静态检查 | lean/full两种顶层均通过Verilator `--Wall` |

整网机器报告位于
`/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/rtl_full_accelerator_pre_vivado_20261006/report.json`；
该报告自身SHA-256为
`45e34698bf4c55ff14d92b46fb67c556b8e00656eefdf00172b5da26753e8eb4`。

## 开源映射给出的资源风险

Yosys 0.69以`xcup` primitive做了无IO pad的综合映射。它能提前发现RAM没有
推断成BRAM、地址乘法误占DSP等结构问题，但没有placement、routing、WNS、
Fmax或功率含义。

| 构建 | DSP48E2 | RAMB36E2 | RAMB18E2 | BRAM36等效 | logic LUT primitive合计 | FF |
|---|---:|---:|---:|---:|---:|---:|
| lean：SL-TM/SL-HYB | 306 | 93 | 40 | 113 | 51,100 | 34,048 |
| full：FS-TM | 306 | 117 | 32 | 133 | 51,250 | 34,052 |

306个DSP由卷积288、H1/H2 16和readout 2组成。full构建比lean多20块
BRAM36等效，正是保留L1/L2跨bin状态的器件代价；计算阵列没有变化。

XCZU3EG名义容量为360 DSP、216块36 Kb BRAM、70,560个物理LUT和141,120 FF。
lean/full的Yosys映射分别对应85.0% DSP、52.31%/61.57% BRAM36等效和
24.13% FF。表中的51,100/51,250是LUT1～6逻辑primitive数，兼容的小函数可由
一个物理LUT6_2双输出打包，不能直接除以70,560当作利用率。再计入5个RAM64M8
各占8个物理LUT容量，lean/full的理论物理LUT界分别为25,590～51,140和
25,665～51,290；下界假设最大兼容打包，上界假设logic primitive完全不共享。
该理论界不是physical utilization或可布局证据，真实打包、拥塞和时序仍须
Vivado。器件容量见
[AMD DS891](https://www.amd.com/content/dam/xilinx/support/documents/data_sheets/ds891-zynq-ultrascale-plus-overview.pdf)，
AUP-ZU3原理图确认板上器件为
[`XCZU3EG-2SFVC784E`](https://realdigital.org/downloads/a2c758bf69ee5d5dc46064fe10ca440f.pdf)。

实现过程中已经修掉两类会虚增资源的结构：多维统一RAM改成显式8-bank
单端口/简单双端口RAM，避免被展开为大规模寄存器；固定小常数地址乘法改成
移位加法，DSP数从311降到真正算术所需的306。

## 已准备好的Vivado入口

`code/hardware/vivado/`已经包含确定顺序的RTL source list、200 MHz OOC
约束和批处理脚本。默认part为`xczu3eg-sfvc784-2-e`。它无需板卡或PS block
design，会依次执行综合、`opt_design`、布局、物理优化和布线，输出综合/布线后
利用率、setup/hold时序、clock utilization、DRC、methodology和vectorless
power报告；布线后WNS为负会使脚本失败。

```bash
# L1/L2无状态稠密基线，可运行SL-TM和SL-HYB
code/hardware/vivado/run_ooc.sh /absolute/result/path/lean 0

# 带L1/L2状态容量的FS-TM公平对照
code/hardware/vivado/run_ooc.sh /absolute/result/path/full 1
```

Vivado前还无法给出四个数字：实际BRAM碎片、布线后WNS/Fmax、器件功率、板上
端到端延迟。当前脚本先解决前三项中的前两项并给出仅供初筛的vectorless功率。
板卡只在随后增加PS/AXI封装、装载参数和真实输入回放时需要；行为仿真和OOC
实现不要求AUP-ZU3连到这台服务器。
