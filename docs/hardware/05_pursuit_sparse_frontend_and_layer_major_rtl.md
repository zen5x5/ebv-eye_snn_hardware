# Pursuit稀疏前端与方案A整网RTL

## 结论

针对pursuit user15 eye0最终L1/L2跨bin无状态SNN，现已实现并逐值验证三种
完整硬件组织：

1. `Dense-TM`：既有稠密raster `SL-HYB`基线，L1/L2/L3的重复电流只算
   一次，深层仍按`t0`整网后接`t1`整网执行；
2. `Sparse-TM`：把L1/L2替换为事件驱动稀疏前端，后端保留原来的时间优先
   顺序和L3整层电流缓存；
3. `Sparse-LM-A`：使用同一稀疏前端，L3只卷积一次，L4～L8及H1/H2把
   `t0/t1`放在同一层、同一权重块内相邻执行，权重只装入一次。

按同一RTL发射口径，三seed完整validation的平均算术lane slot从
`14.592408M`降为`7.911147M`，减少45.79%。两条稀疏路线的发射量相同；
方案A进一步优化的是权重和状态局部性。它把每样本
运行期256-bit权重BRAM读取从Sparse-TM的11,892拍降至6,060拍，并把深层膜
从核心外部的片上状态RAM改为引擎内`t0`写一次、`t1`读一次。

对三个checkpoint的426个validation样本逐值检查共345,080,448个定点值，
稀疏前端、Pool后加常量和L3电流复用相对稠密定点参考均为零失配。seed2022
validation样本0又分别通过三套完整顶层RTL：Dense-TM、Sparse-TM和
Sparse-LM-A的周期为83,286、72,431和67,929拍，三者各检查159,408个值，
零失配。完整1,278个seed—样本组合的RTL校准周期均值为83,286、72,444.9、
67,942.9拍；方案A相对稠密基线减少18.42%，相对Sparse-TM减少6.21%。

这些仍是Vivado前结果。两条稀疏顶层的Yosys映射均为354个DSP48E2，占
AUP-ZU3所含360个DSP的98.3%，布局布线风险很高；两者的物理LUT理论界也都
跨过器件70,560个LUT的容量线，开源映射无法判断能否放入。200 MHz时延和
吞吐仅是满足5 ns约束后的条件换算，当前没有Vivado WNS、实现后资源、功率或
上板数据。

## 固定对象和算术合同

硬件对象、checkpoint和数据协议均没有变化：

- 模型为pursuit user15 eye0最终两阶段训练SNN，参数量96,792；
- 每样本是400个ROI事件，经2×2 SumPool后对应`2×96×128`整数计数图；
- 模型内部`T=2`，两步重复同一完整计数图；L1/L2无跨步膜状态，L3以后保留；
- S16权重、S40内部部分和、S24电流、S16膜、U8 MultiSpike、U9稠密输入接口；
- 逐输出通道2的幂scale，静态参数round-nearest-ties-away，
  `alpha(v)=floor(3v/4)`，完整S40 dot product后才允许钳位到S24；
- 三seed定点validation为`0.750±0.012°`，test为`0.767±0.021°`；同脚本
  CPU浮点test为`0.760±0.016°`。三条硬件路线改变执行顺序，不改变模型输出。

机器JSON中的`fixed_point_contract.accumulator_bits=24`沿用早期字段名，实际
指完整dot product之后的S24电流钳位位宽；所有RTL内部partial sum均为S40，
不能把该字段解释成24-bit部分和。

本报告中的“原版”特指既有完整RTL中最快的稠密raster `SL-HYB`，在比较表中
记为`Dense-TM`。它不是此前周期更长的`SL-TM`，也不是全状态`FS-TM`。

## L1/L2事件驱动前端

### 从400个事件到有序包流

原模型先将400个事件聚合成计数图，已经不使用窗内时间顺序。硬件把每个计数
重新展开成相同数量的单位事件，并按`row, column, polarity`排序；重复地址保留，
所以它与形成计数图后做卷积严格等价。稀疏核心只要求row非递减，同一row内的
column和polarity顺序不影响结果。排序属于输入接口合同；现有RTL从已排序的
400事件开始计时，尚未计入传感器、ROI筛选、排序器或PS/AXI传输。

数据路径为：

```text
400 sorted unit events
  -> L1: 9 taps/event x 8 outputs, add-only
  -> nonzero U8 L1 packets
  -> L2: 3 taps/cycle x 16 outputs, count x S16 weight
  -> complete 16 x 24 x 32 L2 spike tensor
  -> regular dense L3 backend
```

L1将一个单位事件同时散射到3×3支持域和8个输出通道，使用72个LUT加法lane，
不使用乘法器。边界外tap直接禁止；所以每样本平均有效加法为28,679.3，略低于
`400×9×8=28,800`。L1只发出非零的`(address, channel, U8 count)`记录，三seed
validation平均2,554.4个包。

L2输入已经是MultiSpike整数。若将`n*w`展开为`n`次加法，会重复发射相同权重；
因此实现选择每个非零count做一次乘权重。L2每拍处理3个kernel tap和16个输出
通道，共48个DSP乘法lane；三seedvalidation平均执行364,499.3个有效MAC。
L1和L2均用三行、九bank的S40部分和环形缓存，按row完成后立刻做横向2×2
MaxPool并清除旧行，不清空整幅feature map。

### Pool、BN折叠常量和L2默认平面

设某输出通道的折叠权重卷积结果为`z_i`，同一2×2池化窗四个位置共享常量
`b'`，则：

```text
max_i(z_i + b') = max_i(z_i) + b'
```

因此RTL先对S40卷积部分和做MaxPool，再只加一次S24折叠常量，随后钳位S24
并做无状态LIF。该顺序与原模型逐位置加常量再MaxPool逐值一致，同时把常量加法
数量降到四分之一。这里不能在MaxPool前把每个位置先钳位到S24；钳位仍只发生
在完整dot product、Pool和常量相加之后。

三seed的L1零突触输入均不发脉冲，所以未触及的L1位置可以省略。L2不同：三个
checkpoint分别有2、3、4个通道仅靠折叠常量便会发出1或2个脉冲。L2的突触
累加仍只由非零L1包触发，但finalize阶段必须遍历全部768个空间位置、每处同时
生成16个通道，写成完整L2平面。三seedvalidation平均有3,708.7个非零L2
通道记录，其中约1,315.5个来自未被突触触及位置的默认平面。把未触及位置直接
置零会改变L3输入和最终预测。

## 三种完整组织

### Dense-TM：稠密正确性基线

Dense-TM就是既有`pursuit_snn_accelerator_top.sv`的`SL-HYB`：288-lane
共享卷积核心扫描完整空间raster，零输入同样占用lane和周期。L1/L2/L3电流各
算一次，L3的4,608个Pool后S24电流写入13.5 KiB整层缓存；执行完`t0`的
L3～H2后，重放L3电流，再执行`t1`的L3～H2。它保留为位精确、状态RAM和
时间复用对照，不再代表最终稀疏SNN映射。

### Sparse-TM：只替换浅层计算

`pursuit_sparse_time_major_accelerator_top.sv`将Dense-TM的L1/L2换成上述
事件驱动前端。L2完整平面送入原288-lane深层卷积核心；L3仍只算一次并写入
13.5 KiB整层电流缓存，L4～H2仍按先完成`t0`、再完成`t1`的顺序运行。

这个实现回答“只做L1/L2稀疏，其他组织保持原样”能得到多少收益。它与方案A
使用相同的浅层算术、L2默认平面和354个名义DSP，因此两者的差异来自深层
执行顺序，而不是模型稀疏度或PE数量。

### Sparse-LM-A：层内相邻两步

`pursuit_sparse_layer_major_accelerator_top.sv`使用一个物理288-lane卷积核心
顺序处理L3～L8，并没有为每层复制核心。对于L4～L8，每个
`8 output × 4 input × 3×3`权重tile只从BRAM装入一次，先完整扫描`t0`和
`t1`两个feature map，分别保留两套S40部分和，然后才进入下一tile。`t0`膜在
引擎局部RAM保存并直接供`t1`使用；样本结束后不需要把`t1`最终膜写回核心外部
的片上状态RAM。

L3的输入在两个逻辑步完全相同，所以只做一次卷积。每个8通道tile保留
`192×8`个S24电流，即4.5 KiB局部容量，再依次做`t0/t1` LIF。一个样本仍会
写、读共4,608个S24电流，但不再需要核心外部的13.5 KiB整层片上缓存。

H1/H2使用同一个16-DSP成对引擎。每个256-bit权重beat读一次，先累加`t0`
输入，下一拍复用该beat累加`t1`；16个LIF lane再连续完成两步。readout继续
使用2个DSP。

方案A是单核心的层内时间相邻执行，不是跨层波前：L3、L4等层之间没有同时
计算，也没有把288个lane拆成分别负责`t0/t1`的两组，当前样本与下一样本也
没有重叠。它的收益来自权重只装一次、膜保持局部和避免整层L3缓存，而不是
并行执行不同层。

## 算术量与RTL发射lane

硬件主比较采用同一`scheduled/issued lane slot`口径：只要固定宽度数据路径在
该拍发射了一个算术lane便计一次，包括same-padding边界被屏蔽的tap、L1稠密
核心空出的输入通道lane和H1/H2输出tile的填充lane。下表是全部1,278个
seed—validation样本的汇总；Sparse-TM与Sparse-LM-A使用相同数据路径，故
发射量相同。

| 部分 | Dense-TM | Sparse-TM | Sparse-LM-A |
|---|---:|---:|---:|
| L1 | 3,538,944个乘法lane slot | 28,800个加法lane slot | 28,800个加法lane slot |
| L2 | 3,538,944个乘法lane slot | 平均367,826.9个乘法lane slot | 平均367,826.9个乘法lane slot |
| L3～L8 | 7,451,136个乘法lane slot | 7,451,136个乘法lane slot | 7,451,136个乘法lane slot |
| H1/H2/readout | 63,384个乘法lane slot | 63,384个乘法lane slot | 63,384个乘法lane slot |
| 每样本总计 | 14,592,408 | 平均7,911,146.9 | 平均7,911,146.9 |
| 相对Dense-TM | — | −45.79% | −45.79% |

其中Sparse L1固定为`400×9×8=28,800`个lane slot；Sparse L2每个非零L1包
固定发射`9×16=144`个lane slot，三seed平均2,554.4个包。该口径衡量控制器
向算术数据路径发出的固定宽度工作，不表示被屏蔽lane的真实toggle，也不包含
地址路由、部分和RAM、Pool、常量加法、LIF、FIFO和控制，不能由45.79%直接
推导功耗或周期减少45.79%。

两条稀疏路线在全部1,278个样本上的scheduled-lane分布相同：最小
7,651,896，中位7,920,744，p95为8,014,840.8，最大8,080,872；变化只来自
各样本L1非零包数。

为和此前报告衔接，另保留既有常规算法账：Dense-TM按实际模型通道数、但让
same-convolution的每个输出都计完整3×3 kernel，为12,818,616个MAC；稀疏路线
的L1/L2只计in-bounds有效更新，分别平均28,679.3次加法和364,499.3次MAC，
再加规则后端7,510,200次MAC，得到7,903,378.6个混合操作，名义少38.34%。
它在浅层稀疏更新和规则层使用两种定义，只用于延续旧账，不能作为硬件主比较。
无论哪一口径，一次S16加法和一次U8/U9×S16 MAC也不等价于相同面积或能量。

## 存取量

以下为既定存储接口下的每样本逻辑访问。activation read beat混合了不同宽度的
卷积向量与linear/readout逐元素接口，write beat也混有64/128/256-bit接口，
因此只在同一列定义下比较，不把不同beat直接换算成字节或带宽。前端稀疏包和
局部S40部分和另列，不混入BRAM beat。机器JSON中的字段名
`activation_BRAM_read_beats_mixed_36b_8b`是沿用的简写，实际还包含深层32-bit
向量及readout时间和接口；数字本身是下表所述的全部逻辑read beat。

| 访问项 | Dense-TM | Sparse-TM | Sparse-LM-A |
|---|---:|---:|---:|
| 输入入口 | 24,576个U9计数值 | 400个单位事件包 | 400个单位事件包 |
| L1非零包写，三seed均值 | 0 | 2,554.4 | 2,554.4 |
| 不同宽度逻辑activation读beat | 54,420 | 29,844 | 29,844 |
| activation BRAM写beat | 6,634 | 2,794 | 2,794 |
| 运行期权重BRAM读，256-bit beat | 11,982 | 11,892 | 6,060 |
| 核心外部片上状态RAM膜元素读/写 | 8,136 / 16,272 | 8,136 / 16,272 | 0 / 0 |
| 引擎局部膜元素读/写 | 0 / 0 | 0 / 0 | 8,136 / 8,136 |
| L3整层外部电流容量 | 13.5 KiB | 13.5 KiB | 0 |
| L3 tile局部电流容量 | 0 | 0 | 4.5 KiB |
| L3电流每样本payload读/写 | 13.5 / 13.5 KiB | 13.5 / 13.5 KiB | 13.5 / 13.5 KiB，局部 |

相对Dense-TM，两条稀疏路线的activation BRAM读、写分别减少45.16%和
57.88%。Sparse-LM-A相对Sparse-TM再将运行期权重BRAM读取减少49.04%。
前端1296个S16权重在模型加载时写入本地寄存器/存储一次；每样本L1和L2
分别发生28,800和平均367,826.9次本地权重word读取，后者包含边界tap发射，
所以与排除无效padding的364,499.3个有效MAC略有不同。

方案A没有消灭L3电流和深层膜的物理访问，而是缩小作用域：L3电流从整层外部
缓存变成tile局部缓存，深层膜从顶层共享状态存储变成成对引擎内部存储。能否
映射成更少BRAM、更多LUTRAM，及其真实功耗变化，只能由Vivado实现报告和活动
回放确认。

## 周期、条件时延和吞吐

seed2022 validation样本0的完整RTL实测用于校准固定后端：Sparse-TM前端
15,259拍、后端57,172拍，总计72,431拍；Sparse-LM-A前端相同，后端
52,670拍，总计67,929拍。方案A后端的主要组件为L3 11,592拍，L4～L8分别
13,171/8,183/5,213/5,213/5,213拍，H1/H2为3,928/142拍，readout为16拍；
整合控制边界相对组件简单相加少1拍。

完整validation周期分布的生成方式是：对每个真实样本逐周期回放稀疏前端
ready/valid FSM，再加上由上述完整RTL样本校准的固定后端。seed2022样本0的
软件周期模型与两套整网RTL完全一致；其余1,277个组合是RTL校准模型结果，
不是逐个运行整网RTL仿真。

| 路线 | mean cycles | median | p95 | max | 相对Dense-TM | 若200 MHz：mean ms | 若200 MHz：sample/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense-TM | 83,286.0 | 83,286 | 83,286 | 83,286 | — | 0.416430 | 2,401.4 |
| Sparse-TM | 72,444.9 | 72,645 | 74,417.5 | 75,491 | −13.02% | 0.362225 | 2,760.7 |
| Sparse-LM-A | 67,942.9 | 68,143 | 69,915.5 | 70,989 | −18.42% | 0.339715 | 2,943.6 |

方案A相对Sparse-TM平均再减少4,502拍，即6.21%；对应速度比为1.066倍。
相对Dense-TM的平均速度比为1.226倍。表中的200 MHz数值假设样本顺序执行、
没有跨样本重叠且Vivado实际满足5 ns时钟；它只是accelerator kernel服务能力，
不包含400事件到达时间，也不等于系统gaze更新率。

## 正确性证据

软件位精确回放覆盖三个最终checkpoint的全部426个validation物理窗：

- 每seed检查115,026,816个L1～readout电流、膜、脉冲和输出值；三seed合计
  345,080,448个值，失配为0；
- 分别检查L1/L2稀疏累加、`Pool(z+b)=Pool(z)+b`、L3电流只算一次及最终
  prediction完全一致；
- 三seed所有层的S24电流和U8脉冲饱和计数均为0。

完整顶层RTL用seed2022 validation样本0回放400个排序事件，其中361个唯一
极性—位置、最大重复计数4。Sparse-TM和Sparse-LM-A均接受400个事件，L1发出
2,575个非零包，L2接收2,575个包；每套顶层检查5,866笔trace transaction、
共159,408个值，失配和range error均为0。两套顶层均通过Icarus Verilog 13.0
仿真和Verilator 5.052 `--Wall`。Dense-TM的同一样本既有结果也是159,408个值、
零失配。

## Yosys资源与AUP-ZU3风险

名义算术lane如下；它说明设计意图，不替代综合结果。

| 路线 | L1 | L2 DSP | 深层卷积DSP | dense DSP | readout DSP | 总DSP | XCZU3EG占比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense-TM | 共用深层核心 | 共用深层核心 | 288 | 16 | 2 | 306 | 85.0% |
| Sparse-TM | 72个LUT add lane | 48 | 288 | 16 | 2 | 354 | 98.3% |
| Sparse-LM-A | 72个LUT add lane | 48 | 288 | 16 | 2 | 354 | 98.3% |

稠密基线让同一个288-DSP核心先后执行L1、L2和深层，所以L1/L2不会增加物理
DSP。两条稀疏顶层将L1移入LUT，同时为异步包流L2保留独立48-DSP核心，再接
原288-DSP深层核心，因此总数反而比稠密基线多48个。这个选择使前端和规则
后端边界清楚、控制简单，但也是当前只剩6个DSP余量的直接原因。

完整顶层均以Yosys 0.69执行`xcup`、无I/O pad的primitive mapping。结果如下；
BRAM36等效按`RAMB36E2 + RAMB18E2/2`计算。

| 路线 | DSP48E2 | RAMB36E2 | RAMB18E2 | BRAM36等效 | FF | CARRY4 |
|---|---:|---:|---:|---:|---:|---:|
| Dense-TM lean | 306 | 93 | 40 | 113 | 34,048 | 4,226 |
| Sparse-TM | 354 | 93 | 48 | 117 | 36,941 | 6,656 |
| Sparse-LM-A | 354 | 40 | 81 | 80.5 | 36,124 | 6,352 |

LUT必须分开解释。`logic LUT primitive`是Yosys网表中LUT1～LUT6的数量；
兼容的小函数可由一个物理LUT6_2打包，而分布式RAM又占物理LUT容量。下界假设
最大兼容打包，上界假设logic primitive完全不共享，因此二者都不是Vivado
utilization。

| 路线 | logic LUT primitive | LUTRAM等效物理LUT容量 | 物理LUT理论界 | 对70,560容量的结论 |
|---|---:|---:|---:|---|
| Dense-TM lean | 51,100 | 40 | 25,590～51,140 | 理论界低于容量，仍未证明可布局 |
| Sparse-TM | 87,952 | 8,456 | 52,432～96,408 | 容量线落在界内，Yosys无法判定 |
| Sparse-LM-A | 77,845 | 17,288 | 56,211～95,133 | 容量线落在界内，Yosys无法判定 |

Sparse-TM的LUT1～6依次为`0/24,766/9,188/8,787/29,091/16,120`，LUTRAM
为646个RAM32M16、51个RAM64M8和2,880个RAM64X1S。Sparse-LM-A的
LUT1～6为`8/22,741/7,168/10,289/22,274/15,365`，LUTRAM为646个
RAM32M16、1,155个RAM64M8和2,880个RAM64X1S。

方案A将BRAM36等效用量从117降至80.5，减少31.20%，logic LUT primitive也
从87,952降至77,845；但局部成对状态使LUTRAM容量从8,456升至17,288个物理
LUT。它是“少块RAM、更多分布式RAM”的映射取舍，不能只看logic primitive
下降便断言更容易放入。两条稀疏路线均只剩6个DSP余量，真实LUT打包、BRAM
碎片、拥塞、WNS和200 MHz可行性必须由Vivado place-and-route确定。

## 与已有工作的比较边界

Tan 2025是最接近的直接gaze专用实现：UMC 40 nm ASIC、500事件/pitch 200、
`2×120×220`输入、约91k参数、11.5M MAC、W8/A20、144 MAC、252.19 KB SRAM；
在200 MHz报告1,252 FPS，即约0.799 ms，实测15.2 mW和12.7 µJ/次估计，量化
平均/最佳误差为1.15/0.91°。本项目条件0.340 ms和约7.911M个RTL发射lane
slot不能据此声称快于Tan：任务窗口、输入、误差协议、算术定义和FPGA/ASIC
技术均不同，且
本项目尚无时序收敛和功耗证据。

FireFly 2023在同为XCZU3EG器件族的Ultra96v2上使用288 DSP、162 BRAM、
约15k LUT并达到300 MHz，证明该器件族可以容纳大DSP阵列；但它执行INT8权重、
binary spike分类，并在DSP内打包突触，不能把其峰值吞吐套到本项目S16/U8
MultiSpike。FireFly v2的256 DSP、103 BRAM和300/600 MHz双时钟说明层内
时间复用有效，但其状态组织和本项目不同。FireFly-S依赖4 bit及85%～95%权重
稀疏，ESDA依赖8 bit submanifold sparse convolution和更大的XCZU9EG；当前
模型权重几乎没有零，普通same-padding卷积也不能直接采用它们的零DSP或坐标
传播结果。完整资源、功率证据类型和任务边界见硬件文献审计。

## 复现入口和下一证据

主要入口为：

- `code/hardware/evaluate_pursuit_sparse_hybrid.py`：三seed完整validation的
  位精确、算术、访存和周期模型；
- `code/hardware/run_sparse_time_major_golden_check.py`：Sparse-TM整网回放、
  lint和Yosys映射；
- `code/hardware/run_sparse_layer_major_golden_check.py`：Sparse-LM-A整网回放、
  lint和Yosys映射；
- `code/hardware/vivado/run_sparse_ooc.sh`：两套稀疏顶层在
  `xczu3eg-sfvc784-2-e`、5 ns约束下的OOC实现。

正式validation、整网RTL、开源映射和manifest统一归档于
`/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/rtl_sparse_three_schedule_pre_vivado_20261008/`。

Vivado前的两套稀疏整顶层Yosys结果已经归档。下一步需要在Vivado分别跑
`Sparse-TM`和`Sparse-LM-A`，报告post-route WNS、LUT/FF/BRAM/DSP和层次资源；
功率若只来自vectorless XPA必须标为估计。真实论文功率应使用validation回放生成
的SAIF/VCD或板级测量。板卡只在随后PS/AXI回放和真实功率测量阶段需要，OOC
综合、布局布线和时序不需要实体AUP-ZU3。
