# Pursuit无尾窗数据协议与ANN对照

## 1. 范围

本文档记录当前pursuit主线的数据协议、在线事件重组、正式ANN和Compact90K ANN对照。SNN的alpha对照单独放在[02文档](02_snn_fixed_alpha_comparison.md)。

## 2. 数据与在线重组

### 2.1 物理窗口

数据为EBV-Eye user15 eye0 pursuit。ROI为`top=0, left=32, height=192, width=256`，经2×2 sum pooling形成`2×96×128`双极性计数图。每个物理窗含ROI内连续400事件。

- 基础train/validation/test：1678/426/426；
- 仅train相邻完整窗增加stride200：1283；
- 最终物理train/validation/test：2961/426/426；
- 候选生成时直接舍弃片段尾部不足400事件的余量，尾窗回填数为0；
- validation和test不做stride或在线合成，三个split的原始事件不交叠。

标签是窗口末事件时刻线性插值的屏幕刺激点坐标。这是同用户、同轨迹内的窗口留出，不是留出用户或留出整个target。

### 2.2 全物理train来源的在线重组

2961个物理train窗全部进入source/partner池。伙伴必须与source属于同一`group_index`、标签欧氏距离不超过32 px，且原始起始事件索引差至少400。优先使用差值大于400的非相邻伙伴；无此类伙伴时才允许差值恰为400的相邻非重叠窗，差值小于400的来源禁止组合。

- 2948个source可合成：2903个有优先伙伴，45个使用相邻fallback，13个无伙伴；
- 可选partner共2926个；
- 合格source中1274个为stride窗，可选partner中1267个为stride窗。

每个epoch，每个合格source选择一个伙伴，再从两窗各无放回抽取200个事件出现次数，逐像素相加为恰含400事件的静态计数图；标签取两窗row/col均值。伙伴与事件子集随epoch刷新，并由seed、epoch和source索引唯一决定。合成图不再递归参与合成。

每轮共2961个物理图与2948个合成图，即5909个训练输入。batch16且`drop_last=True`时每轮实际使用5904个输入、更新369次；200轮共73,800次更新。报告中的train指标只在2961个物理train图上评价。

## 3. ANN模型

### 3.1 结构对照

两种模型都使用八层3×3 Conv（`bias=False`）、BN和ReLU，并在L1～L5的ReLU后做2×2 MaxPool。区别只在后四层通道和回归头宽度。

| 项目 | 正式ANN | Compact90K ANN |
|---|---:|---:|
| 卷积通道 | `2→8→16→24→32→40→40→40→40` | `2→8→16→24→32→36→36→36→36` |
| L5～L8特征图 | `40×3×4` | `36×3×4` |
| 回归头 | `480→60→12→2` | `432→48→12→2` |
| 卷积＋BN参数 | 66,864 | 57,472 |
| 回归头参数 | 29,616 | 21,396 |
| 总参数 | 96,480 | 78,868 |
| 相对正式ANN | 100% | 81.75% |

若只把Compact90K的输入改为`2×120×220`并保持五次池化，末端特征为`36×3×6=648`，回归头变为`648→48→12→2`，参数量为89,236。该数字是结构计算值，本次没有训练120×220输入。

### 3.2 训练与评价

两种ANN都使用seed2021、2022、2023，各从头训练200轮。数据、在线重组、每轮369次更新和评价集完全相同。Adam初始学习率0.001，batch16，train `drop_last=True`，无早停。ReduceLROnPlateau按validation样本平均角误差调度，factor0.5、patience9、threshold0、最小学习率1e-5；按最低validation角误差保存checkpoint。loss是还原到屏幕像素尺度的二维平方距离，再除以半屏对角线平方。

## 4. ANN三seed汇总

下表为均值±样本标准差。

| 指标 | 正式ANN | Compact90K ANN |
|---|---:|---:|
| 像素误差（px） | train: 1.77±0.17<br>validation: 11.03±0.78<br>test: 11.73±0.13 | train: 1.84±0.24<br>validation: 12.04±0.32<br>test: 11.94±0.48 |
| 角误差（°） | train: 0.092±0.009<br>validation: 0.537±0.040<br>test: 0.585±0.005 | train: 0.094±0.013<br>validation: 0.589±0.019<br>test: 0.599±0.030 |
| 最佳轮次 | 163 / 175 / 200 | 190 / 189 / 189 |

Compact90K减少17,612个参数，即18.25%。按同seed配对，它相对正式ANN的validation误差增加`1.02±1.09 px / 0.051±0.059°`，三个seed都变差；test增加`0.22±0.44 px / 0.015±0.028°`，其中seed2022改善，seed2021/2023变差。因此它给出了较小模型下接近的test均值，但不能表述为无精度代价压缩。

## 5. ANN逐seed结果

### 5.1 正式ANN

| seed | 像素误差（px） | 角误差（°） | 最佳轮次 | 训练墙钟 |
|---|---:|---:|---:|---:|
| 2021 | train: 1.93<br>validation: 10.13<br>test: 11.67 | train: 0.100<br>validation: 0.491<br>test: 0.584 | 163 | 24.12 min |
| 2022 | train: 1.79<br>validation: 11.40<br>test: 11.65 | train: 0.093<br>validation: 0.558<br>test: 0.580 | 175 | 20.04 min |
| 2023 | train: 1.60<br>validation: 11.55<br>test: 11.87 | train: 0.082<br>validation: 0.563<br>test: 0.591 | 200 | 17.99 min |

### 5.2 Compact90K ANN

| seed | 像素误差（px） | 角误差（°） | 最佳轮次 | 训练墙钟 |
|---|---:|---:|---:|---:|
| 2021 | train: 1.68<br>validation: 12.41<br>test: 12.31 | train: 0.086<br>validation: 0.610<br>test: 0.628 | 190 | 24.12 min |
| 2022 | train: 1.72<br>validation: 11.80<br>test: 11.40 | train: 0.088<br>validation: 0.574<br>test: 0.568 | 189 | 27.43 min |
| 2023 | train: 2.11<br>validation: 11.92<br>test: 12.13 | train: 0.110<br>validation: 0.581<br>test: 0.602 | 189 | 20.81 min |

两种结构的六份结果均有200轮history、最佳checkpoint、`configuration.json`和`summary.json`；checkpoint轮次与summary一致。墙钟受共享GPU负载影响，只作执行审计。

## 6. 结果边界

- validation/test不参与在线重组，跨split原始事件不交叠。
- 在线重组是两个同train组静态窗的事件子集混合，不包含两来源间的连续时间顺序。
- Compact90K与正式ANN是同数据、同增强、同训练预算的结构对照；没有用test选择结构或checkpoint。
- 当前数字只支持user15 eye0、同轨迹窗口留出的结论，不代表跨用户或未见target泛化。

## 7. 正式产物

- 数据：`/data/xuzx/data/ebv-eye/cache/formal_eye0/pursuit/fixed_event_window_400_stride200_96x128/user15/eye0/`
- 正式ANN：`$EBV_EYE_RESULT_ROOT/pursuit/fixed_event_window/ann/user15/eye0/no_tail_all_physical_source_recombination_epoch200/`
- Compact90K ANN：`$EBV_EYE_RESULT_ROOT/pursuit/fixed_event_window/ann/user15/eye0/no_tail_all_physical_source_recombination_compact90k_epoch200/`

## 8. fs_cc迁移单seed执行核验

2026-10-06在LIACS fs_cc的NVIDIA L40S上按相同正式缓存、模型、增强、seed2021和200轮合同重跑正式ANN。新环境为Python3.11.16、torch2.13.0+cu130；GPU与软件执行环境均不同，因此本节只检查迁移后的性能量级和墙钟，不替换原3090三seed结果，也不把差异归因于单一因素。

| 环境 | 像素误差（px） | 角误差（°） | 最佳轮 | 墙钟 |
|:--|--:|--:|--:|--:|
| 原3090记录 | train: 1.93<br>validation: 10.13<br>test: 11.67 | train: 0.100<br>validation: 0.491<br>test: 0.584 | 163 | 24.12 min |
| fs_cc L40S | train: 1.798<br>validation: 11.592<br>test: 11.118 | train: 0.0924<br>validation: 0.5669<br>test: 0.5533 | 159 | 526.58 s，即8.78 min |

L40S相对旧记录的train/validation/test角误差差值为`−0.0076/+0.0759/−0.0307°`，方向不一致，说明同seed未逐项复现；总体仍在相同误差量级。按旧报告四舍五入后的24.12分钟计算，墙钟缩短约63.6%，约为2.75倍速度。200轮history连续，最佳checkpoint在CPU反序列化后为epoch159，错误日志为空。
