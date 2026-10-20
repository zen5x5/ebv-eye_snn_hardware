# 五档默认活动功耗

统一使用Vivado 2025.2.1、XCZU3EG、160 MHz、25°C结温、典型工艺及标称电压。
输入为`results/implementation/<variant>/post_route.dcp`，未改动检查点。
`reset_switching_activity -all`清除活动标注，`-default`恢复默认活动；无SAIF、无手动活动设置。
普通输入默认高电平概率0.5、每周期翻转率12.5%；复位由工具默认置为无效。

Pursuit和旧连续xy输出头Saccade在同一深度共用电路，权重在运行时加载。
本次未提供任务相关活动，五份报告覆盖两任务的五档默认估算。

| 无状态层 | 总功率估算 | 动态功率 | 器件静态功率 |
| --- | ---: | ---: | ---: |
| 无 | 0.733 W | 0.511 W | 0.221 W |
| L1 | 0.595 W | 0.374 W | 0.221 W |
| L1～L2 | 0.602 W | 0.381 W | 0.221 W |
| L1～L3 | 0.604 W | 0.383 W | 0.221 W |
| L1～L4 | 0.691 W | 0.470 W | 0.221 W |

工具可信度均为Low，并提示部分内部复位/控制信号的默认活动不理想。
真实输入活动的原四档64份结果继续位于`results/power/`，保持原始数值及文件。
本目录的能量字段只按默认估算功率×核心时间计算；功率含芯片静态部分及评估接口，
输入加载时长、主机和板卡外围未纳入该能量。

- `summary.json`、`summary.csv`：五档原始功率、核心时间、派生能量与输入/报告校验值。
- `<variant>/reports/`：原始文本/XML、输入和BRAM使能活动、操作条件及明确设置。
- `<variant>/console.log`、`result.json`：实际执行记录和单档结果。
- `sources/`：本次执行的脚本原文；历史绝对路径保留用于追溯。
- `verification.json`：独立核验结果，检查报告一致性、默认活动来源和检查点未变。

只读核验：`python scripts/verify_vectorless_power.py results/power_vectorless`。
重新计算：`python scripts/power_vectorless.py --output-dir runs/vectorless_power_new --workers 2`。
