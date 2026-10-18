# 已确认结果

本目录按实验阶段组织，不再按调试日期或失败尝试分目录。
全部内容随Git仓库同步，文件压缩只改变存储格式，解压后的SHA-256与原始文件一致。

| 位置 | 内容 |
| --- | --- |
| summary.json | 当前四组完整时序、资源、速度和功耗总表 |
| paper_metrics.csv | 可直接导入论文表格的四组指标 |
| model_audit.json | 四组共96份训练端点的来源与性能审计 |
| quantization/user10/ | 四组user10完整1194 test样本量化指标 |
| quantization/multiuser/ | L1～L2无状态双任务各24用户的参数、预测和总结果 |
| rtl/ | 64次整网回放原日志与逐层核验汇总 |
| synthesis/ | 四组最终综合DCP及原始报告 |
| implementation/ | 四组最终布线DCP、时序/资源报告、接口审计和压缩功能电路 |
| power/ | 64份原始活动、仿真通过日志、功耗与活动覆盖报告 |
| provenance/ | 保留文件对应表、原始校验值和实际测量时的源码快照 |
| sparse_comparison.json | 旧稀疏设计保留的指标摘要，无运行产物 |

每个power/版本下保留sample_00～sample_15，包含inference.saif.gz、
saif_annotation.rpt.gz、power.rpt、power.xml、input_activity.rpt、bram_activity.rpt、
summary.json和simulation.log。原始SAIF可直接用于Vivado功耗分析，不需要再跑活动仿真。

当前四版本均完成160 MHz完整时序和user10输入活动估算。L1～L2无状态的24用户定点
结果已齐；其他三版本目前覆盖user10。24用户功耗均值不列为本篇必做实验。
