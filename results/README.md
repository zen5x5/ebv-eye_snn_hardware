# 已确认结果

本目录按实验阶段组织，不再按调试日期或失败尝试分目录。
全部内容随Git仓库同步，文件压缩只改变存储格式，解压后的SHA-256与原始文件一致。

| 位置 | 内容 |
| --- | --- |
| summary.json | 当前四组完整时序、资源、速度和功耗总表 |
| latency_bram.json、latency_bram.csv | 五档已通过160MHz完整时序的核心时间与BRAM表，含新增L1～L4 |
| task_equivalence/ | 192模型配置核对、真实Saccade四档64回放、输入/参数/逐层参考、压缩mem和原日志；确认同配置两任务的核心时间与BRAM相同 |
| paper_metrics.csv | 可直接导入论文表格的四组指标 |
| model_audit.json | 四组共96份训练端点的来源与性能审计 |
| quantization/user10/ | 五组user10完整1194 test样本量化指标 |
| quantization/multiuser/ | L1～L2无状态双任务各24用户的参数、预测和总结果 |
| quantization/state_depth/ | 四档×两任务×24用户量化总表；192模型输入快照、141份新增参数/预测、51份复用来源、源码与独立核验 |
| quantization/l1_l4/ | Pursuit L1～L4无状态24用户完整test量化；模型/输入快照、定点参数、25,716份预测、源码与独立核验 |
| quantization/saccade_l1_l4/ | Saccade旧连续输出头L1～L4无状态24用户完整test量化；模型/输入快照、参数、28,389份预测、源码与独立核验 |
| rtl/ | 原四组64次整网回放，以及l1_l2_l3_l4_stateless/的16次新增回放与源码快照 |
| synthesis/ | 五组最终综合DCP及原始报告 |
| implementation/ | 五组最终布线DCP、时序/资源报告、接口审计和压缩功能电路 |
| power/ | 64份原始活动、仿真通过日志、功耗与活动覆盖报告 |
| power_vectorless/ | 五档默认活动功耗，双任务共用；含原始文本/XML、活动/操作条件、日志、执行脚本快照、汇总和独立核验；与power/分别保存 |
| provenance/ | 保留文件对应表、原始校验值和实际测量时的源码快照 |
| sparse_comparison.json | 旧稀疏设计保留的指标摘要，无运行产物 |

每个power/版本下保留sample_00～sample_15，包含inference.saif.gz、
saif_annotation.rpt.gz、power.rpt、power.xml、input_activity.rpt、bram_activity.rpt、
summary.json和simulation.log。原始SAIF可直接用于Vivado功耗分析，不需要再跑活动仿真。

当前五版本均完成160 MHz完整时序，前四版本完成user10输入活动估算。
五版本另完成统一160 MHz、25°C、典型工艺和标称电压的默认活动估算，
工具可信度均为Low；结果只追加到`power_vectorless/`，原64份真实输入功耗保持不变。
四版本的Pursuit及旧连续输出头Saccade各24用户完整test定点结果现已齐全。
Pursuit和Saccade第五档L1～L4也已补齐：24用户等权定点角误差分别0.832797°、1.342359°。
24用户功耗均值不列为本篇必做实验。
