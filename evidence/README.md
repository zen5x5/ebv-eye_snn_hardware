# 机器证据

## 当前实现

当前模型为 Pursuit user17/seed2022，第 193 轮 checkpoint。参数和标准数值在复制时逐文件检查，数据文件内容保持一致。

| 位置 | 内容 |
|---|---|
| current/golden_vectors/ | 16 个样本的标准数值，以及回放样本 0 的 MEM、参数 |
| current/hardware_image/ | 256 位权重流、通道常量和小数位设置 |
| current/quantization/ | user17 完整 1,364 个 test 样本的量化报告 |
| current/rtl_replay/ | 2026-10-16 的本机整网回放和 L1 逐状态计数 |
| current/vivado/reports/ | 最新布线后资源、200/160 MHz 时序、工具功率报告 |
| current/metrics.json | 当前指标与数值来源 |
| current/update_provenance.json | 此次代码来源、资产校验与旧文件清理记录 |

manifest 和 layout 中的资产路径使用仓库相对路径；外部训练记录使用 external:// 标记原来源。MEM、NPZ 和权重镜像逐字节保留。

复核资产：python -B scripts/verify_evidence.py。复核整网：python -B scripts/replay_current.py --output-dir build/replay。

## 历史及正式多用户结果

formal_quantization/ 保留 48 个模型的正式软件定点评估。fixed_point_reference/ 和 rtl_three_route/ 保留原 user15 的三种硬件组织对照；其中 67,929 拍和 Yosys 映射属于 2026-10-13 的历史结构。

旧 user15 标准向量及权重镜像可从 Git 提交 3fef30e 取得，与该提交的代码共同复现早期对照。

Vivado 的 DCP 大文件保存在原工作目录，仓库提供报告和运行入口。当前功率为工具估算，真实活动记录和板上测量留待接入。

