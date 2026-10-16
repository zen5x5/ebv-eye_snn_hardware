# 当前仓库验证

日期：2026-10-16。以下检查从本次克隆和更新后的仓库执行。

| 检查 | 结果 |
|---|---|
| 可移植资产校验 | 116 条资产校验通过 |
| 当前整网 XSim 回放 | 159,408 个值零失配，104,005 拍 |
| 输入与中间包 | 400 个事件，L1/L2 各传递 3,315 个非零包 |
| 仿真阶段计数 | 前端 51,305；共享卷积 48,609；全连接 4,074 拍 |
| 当前 13 个综合 RTL 源文件与已布线快照比较 | SHA-256 一致 |
| Python 语法 | 49 个文件通过 |
| Shell 语法 | 两个 Vivado 入口通过 bash -n |
| Tcl 脚本接口 | 6.250/5.000 ns 的源文件、时钟设置和运行记录检查通过 |
| Markdown 本地文件链接 | 全部有效 |
| 当前架构 SVG | XML 解析通过 |
| 仓库清单 | MANIFEST.sha256 逐文件检查通过 |

XSim 回放使用随仓库提供的 user17/seed2022 真实输入、权重和逐层标准数值；工具为 Vivado 2025.2.1 的 xvlog、xelab、xsim。复现命令：

```bash
source /opt/amd/2025.2.1/Vivado/settings64.sh
python -B scripts/replay_current.py --output-dir build/replay
```

[导出后重新回放的记录](evidence/current/rtl_replay/export_report.json)、[仿真日志](evidence/current/rtl_replay/simulation.log)与[此前逐状态计数](evidence/current/rtl_replay/report.json)分别保存。

本次整理使用此前已经完成的 Vivado 布线报告，检查当前 RTL 与其源码快照一致。本次新增验证覆盖功能回放和运行入口；完整布局布线留待需要重建 checkpoint 时执行。

原 2026-10-13 user15 的回放和三路线数据保存在 evidence/rtl_three_route/，对应完整源代码和向量可从提交 3fef30e 取得。

