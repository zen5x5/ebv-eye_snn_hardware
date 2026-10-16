# 仓库工作说明

这是独立 Git 仓库 zen5x5/ebv-eye_snn_hardware。用户于 2026-10-16 明确要求在本机克隆、更新、精简并推送。父项目本身保持无 Git 管理。

- 先读根 README，再按任务读 docs/hardware/current_design.md、docs/hardware/comparison.md 和 evidence/README.md。
- 当前顶层：code/hardware/rtl/pursuit_sparse_layer_major_accelerator_top.sv；部署候选为 Pursuit user17/seed2022，标准输入和权重在 evidence/current/。
- 当前整网样本：159,408 值零失配，104,005 拍。160 MHz OOC 布线时序通过，核心换算 0.65003125 ms；200 MHz setup WNS 为 -1.109 ns。
- 当前资源以 evidence/current/vivado/reports/post_route_utilization.rpt 为准：61,657 LUT、24,835 FF、124 RAMB36 + 54 RAMB18、308 DSP。L1 为 24 路加法；每事件 3 次更新，每次包含 1 拍同步读取。
- 功能仿真、布线时序、工具功率估算和板上实测分别标注。当前已有 1.835 W 无真实活动记录的工具估算；板上推理及功率测量尚待完成。
- evidence/rtl_three_route/、evidence/fixed_point_reference/ 和 docs/history/ 保存 user15 的早期对照，其 67,929 拍及 354 DSP 属于历史版本。
- 48 模型正式量化结果保留；Pursuit/Saccade 等权定点误差为 0.825481°/1.326037°，Saccade 使用连续坐标。当前 user17 单用户定点误差为 0.488592°。
- 标准数值和权重文件逐字节保留。可移植 manifest 只重写路径及指向重写 manifest 的校验值，外部训练来源保留出处。
- scripts/replay_current.py 使用 Vivado 自带的 XSim；scripts/verify_evidence.py 检查标准资产；MANIFEST.sha256 核验仓库文件。
- 解释按实际顺序展开，用清楚、具体的中文；公式使用纯文本。用户询问概念时再展开额外背景。
- 按已经确认的数据、接口与条件实现，针对已查明的错误修正。

