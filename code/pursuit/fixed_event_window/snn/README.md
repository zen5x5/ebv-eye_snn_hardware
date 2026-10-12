# Pursuit 当前SNN固定alpha对照

输入显式为`B×T×2×96×128`，正式基线把同一完整400事件计数图重复输入T=2。

- 八层通道：`2→8→16→24→32→40→40→40→40`。
- L1～L5：`Conv3×3(bias=False)→BN→MaxPool2×2→LIF`。
- L6～L8：`Conv3×3(bias=False)→BN→LIF`。
- 脉冲MLP：`480→60→12`，两个Linear带bias并后接LIF。
- 连续读出：12维末层脉冲沿T求平均，再做`Linear(12,2,bias=False)`。
- 神经元：固定alpha取0.25、0.5或0.75，`min_v_mem=-1`、减阈值复位、MultiSpike、PeriodicExponential；每个输出通道学习一个初值1的正阈值。
- 全层保留跨bin状态，不施加bias约束；默认Exodus。
- 在线增强与ANN完全相同，使用`all_physical_source_epoch_random`。

三档alpha均完成seed2021/2022/2023、200轮训练；配置和结果见[SNN固定alpha对照](../../../../docs/model/02_snn_fixed_alpha_comparison.md)。数据协议与ANN参考见[01文档](../../../../docs/model/01_all_physical_source_event_recombination_baseline.md)。`analyze_activity.py`从最佳checkpoint统计逐层脉冲、原始稀疏SynOps、逐时间步transition及单列的重复输入映射复用候选；映射估算不覆盖原始SynOps。

当前正式数据上的L1/L2跨bin状态实验使用：

- `select_l2_stateless_channels.py`：固定epoch100权重和L1全无状态，只用validation条件贪心选择指定数量的L2通道，默认4个，并核对缓存后缀与完整前向等价。
- `run_l1_l2_stateless_controls.py`：审计epoch100精确恢复，组织搜索/随机四通道续训，以及L1、L1＋随机L2四通道、L1＋L2全层三个从头200轮对照。
- `run_l1_l2_eight_channel_controls.py`：组织L1全无状态＋L2贪心8通道从epoch100续至200，以及L1全无状态＋L2随机8通道从头200轮的三seed对照。

当前实验以正式无尾窗缓存和L1/L2状态消融为准；代码中尚保留部分训练选项，不代表已授权扩展实验。
