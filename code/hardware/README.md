# 硬件代码

当前顶层为 rtl/pursuit_sparse_layer_major_accelerator_top.sv，参数和标准数值使用 evidence/current/ 中的 user17/seed2022 资产。

## 常用入口

从仓库根运行：

```bash
source /opt/amd/2025.2.1/Vivado/settings64.sh
python -B scripts/replay_current.py --output-dir build/replay
bash code/hardware/vivado/run_sparse_ooc.sh build/route160 sparse_lm_a
```

[设计与资源](../../docs/hardware/current_design.md)、[性能比较](../../docs/hardware/comparison.md)、[机器证据](../../evidence/README.md)。

## 实现

- rtl/sparse_conv3x3_pool_lif_core.sv：L1/L2 三行累加、池化和脉冲计算。
- rtl/pursuit_sparse_l1_event_layer.sv：单位事件入口、24 路加法和非零结果发送。
- rtl/pursuit_sparse_l2_packet_layer.sv：非零计数包入口及完整 L2 输出存储。
- rtl/paired_conv3x3_layer_engine.sv：L3～L8 共享计算，层内相邻完成两个时间步。
- rtl/paired_linear_lif_layer_engine.sv：H1/H2 共用的 16 路整数计算。
- rtl/readout_engine.sv：两维坐标整数读出。
- rtl/pursuit_sparse_layer_major_accelerator_top.sv：层级控制、权重存储和整网连接。

S16 权重、S40 部分和、S24 电流、S16 膜电位、U8 脉冲为当前算术约定。完整过程见当前设计报告。

## 其他工具

定点参考、数值范围统计、标准向量生成及权重打包脚本保留。运行这些 Python 模块需要原项目的模型环境。当前 XSim 回放仅依赖标准库和 Vivado。

run_sparse_layer_major_golden_check.py 保留 Icarus/Verilator/Yosys 入口，固定检查当前 user17 样本周期。稠密、时间优先等脚本用于历史研究，历史周期和映射数据在 evidence/rtl_three_route/。
