# Codex 工作说明

开始研究前按顺序完整阅读：

1. `README.md`；
2. `docs/hardware/07_advisor_hardware_progress.md`；
3. `docs/hardware/figures/ebv_eye_final_pe_bram_dataflow.svg`；
4. `evidence/README.md`；
5. 需要深入某项实现时，再读 `docs/hardware/01...06` 和对应源码。

对外描述三种整网组织时使用以下中文名称：

- 稠密扫描基线；
- 仅浅层事件驱动；
- 浅层事件驱动＋深层逐层相邻执行双时间步。

源码和 JSON 中的 `Dense-TM`、`Sparse-TM`、`Sparse-LM-A` 只作为内部标识保留。
最终顶层为 `code/hardware/rtl/pursuit_sparse_layer_major_accelerator_top.sv`。

所有定量硬件结论先与 `evidence/rtl_three_route/comparison.json` 核对。报告中的
200 MHz 延迟与服务率是冻结设计点的周期换算；不得写成 Vivado 已收敛频率。
scheduled lane slot 不等于翻转活动或能量，Yosys primitive mapping 不等于布局布线
后的资源、Fmax 或功率。当前没有 Vivado 实现或 FPGA 板上结果。

Saccade 精度使用网络连续坐标输出：浮点 1.325509°、16 位定点 1.326037°。
不要改写为最近刺激点取整误差或命中率。完整 RTL 对象是 Pursuit user15 eye0；
48 模型结果是软件位精确定点评估，不要扩写成 48 套 RTL 回放。

`evidence/golden_vectors_seed2022/manifest.json` 与
`evidence/hardware_image_stateless_seed2022/layout.json` 的路径字段已改为仓库相对路径，
以便从仓库根目录执行回放；各 `.mem`、`.npz` 的内容及 SHA-256 未改变。不要手工修改
golden 或机器结果。新增或修改代码后运行与改动对应的测试，并在结论中区分新结果与
既有归档结果。

这个仓库不含原始数据、正式缓存或训练 checkpoint。缺少这些对象时仍可阅读全部实现、
核验归档 JSON，并使用随仓库提供的 golden 与硬件镜像重跑 RTL；不要为缺失数据创建
伪造回退输入。

