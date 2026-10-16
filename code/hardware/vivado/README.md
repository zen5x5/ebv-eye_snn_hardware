# Vivado 核心实现

工具：Vivado 2025.2.1。器件：xczu3eg-sfvc784-2-e（AUP-ZU3）。

在仓库根目录运行：

```bash
source /opt/amd/2025.2.1/Vivado/settings64.sh
bash code/hardware/vivado/run_sparse_ooc.sh build/route160 sparse_lm_a
```

完整参数：

```text
run_sparse_ooc.sh OUTPUT_DIR VARIANT [PART] [JOBS] [PERIOD_NS]
```

默认 PART 为 xczu3eg-sfvc784-2-e、JOBS 为 8、PERIOD_NS 为 6.250。VARIANT 使用 sparse_lm_a 运行当前顶层。运行 200 MHz 对照时显式传入 5.000 ns：

```bash
bash code/hardware/vivado/run_sparse_ooc.sh build/route200 sparse_lm_a xczu3eg-sfvc784-2-e 8 5.000
```

输出包含综合、布局布线 checkpoint、资源、时序、功率、DRC 和源码校验值。时序检查不通过时，脚本保留报告并返回失败状态。输出目录使用新的空路径。

## 已有证据

evidence/current/vivado/ 保存已布线核心的报告。原实现按 5.000 ns 布线，setup 余量为 -1.109 ns；相同布线覆盖为 6.250 ns 后，setup/hold 余量为 +0.141/+0.040 ns。

本次仓库更新把稀疏实现入口的默认目标设置为 160 MHz；新的完整实现会重新执行布局和布线。归档报告仍保留原始 200 MHz 运行说明和单独的 160 MHz 复核文件。

OOC 表示单独实现加速器核心。PS、AXI、板级引脚和时钟连接需要在后续板级封装中接入。用于开源历史基线的 run_ooc.sh 保留原 200 MHz 约束。
