# 机器证据说明

这里保存导师或后续 Codex 可直接检查的机器可读结果。目录名已缩短，数值内容来自主
工程的冻结硬件产物。

| 目录 | 范围 |
|---|---|
| `formal_quantization/` | 48 个正式模型、54,105 个 test 输入的配置、汇总结果与独立核验 |
| `fixed_point_reference/` | Pursuit user15 三 seed、全 split 的浮点/定点对照与算术合同 |
| `golden_vectors_seed2022/` | seed2022 validation 样本 0 的参数、逐层 trace、MEM 和 NPZ |
| `hardware_image_stateless_seed2022/` | RTL 使用的 256-bit 权重流、逐通道常量和 fraction bits |
| `rtl_three_route/` | 三种整网组织的汇总、周期分布、顶层回放和 Yosys 统计 |

`rtl_three_route/comparison.json` 是硬件指标的首要机器入口。它同时记录三条路线的
scheduled lane、周期、逻辑访存、权重读取、状态访问、Yosys primitive mapping、
相对变化和证据边界。`analysis/three_route_pooled_validation_schema3.json` 保存三个 seed、
1,278 个 seed—样本的逐样本前端 FSM 周期及汇总。

三条路线目录只保留最终 `report.json`、primitive stat、仿真/lint 摘要和 400 事件输入。
生成的 `.vvp`、大体积 Yosys 文本日志及与 golden 重复的拼接 trace 没有再次纳入仓库；
它们可由 `code/hardware/` 和随仓库提供的 golden 重新生成。

为使仓库移动后仍能运行，`golden_vectors_seed2022/manifest.json` 与
`hardware_image_stateless_seed2022/layout.json` 中指向已复制文件的 `path` 字段改为
从仓库根目录出发的相对路径，并同步更新 layout 对 portable manifest 的 SHA-256。
参数、trace、权重镜像及其逐文件 SHA-256 均未改变。

最强的整网 RTL 复核命令见根目录 `README.md`。预期最终顶层结果为：

```text
checked values: 159408
mismatches:     0
range errors:   0
cycles:         67929
```

正式量化结果与 RTL 证据覆盖对象不同：前者覆盖 Pursuit/Saccade 各 24 个最终模型；
后者针对 Pursuit user15 eye0 的一套最终模型做完整 RTL 实现和逐拍回放。

