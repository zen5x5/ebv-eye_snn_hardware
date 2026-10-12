# Hardware analysis utilities

这里存放跨任务可复用、且不属于 RTL 本体的硬件分析工具：

- `profile_snn_numeric_ranges.py`：记录定点实现需要覆盖的输入、归一化电流、
  膜电位、MultiSpike 计数及折叠参数范围；
- `profile_snn_channel_statistics.py`：记录逐通道发放、负载不均衡、稀疏
  SynOps贡献和折叠权重范围；
- `explore_pursuit_dataflow.py`：枚举 `P_OC/P_IC/P_K` 和乘法通道数，给出
  100 MHz下的理想调度周期、片上容量和每拍操作数带宽；
- `compare_synapse_arithmetic.py`：在真实整数脉冲上比较逐脉冲加权累加、
  非零计数乘权重和二进制移位加法三种突触算术；
- `evaluate_pursuit_fixed_point.py`：执行BN/阈值折叠、逐输出通道2的幂scale
  和位显式SNN推理，并对照同一CPU Sinabs浮点参考；
- `evaluate_formal_quantization.py`：量化L40S训练的双任务24用户seed2021正式
  checkpoint，沿用冻结16位合同并评估完整test；保存逐用户预测、参数和来源SHA；
- `tests_formal_quantization.py`：用独立int64卷积/FC验证整数累加，并检查
  负膜衰减、减阈值复位与saccade网格后处理；
- `generate_pursuit_golden_vectors.py`：导出量化参数、逐层`t0/t1`trace、NPZ、
  `.mem`和完整SHA-256 manifest；可分别生成L1/L2无状态和配对全状态模型；
- `pack_pursuit_hardware_image.py`：把卷积、H1/H2、readout和逐通道参数打包成
  RTL同步RAM直接读取的256-bit镜像，并反向重建全部张量自检；
- `summarize_pursuit_fixed_point.py`：校验三seed全split报告使用同一算术合同，
  汇总均值、样本标准差、饱和计数和输入文件哈希；
- `estimate_pursuit_rtl_schedule.py`：对144/288-DSP候选计入权重装载、padding
  扫描、由完整L1校准的14拍填充/排空与L3电流回放，给出实现前周期和存储
  估计；
- `run_rtl_checks.py`：统一运行Icarus、Verilator和Yosys，并生成可追溯JSON；
- `run_l1_golden_check.py`：校验正式golden文件哈希，运行完整L1逐位回放、严格
  lint和UltraScale+ primitive mapping；
- `run_conv_engine_golden_check.py`：同一个可编程卷积核心回放L1～L8两个bin；
- `run_full_accelerator_golden_check.py`：整合stateless/full golden与硬件镜像，
  回放SL-TM、SL-HYB和FS-TM，运行lean/full lint并汇总Yosys映射；
- `evaluate_pursuit_sparse_hybrid.py`：在三seed完整validation上逐值比较稀疏
  L1/L2、Pool后常量和L3电流复用，并统计Dense-TM、Sparse-TM与
  Sparse-LM-A的算术、逻辑访存和RTL校准周期分布；
- `run_sparse_time_major_golden_check.py`：回放“L1/L2稀疏＋原时间优先后端”
  完整顶层，检查逐层trace、周期、lint和Yosys primitive mapping；
- `run_paired_conv_golden_check.py`、`run_paired_linear_golden_check.py`：分别
  验证L3～L8相邻`t0/t1`卷积引擎和H1/H2成对linear引擎；
- `run_sparse_layer_major_golden_check.py`：回放“L1/L2稀疏＋方案A层内相邻
  两步”完整顶层，检查逐层trace、周期、lint和Yosys primitive mapping；
- `draw_advisor_hardware_architecture.py`：生成“浅层事件驱动＋深层逐层相邻
  执行双时间步”最终硬件架构图的SVG、PDF和高清PNG，用于导师汇报与报告内嵌；
- `rtl/`：计数乘法、逐单位加法、单bit移位项、逐级流水核心、位精确LIF、
  稀疏L1/L2三行九bank核心、规则与成对卷积/H1/H2/readout、激活/状态/
  电流缓存，以及稠密、Sparse-TM和Sparse-LM-A三套完整整网控制器；
- `vivado/`：AUP-ZU3 `xczu3eg-sfvc784-2-e`的200 MHz OOC综合、布局布线、
  时序、利用率、DRC、methodology与初步功率报告脚本；
- `environment.yml`：独立开源RTL环境的最小依赖。

这些工具不会训练或修改 checkpoint；CPU浮点复核不替换原L40S浮点指标，
正式多用户定点评估则单列为量化结果。
正式机器可读输出应写入 `EBV_EYE_RESULT_ROOT` 下的硬件分析目录；报告正文只
保存结论和复现入口。

双任务正式量化入口（输出目录必须不存在）：

```bash
/data/xuzx/env/ebv-eye/bin/python -m code.hardware.evaluate_formal_quantization \
  --threads 4 --output <result-root>/analysis/<new-quantization-directory>
/data/xuzx/env/ebv-eye/bin/python -m code.hardware.tests_formal_quantization
```

默认读取`code.paths`指定的正式缓存与结果根；完整48模型结果位于
`/data/xuzx/result/ebv-eye/analysis/formal_fixed_point_seed2021_20261012/`。
报告见`report/hardware/06_formal_l40s_quantization.md`。

示例：

```bash
python -m code.hardware.profile_snn_numeric_ranges \
  --checkpoint <checkpoint.pt> \
  --configuration <configuration.json> \
  --cache-dir <formal-cache-directory> \
  --split validation \
  --batch-size 16 \
  --output <result-root>/analysis/hardware/<profile>.json
```

脚本有三项语义自检：

- 逐步重建 `u=0.75*v_prev+I`、MultiSpike 和减阈值复位；
- 重建的最终膜必须与模型内部状态一致；
- 对全部无状态层，重复输入的两个时间步必须产生完全相同的脉冲。

输出文件带分析脚本、checkpoint、configuration 和缓存文件的 SHA-256，并拒绝
覆盖既有文件，便于后续定点参考和 RTL golden vector 追溯。train split 指正式
缓存中的物理来源窗；脚本不会复现训练时逐 epoch 随机生成的 200+200 重组样本。

## 环境

- 浮点模型与数值剖析继续使用 `/datanaco/xuzx/env/ebv-eye`；
- RTL仿真与开源综合使用 `/data/xuzx/env/ebv-eye-hardware`；
- Vivado在有安装的机器上完成AUP-ZU3器件相关综合、实现和时序。

当前硬件环境已验证 Icarus Verilog 13.0、Verilator 5.052、Yosys 0.69 和
cocotb 2.1。重建命令为：

```bash
micromamba create -y -p /data/xuzx/env/ebv-eye-hardware \
  -f code/hardware/environment.yml
```

当前交互机运行分析自检时需使用该机的环境动态库：

```bash
LD_LIBRARY_PATH=/datanaco/xuzx/env/ebv-eye/lib \
MPLCONFIGDIR=/tmp/mpl-ebv-eye-hardware \
/datanaco/xuzx/env/ebv-eye/bin/python -m code.hardware.tests
```

完整L1正式回放使用独立RTL环境：

```bash
/data/xuzx/env/ebv-eye-hardware/bin/python \
  -m code.hardware.run_l1_golden_check \
  --golden-dir <golden-vector-directory> \
  --output-dir <new-result-directory>
```

整网正式回放需要stateless/full两套golden和镜像。已经验收的机器报告为：

```text
/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/
  rtl_full_accelerator_pre_vivado_20261006/report.json
```

它对三条dataflow共检查699,408个逐层/最终值；SL-TM、SL-HYB、FS-TM周期
分别为119,791、83,286、119,791。该顶层按完整raster调度，零脉冲不跳过，
现作为正确性与状态/时间复用基线。

两套稀疏整网回放使用相同的stateless golden与硬件镜像：

```bash
/data/xuzx/env/ebv-eye-hardware/bin/python \
  -m code.hardware.run_sparse_time_major_golden_check \
  --golden-dir <stateless-golden-directory> \
  --image-dir <stateless-hardware-image-directory> \
  --output-dir <new-sparse-tm-result-directory>

/data/xuzx/env/ebv-eye-hardware/bin/python \
  -m code.hardware.run_sparse_layer_major_golden_check \
  --golden-dir <stateless-golden-directory> \
  --image-dir <stateless-hardware-image-directory> \
  --output-dir <new-sparse-lm-a-result-directory>
```

seed2022 validation样本0的Sparse-TM/Sparse-LM-A分别逐值检查159,408个值，
零失配，周期72,431/67,929。三seed完整validation的软件位精确与周期分布入口
为`evaluate_pursuit_sparse_hybrid.py`；其余样本周期是逐样本前端FSM精确回放
加完整RTL校准固定后端，不冒充全部样本均做过整网RTL仿真。该分析还以同一
固定宽度RTL口径报告Dense-TM的14.592408M和两条稀疏路线平均7.911147M个
scheduled lane slot；它不是切换活动或能量。Vivado入口和证据边界见
`vivado/README.md`。正式validation、整网回放、映射与manifest归档于：

```text
/data/xuzx/result/ebv-eye/pursuit/fixed_event_window/hardware/
  rtl_sparse_three_schedule_pre_vivado_20261008/
```

Yosys整顶层映射中，两条稀疏路线均为354个DSP48E2；Sparse-TM为117块
BRAM36等效、36,941 FF，Sparse-LM-A为80.5块BRAM36等效、36,124 FF。
计入LUTRAM后的物理LUT理论界分别为52,432～96,408和56,211～95,133，均
跨过AUP-ZU3的70,560容量线，不能用开源映射替代Vivado的packing与布局结论。
