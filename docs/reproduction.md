# 独立使用与结果复现

仓库目录下的命令只读本仓库模型、输入和结果，不依赖相邻软件项目。
所有新计算写入显式指定的runs/子目录，已确认的results/保留原始记录。

## 论文取数

Python标准库即可执行，完整检查不运行仿真：
```bash
python scripts/verify.py
python scripts/verify.py --full
python scripts/report.py --output runs/paper_metrics.csv
```

完整检查会解压读取SAIF信号活动和报告，并验证原始SHA-256。
已有CSV直接位于results/paper_metrics.csv。

## 数值参考与RTL回放

使用Python 3.11及requirements.txt依赖；仅回放和整数单元测试需要NumPy，
模型量化另需PyTorch和Sinabs。CPU模型加载使用本仓库hardware/quantization/model.py，
与软件侧模型定义保持相同数学实现。

```bash
python -m unittest discover -s tests -v
source /opt/amd/2025.2.1/Vivado/settings64.sh
python scripts/replay.py --variant l1_l2_stateless --samples 0 --output-dir runs/rtl_l1_l2
python scripts/quantize.py --variant l1_l2_stateless --output-dir runs/quant_l1_l2
python -m hardware.quantization.pack \
  --parameters models/l1_l2_stateless/parameters.npz --output-dir runs/weight_image
```

replay默认覆盖16输入；--samples可以指定已有验证索引0～15。
NPZ参考数组生成输入与逐层.mem文件时逐个检查原始校验值，不进行模型推理。
整理后的运行核验记录见[verification.md](verification.md)。

## 综合和布局布线

```bash
mkdir -p runs/synthesis/full_state
vivado -mode batch -nojournal -log runs/synthesis/full_state/vivado.log \
  -source hardware/vivado/synthesize.tcl -tclargs runs/synthesis/full_state 0
mkdir -p runs/implementation/full_state
vivado -mode batch -nojournal -log runs/implementation/full_state/vivado.log \
  -source hardware/vivado/implement.tcl \
  -tclargs runs/synthesis/full_state runs/implementation/full_state
vivado -mode batch -nojournal -source hardware/vivado/audit.tcl \
  -tclargs runs/implementation/full_state
```

深度参数0/1/2/3对应全状态、L1、L1～L2、L1～L3；每组使用自己的输出目录。
已有最终DCP和全部原报告在results/，写论文无需执行这些步骤。

## 功耗与活动

直接使用已保存的活动重算功耗：
```bash
python scripts/power.py --variant l1_l2_stateless --output-dir runs/power_l1_l2
```

只有需要重新记录波形时才执行：
```bash
python scripts/activity.py --variant l1_l2_stateless --samples 0 \
  --output-dir runs/activity_l1_l2
```

该命令使用已保存的布线后功能电路，不包含逐线SDF延时。结果统计条件为
160 MHz、25°C结温、典型工艺与标称电压，覆盖推理窗口。
