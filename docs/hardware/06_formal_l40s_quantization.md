# L40S正式模型：双任务24用户16位量化

2026-10-12完成48个正式checkpoint的完整test定点评估。论文与导师汇报采用
连续回归误差：Pursuit浮点/定点为0.823945°/0.825481°，Saccade浮点/定点为
1.325509°/1.326037°。不复用旧RTX3090 user15量化指标。既有Saccade网格
取整结果只保留在机器归档中作内部诊断，不作为汇报指标。

## 对象与算术

- 两任务各24用户（user4～27、eye0），seed2021、T=2重复完整400事件表示、
  Conv1/Conv2无状态；读取正式100→200轮续训分支的最佳checkpoint。
- 与浮点评估使用相同完整test：Pursuit 25,716、Saccade 28,389，共54,105输入。
  先逐用户平均，再将24用户均值等权平均；best为最低用户均值。
- 量化前折叠BN及可学习阈值；逐输出通道2的幂scale，最多15个小数位。
  卷积、两个脉冲FC及末读出权重S16，常量/完整点积后电流S24，膜S16、脉冲U8；
  静态参数最近舍入（半距远离零），膜衰减floor(3v/4)，减阈值复位。
- 本次是训练后量化前向，不训练、不用test标签调scale。模型来自L40S正式训练；
  定点参考在carbonite CPU执行，PyTorch 2.13.0+cu130、4线程、batch16。
- 点积使用float64承载精确整数，逐层验证保守和界低于2^53；比旧float32参考
  扩大了精确整数范围，未改变量化算术。已保存逐用户量化参数与完整预测。

## 性能

本次仅评估test，表内train/validation用“—”表示未重新测量。两任务均报告
连续坐标输出的像素与角误差；角度使用40英寸1920×1080、400mm几何。

| 任务及输出 | 浮点像素误差（px） | 定点像素误差（px） | 浮点角误差（°） | 定点角误差（°） |
|---|---:|---:|---:|---:|
| Pursuit连续 | train —<br>validation —<br>test 16.119761 | train —<br>validation —<br>test 16.126505 | train —<br>validation —<br>test 0.823945 | train —<br>validation —<br>test 0.825481 |
| Saccade连续 | train —<br>validation —<br>test 26.047575 | train —<br>validation —<br>test 26.056241 | train —<br>validation —<br>test 1.325509 | train —<br>validation —<br>test 1.326037 |

Pursuit最好用户均为user10，浮点/定点0.490071/0.506021°；Saccade连续最好
用户均为user17，0.720360/0.725520°。用户等权连续误差变化分别+0.001536°与
+0.000528°。保留8,136个S16膜值，容量15.890625KiB。

## 验证

- 48模型、54,105输入完整覆盖；264份输入文件及4份来源代码SHA再次核对一致。
- 每个模型首个真实输入的新旧参考逐层电流/脉冲/膜与读出数组全部相同，共3,024数组。
- 独立int64卷积与FC、负膜衰减/复位等4项数值测试全部通过。
- 保存预测用独立归一化视线弦长公式复算，角误差最大差1.43e-14°；每人均值、
  24人等权汇总逐项一致。
- 全部膜值位于S16范围，脉冲最大174且U8饱和0。
  S24电流共有42次饱和（含重复时间步、Pool前位置）：Pursuit user9/user22为4/24，
  Saccade user6/10/11/27为4/2/4/4，均在L2；最大完整电流绝对值14,482,375。
  汇报误差已包含这些饱和，不沿用旧user15“全split无饱和”的结论。

这是正式多用户数值量化结果；原RTL验证记录保持原模型范围，本次未重新运行RTL。

## 复现与产物

入口为`code/hardware/evaluate_formal_quantization.py`，检查入口为
`code/hardware/tests_formal_quantization.py`。默认路径由`code.paths`取得。

```bash
/data/xuzx/env/ebv-eye/bin/python -m code.hardware.evaluate_formal_quantization \
  --threads 4 --output <new-result-directory>
/data/xuzx/env/ebv-eye/bin/python -m code.hardware.tests_formal_quantization
```

正式归档：`/data/xuzx/result/ebv-eye/analysis/formal_fixed_point_seed2021_20261012/`。
包含`configuration.json`、`results.json`、`verification.json`、源码快照，以及
每任务/用户的`summary.json`、`parameters.npz`、`predictions_test.npz`。
原checkpoint、summary、缓存与已存浮点预测不变。
