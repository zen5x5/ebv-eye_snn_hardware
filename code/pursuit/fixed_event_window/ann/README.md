# Pursuit 当前ANN基线

输入为`2×96×128`双极性400事件计数图；八层卷积为`2→8→16→24→32→40→40→40→40`，前五层在`Conv→BN→ReLU`后做`2×2` MaxPool，回归头为`480→60→12→2`，共96,480参数。

正式训练启用`all_physical_source_epoch_random`：2961个物理train图保留，2948个合格source每轮各生成一张200＋200事件合成图。seed2021/2022/2023均训练200轮。完整规则和结果见[正式基线报告](../../../../docs/model/01_all_physical_source_event_recombination_baseline.md)。

```bash
python -m code.pursuit.fixed_event_window.ann.tests
python -m code.pursuit.fixed_event_window.ann.training train \
  --cache-dir <当前无尾窗缓存/user15/eye0> \
  --output-dir <新结果目录> --seed 2021 --epochs 200 --device cuda:0 \
  --local-recombination \
  --local-recombination-mode all_physical_source_epoch_random
```

训练入口拒绝覆盖已有结果目录；配置、checkpoint、history和summary只写入对应结果目录。

## Compact 90k对照

`StaticPursuitANNCompact90K`保留八层、五次MaxPool和两隐藏层回归头，把后四层通道从40降为36，并把回归头改为`432→48→12→2`。在当前`2×96×128`输入下为78,868参数，即正式ANN的81.75%。它已经接入`--architecture eight_layer_8_16_24_32_36_36_36_36_hidden48_12_maxpool_96x128`。

同一结构改用`2×120×220`输入时，五次池化后为`36×3×6=648`，回归头相应为`648→48→12→2`，总参数89,236；该尺寸只完成参数和前向核验，没有训练。当前96×128版本已按正式数据与增强完成三seed200轮，validation/test为`0.589±0.019/0.599±0.030°`；它不替换96,480参数正式ANN。
