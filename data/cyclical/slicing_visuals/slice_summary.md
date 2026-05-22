# Cyclical Slicing Method Summary

## Dataset

- Rows analysed: `24001`
- Machine: `inkjet_02`
- Fixed-period window size: `1000` rows

## Method Comparison

| Method | Cycles | Normal | Anomalous | Mean rows | Std rows | Min-Max rows | Length CV | Normal corr | Normal NRMSE |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| peak | 9 | 7 | 2 | 2551.6 | 198.8 | 2036-2780 | 0.078 | 0.883 | 0.087 |
| zero-crossing | 12 | 9 | 3 | 1744.0 | 618.3 | 1195-2616 | 0.355 | 0.165 | 0.668 |
| fixed-period | 24 | 22 | 2 | 1000.0 | 0.0 | 1000-1000 | 0.000 | -0.022 | 4.009 |

## Recommendation

Based on the actual slices and the generated 3x2 plots, `peak` is the preferred slicing method for this eval file.

Peak detection is the strongest practical choice here because it slices on recurring pressure maxima, so the resulting windows align to the physical-looking cycle shape instead of arbitrary row offsets. In the visual comparison, the peak-detected normal cycles are the closest to near-identical cycle shapes, while fixed-period windows can cut through different phases of the same pressure pattern and zero-crossing can split or merge regions depending on the signal baseline.

## Metric Notes

- `Normal corr` is the mean pairwise correlation between resampled normal cycles. Higher means cycle shapes are more similar.
- `Normal NRMSE` is the average error from the normal-cycle template after resampling. Lower means cycles are more consistent.
- `Length CV` is the coefficient of variation of cycle lengths. Lower means cycle durations are more uniform.
- A sliced cycle is labelled anomalous if any row inside it has `cycle_label = anomalous`, so non-fixed slicing methods can report more anomalous cycles than the number of originally injected fixed windows.
