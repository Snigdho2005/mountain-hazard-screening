# Segmentation benchmark: U-Net vs rule baseline, held-out east test strip

Labels: 63 hand-digitised polygons (`chamoli-labels`), rasterised to t1/t2 label layers.
Test-strip support (t1 + t2 pixels): background 1,225; snow_ice 11,322; rock 1,535; soil 5,285; water 0; debris 0.
Water and debris have no test-strip labels and are **not evaluable** on this split.

## Main result (frozen run, `notebooks/kaggle_01_frozen_benchmark.ipynb`)

DEM: Copernicus GLO-30 via Microsoft Planetary Computer. Seeds 0, 1, 2.

| seed | best epoch | test macro F1 |
|---|---|---|
| 0 | 16 | 0.621 |
| 1 | 21 | 0.671 |
| 2 | 1 | 0.651 |

| class | support | U-Net F1 mean ± sd | min–max | baseline F1 |
|---|---|---|---|---|
| background | 1225 | 0.982 ± 0.018 | 0.961–0.994 | 0.713 |
| snow_ice | 11322 | 0.997 ± 0.005 | 0.991–1.000 | 1.000 |
| rock | 1535 | 0.301 ± 0.032 | 0.275–0.337 | 0.354 |
| soil | 5285 | 0.311 ± 0.097 | 0.199–0.377 | 0.009 |
| **macro F1** | | **0.648 ± 0.025** | 0.621–0.671 | **0.519** |
| overall accuracy | | 0.740 ± 0.020 | 0.726–0.763 | 0.695 |

## Repeat run (`notebooks/kaggle_02_event_evaluation.ipynb`)

Planetary Computer's DEM endpoint returned 404 on 1 Nov 2026, so the same Copernicus GLO-30 tile was read
from the AWS open-data bucket. Same labels, settings and seeds. GPU: Tesla T4.

A single seed-0 run first gave macro F1 **0.747** (rock 0.432, soil 0.555). Re-running the same seed on the
same machine gave **0.617**: training is not bit-reproducible on this GPU. The 3-seed repeat:

| seed | best epoch | test macro F1 |
|---|---|---|
| 0 | 25 | 0.617 |
| 1 | 6 | 0.730 |
| 2 | 8 | 0.581 |

| class | support | U-Net F1 mean ± sd | min–max | baseline F1 |
|---|---|---|---|---|
| background | 1225 | 0.984 ± 0.016 | 0.967–0.996 | 0.713 |
| snow_ice | 11322 | 1.000 ± 0.000 | 1.000–1.000 | 1.000 |
| rock | 1535 | 0.278 ± 0.094 | 0.170–0.343 | 0.354 |
| soil | 5285 | 0.309 ± 0.393 | 0.015–0.755 | 0.009 |
| **macro F1** | | **0.643 ± 0.078** | 0.581–0.730 | **0.519** |
| overall accuracy | | 0.766 ± 0.087 | 0.704–0.866 | 0.695 |

## Reading

- The 3-seed means agree (0.648 vs 0.643): the DEM source did not change the average.
- All 7 U-Net runs (0.581–0.747) exceed the rule baseline (0.519).
- Single runs are unstable, above all for soil (F1 0.015–0.755) and rock (0.170–0.432). Report means over seeds.
- Rock and soil are not reliably separated at 10 m in winter imagery; the baseline is no worse on rock.
- Model selection uses validation cross-entropy, but the validation strip has no rock or debris labels,
  so selection cannot favour those classes (seed 2 of the frozen run stopped at epoch 1).
