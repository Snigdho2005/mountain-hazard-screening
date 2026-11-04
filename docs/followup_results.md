# Follow-up results (November 2026)

Run under the protocol fixed in advance in [`followup_protocol.md`](followup_protocol.md). These analyses
were done after the first test had shown where the optical method failed, so they are **follow-ups, not
blind tests**. Outputs: [`results/chamoli_2021/followup/`](../results/chamoli_2021/followup/); notebook:
`notebooks/kaggle_03_followup.ipynb`.

## A. Source detection, volume and energy from elevation change

Data: Shean et al. (2021) 2 m DEM difference (10–11 Feb 2021 minus Sept 2015), resampled to 10 m.

| item | value |
|---|---|
| Coverage of the scene | 67 % |
| Stable-terrain noise | bias −0.31 m (removed), NMAD 3.24 m (576,851 px) |
| Lowering threshold | 9.73 m (3 × NMAD) |
| Lowered zones | 552 |

**Detection.** Three lowered zones lie within 500 m of the documented source, with up to 183 m of lowering.
Elevation change detects the source that winter optical imagery missed.

**Volume.** The declared rule (largest-volume zone within 1 km of the documented point) selected a zone of
1.40 × 10⁷ m³ (95 % CI 1.33–1.46 × 10⁷), **0.52 × the published 2.69 × 10⁷ m³**; the published value lies
outside its interval. *Post-hoc observation, not a result:* a zone 1.5 km east of the documented point on
the same face (5,414 m, mean lowering 85 m) has 2.61 × 10⁷ m³ (97 % of the published volume) and an energy of
7.5 × 10¹⁴ J (90 %: 4.1–10.9 × 10¹⁴), consistent with the concept paper's 3.6–7.7 × 10¹⁴ J. It is likely the
main scar, with the reference coordinate at its western edge; this should be checked against the published
source polygon. The published volume was derived from the same DEMs, so agreement would confirm the method,
not the data.

**Pathway** from the declared source zone: all four checkpoints reached in order (Rishiganga HEP 15.1 km vs
~15 km published; Tapovan HEP 23.1 km vs 26 km).

**Ranking (552 zones; random expectation for the best rank: 138 within 500 m, 50 within 1 km).**

| ranking | best rank, ≤500 m | ≤1 km | ≤2 km |
|---|---|---|---|
| full framework | 14 | 14 | 13 |
| without energy | 14 | 14 | 13 |
| without instability | 14 | 14 | 13 |
| without exposure | **2** | **2** | **1** |
| without route (and corridor exposure) | 2 | 2 | 1 |
| largest area only | 2 | 2 | 1 |
| largest volume only | 2 | 2 | 1 |

The framework places the source in the top 2.5 %, far better than chance, but **a simple volume or area
ranking does better**. The twelve zones ranked above it are small lowered patches in the Rishiganga gorge
whose corridors pass seven assets rather than six. Because the ranking orders by exposure count before
energy, a one-asset difference outweighs a ~30-fold difference in energy. This is a design weakness of the
ranking rule. It is **not** changed here (that would be tuning on the answer); a revised rule must be
declared before a test on another event. Removing instability changes nothing because persistence is not
available for DEM zones.

## C. Change-detection accuracy on the test strip

Design: stratified random sample (strata from the seed-0 U-Net change map × the rule-baseline change map),
labelled blind by one analyst on 10 m SWIR composites; 70 points labelled, **43 skipped (61 %)** as dark,
masked or mixed.

| stratum (pixels) | points | labelled | change / no change | 95 % exact interval for share of change |
|---|---|---|---|---|
| both maps changed (3,487) | 15 | 10 | 10 / 0 | 0.69–1.00 |
| U-Net only (3,692) | 15 | 4 | 0 / 4 | 0.00–0.60 |
| rule baseline only (70,276) | 15 | 2 | 2 / 0 | 0.16–1.00 |
| neither (601,336) | 25 | 11 | 0 / 11 | 0.00–0.29 |

Area-weighted point estimates (the bootstrap intervals in `change_accuracy.md` are degenerate because every
stratum's labels are unanimous; the exact per-stratum intervals above show the real uncertainty):

| map | precision | recall | F1 |
|---|---|---|---|
| U-Net seed 0 | 0.49 | 0.05 | 0.09 |
| U-Net seed 1 | 1.00 | 0.51 | 0.68 |
| U-Net seed 2 | 1.00 | 0.04 | 0.07 |
| rule baseline | 1.00 | 1.00 | 1.00 |

**Reading.** On clearly visible ground the labelled changes are snow → bare ground; the rule baseline flagged
every one, while two of three U-Net runs missed most of them and all four U-Net-only points were false alarms.
The point estimates must not be read as precise: the rule-baseline-only stratum, which drives recall, rests
on two labels; 0 of 11 in the unchanged stratum still allows up to 29 % change there, which would cut every
recall estimate sharply; and 61 % of points could not be judged, mostly in shadow, so all figures apply only
to sunlit, unmixed surfaces. The main finding is therefore qualitative: **in winter imagery most of the
change area cannot be verified at 10 m, and on the part that can, the U-Net's confidence and illumination
gates suppress real snow → ground change that the simpler rules keep.** This contrasts with the per-class
segmentation result, where the U-Net was better than the rules.

## D. Rainfall and temperature before the event (ERA5 at the source)

| window ending 6 Feb 2021 | precipitation (mm) | 1991–2020 median | percentile score | temperature anomaly (°C) | percentile score |
|---|---|---|---|---|---|
| 1 day | 0.0 | 0.7 | 0.00 | −0.85 | 0.42 |
| 3 days | 10.1 | 8.6 | 0.32 | −1.29 | 0.26 |
| 7 days | 11.7 | 18.2 | 0.22 | +0.75 | 0.65 |
| 30 days | 19.3 | 115.7 | 0.00 | +2.16 | 1.00 |

The 30 days before the event were the driest and warmest of the 1991–2020 record for that window at this ERA5
cell; the final week was near normal. Winter mean-temperature trend: +0.05 °C per decade (29 winters).
This is event-level context for the instability record. One ~31 km cell cannot rank zones, and an anomaly is
not evidence of cause.

## Repeat benchmark (third set of three seeds)

Macro F1 0.705 ± 0.091 (0.607–0.786) vs 0.519 for the rule baseline. Across all ten U-Net runs to date, every
run exceeds the baseline; single-run spread stays large.

## What remains open

- A second event and non-event catchments, with a revised ranking rule declared in advance (§12).
- A second, independent labeller for the change points, and specialist review of the source zones.
- Shadow-insensitive change detection (Sentinel-1 SAR) to make the 61 % of unverifiable points usable.
