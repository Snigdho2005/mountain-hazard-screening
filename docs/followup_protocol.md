# Follow-up protocol (declared 2 November 2026, before any of these analyses was run)

The first case study (`chamoli_case_study.md`) left several items of the concept paper's validation plan
undone. This file fixes, in advance, how each follow-up is run and judged. Thresholds are in
`configs/chamoli.yaml` and `mhs/config.py`; they will not be changed after results are seen. Any later
change will be reported as a separate, post-hoc analysis.

These follow-ups are **not blind** in the way the first test was: the first test already showed where the
optical method failed. They are reported as follow-ups, with that caveat.

## A. Source detection, volume and energy from elevation change (§4.2, §8, §9)

- **Data:** Shean et al. (2021) post-event (10–11 Feb 2021) minus pre-event (Sept 2015) 2 m DEM difference,
  Zenodo 10.5281/zenodo.4558692 (CC BY-NC 4.0), resampled to the 10 m grid by block averaging.
- **Noise:** stable terrain = slope < 35°, NDSI < 0.4 on both image dates, > 500 m from the route traced from
  the documented source. The stable-terrain median is removed as vertical bias; NMAD is the 1-σ noise.
- **Lowered pixel:** dz ≤ −max(5 m, 3 × NMAD). Opening 1 px, 8-connectivity, zones ≥ 5,000 m².
- **Per zone:** loss volume with correlated error (σ_V = NMAD · A / √n_eff, correlation length 100 m);
  mass-weighted source elevation; D8 route and first valley-floor reference; drop; Monte Carlo mass and energy
  with **one density range for all zones, 900–2,700 kg m⁻³** (glacier ice to dense rock; material is not known
  from elevation change); exposure from the OpenStreetMap inventory.
- **Tests:**
  1. *Detection:* is there a lowered zone within 500 m / 1 km / 2 km of the documented source?
  2. *Ranking:* best rank of such a zone among all lowered zones under the full ranking (state, exposure,
     energy, ...), against the random expectation and two baselines (largest area; largest volume).
  3. *Removal tests:* the same with energy, exposure, or route (with corridor exposure) removed. Persistence is
     not available for DEM zones, so removing instability cannot change the order; this is reported, not hidden.
  4. *Volume:* the largest-volume lowered zone within 1 km of the source is "the source zone". Its loss volume
     and 95 % interval are compared with the published 26.9 × 10⁶ m³ (95 % CI 26.5–27.3 × 10⁶).
     **Caveat:** the published volume was derived from the same DEMs, so agreement checks the method, not the
     data. Sept 2015 → Feb 2021 also includes five years of glacier change, pre-failure movement and snow.
  5. *Energy:* reported as a Monte Carlo interval; compared with the concept paper's sensitivity range
     (3.6–7.7 × 10¹⁴ J), noting the different density range and drop definition.
  6. *Pathway:* route from the source zone through the four documented checkpoints.
- **Success is not defined as "the source ranks first".** The result is reported whatever it is.

## B. Ranking and removal tests

Covered by A (tests 2–3), now with measured energy instead of an area × drop index.

## C. Change-detection accuracy on the test strip (§8)

- **Design:** stratified random sample (Olofsson et al. 2014). Strata in the eroded test strip, usable on both
  dates, from one U-Net run and the rule baseline: both changed / U-Net only / baseline only / neither.
  15, 15, 15 and 25 points, ≥ 100 m apart, seed 29.
- **Blind labelling:** points renamed C1…C70 in random order; the stratum key is kept in a separate file.
  Label = `change` if the surface type differs between t1 and t2, `no_change` if it does not, `skip` if unsure
  or mixed. Shadow or brightness differences alone are not change.
- **Stopping rule** (added 2 Nov 2026, after the points were generated and before any was labelled): points are
  labelled in ID order from C1; labelling may stop after C40, C50, C60 or C70. Because IDs are in random order,
  the labelled prefix is a random subsample; every labelled point is used and per-stratum weights use the
  number actually labelled in each stratum.
- **Estimates:** area-weighted precision, recall and F1 of "change", overall accuracy and estimated change area,
  with 95 % bootstrap intervals, for each U-Net seed's change map and for the rule baseline.

## D. Rainfall and temperature indicators (§7)

- **Data:** ERA5 daily reanalysis at the documented source via the Open-Meteo archive API.
- **Indicators:** precipitation total and mean temperature over 1, 3, 7 and 30 days ending 6 Feb 2021, each
  compared with the same calendar window in 1991–2020 (30 values → percentile score per paper §7); Dec–Feb mean
  temperature trend.
- **Use:** event-level context for the instability record only. One ~31 km cell cannot rank zones within the
  scene, and an anomaly is not evidence of cause.

## Not covered here

- **Second event and non-event catchments (§12):** a separate phase with its own protocol.
- **Second labeller and specialist review (§12):** require other people.
