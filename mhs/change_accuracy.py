"""Follow-up experiment C: change-detection accuracy on the held-out test strip (paper §8).

Design-based accuracy assessment (stratified random sampling with area weights, as recommended for
land-change maps by Olofsson et al. 2014, Remote Sensing of Environment 148, 42-57):

  strata   pixels of the test strip usable on both dates, split by two maps:
           U-Net class change (transition.tif != 0) x rule-baseline change (baseline_change.tif)
           -> both / unet_only / rules_only / neither
  sample   n_h random pixels per stratum, >= min_sep_m apart, shuffled and renamed C1..Cn so the
           labeller cannot see the stratum (the key is written to a separate file)
  label    by eye on the t1 and t2 previews: change / no_change / skip
           change = the surface type differs between the dates (e.g. snow -> rock, soil -> debris);
           a shadow or brightness difference alone is not change
  estimate weighted confusion matrix (weight N_h / n_h), precision and recall of "change", F1 and
           overall accuracy, with 95% bootstrap intervals (resampling within strata)

Any map can be scored with the same sample, because inclusion probabilities are known.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rasterio.warp import transform_geom
from scipy import ndimage

from .io import read_raster

STRATA = ("both", "unet_only", "rules_only", "neither")
YES = {"change", "changed", "c", "yes", "y", "1", "true"}
NO = {"no_change", "nochange", "no change", "unchanged", "n", "no", "0", "false"}


def _strata(cfg, run_dir):
    from .model.train import SPLIT_TEST, geographic_split

    run = Path(run_dir)
    c1 = read_raster(run / "maps" / "cls1.tif").data[0]
    c2 = read_raster(run / "maps" / "cls2.tif").data[0]
    unet = read_raster(run / "maps" / "transition.tif").data[0] != 0
    rules = read_raster(run / "maps" / "baseline_change.tif").data[0].astype(bool)
    test = geographic_split(c1.shape, **cfg["model"]["split"]) == 2  # SPLIT_TEST
    test = ndimage.binary_erosion(test, iterations=3)
    usable = test & (c1 != 255) & (c2 != 255)
    s = np.full(c1.shape, -1, np.int8)
    s[usable & unet & rules] = 0
    s[usable & unet & ~rules] = 1
    s[usable & ~unet & rules] = 2
    s[usable & ~unet & ~rules] = 3
    return s


def change_points(cfg: dict, run_dir: str | Path, out_dir: str | Path, n_per_stratum=(15, 15, 15, 25),
                  seed: int = 29, min_sep_m: float = 100.0) -> dict:
    grid = read_raster(cfg["inputs"]["t1"])
    s = _strata(cfg, run_dir)
    rng = np.random.default_rng(seed)
    sep2 = (min_sep_m / grid.res) ** 2
    pts, counts = [], {}
    for h, name in enumerate(STRATA):
        rr, cc = np.nonzero(s == h)
        counts[name] = int(rr.size)
        got = 0
        for k in rng.permutation(rr.size):
            r, c = int(rr[k]), int(cc[k])
            if all((r - a) ** 2 + (c - b) ** 2 >= sep2 for a, b, _ in pts):
                pts.append((r, c, name))
                got += 1
            if got >= n_per_stratum[h]:
                break
    order = rng.permutation(len(pts))
    feats, key = [], {}
    for i, k in enumerate(order, start=1):
        r, c, name = pts[k]
        x, y = grid.xy(r, c)
        g = transform_geom(grid.crs, "EPSG:4326", {"type": "Point", "coordinates": [float(x), float(y)]})
        feats.append({"type": "Feature", "geometry": g, "properties": {"id": f"C{i}", "label": ""}})
        key[f"C{i}"] = {"stratum": name, "row": r, "col": c}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "change_points.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    (out / "change_points_key.json").write_text(json.dumps(
        {"strata_pixel_counts": counts, "points": key, "run_dir": str(run_dir), "seed": seed}, indent=1))
    return {"points": len(feats), "strata_pixel_counts": counts,
            "files": [str(out / "change_points.geojson"), str(out / "change_points_key.json")]}


def _read_labels(path) -> dict:
    gj = json.loads(Path(path).read_text())
    out = {}
    for f in gj["features"]:
        p = {str(k).lower(): v for k, v in (f.get("properties") or {}).items()}
        lab = str(p.get("label") or "").strip().lower()
        out[str(p.get("id"))] = True if lab in YES else False if lab in NO else None
    return out


def weighted_scores(truth, pred, w) -> dict:
    tp = float(w[truth & pred].sum()); fp = float(w[~truth & pred].sum())
    fn = float(w[truth & ~pred].sum()); tn = float(w[~truth & ~pred].sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec and np.isfinite(prec + rec) else float("nan")
    return {"precision": prec, "recall": rec, "f1": f1, "overall_accuracy": (tp + tn) / (tp + fp + fn + tn),
            "change_area_fraction": (tp + fn) / (tp + fp + fn + tn)}


def change_accuracy(cfg: dict, labels_path, key_path, run_dirs: list, n_boot: int = 2000, seed: int = 3) -> dict:
    key = json.loads(Path(key_path).read_text())
    labels = _read_labels(labels_path)
    ids = [i for i, v in labels.items() if v is not None and i in key["points"]]
    strata = np.array([key["points"][i]["stratum"] for i in ids])
    rows = np.array([key["points"][i]["row"] for i in ids])
    cols = np.array([key["points"][i]["col"] for i in ids])
    truth = np.array([labels[i] for i in ids])
    N = key["strata_pixel_counts"]
    n_lab = {h: int((strata == h).sum()) for h in STRATA}
    w = np.array([N[h] / n_lab[h] for h in strata], float)
    maps = {}
    for rd in run_dirs:
        maps[f"{Path(rd).name}: class change"] = read_raster(Path(rd) / "maps" / "transition.tif").data[0][rows, cols] != 0
    maps["rule baseline (classes + CVA)"] = read_raster(Path(run_dirs[0]) / "maps" / "baseline_change.tif").data[0][rows, cols].astype(bool)
    rng = np.random.default_rng(seed)
    res = {"n_labelled": n_lab, "n_skipped_or_blank": len(labels) - len(ids),
           "strata_pixel_counts": N, "labelled_change_by_stratum": {h: int(truth[strata == h].sum()) for h in STRATA},
           "maps": {}}
    empty = [h for h in STRATA if N.get(h, 0) > 0 and n_lab[h] == 0]
    if empty:
        res["warning"] = f"strata {empty} have pixels but no usable labels; their area is not represented"
    idx_by = {h: np.flatnonzero(strata == h) for h in STRATA}
    for name, pred in maps.items():
        est = weighted_scores(truth, pred, w)
        boots = []
        for _ in range(n_boot):
            ii = np.concatenate([rng.choice(v, v.size, replace=True) for v in idx_by.values() if v.size])
            boots.append(weighted_scores(truth[ii], pred[ii], w[ii]))
        ci = {k: [float(np.nanpercentile([b[k] for b in boots], 2.5)), float(np.nanpercentile([b[k] for b in boots], 97.5))]
              for k in est}
        res["maps"][name] = {"estimate": est, "ci95": ci}
    return res


def summary_markdown(r: dict) -> str:
    L = ["# Experiment C: change-detection accuracy, test strip (stratified, area-weighted)", "",
         f"labelled per stratum: {r['n_labelled']}; skipped/blank: {r['n_skipped_or_blank']}",
         f"labelled 'change' per stratum: {r['labelled_change_by_stratum']}",
         f"stratum sizes (pixels): {r['strata_pixel_counts']}", r.get("warning", ""), "",
         "| map | precision | recall | F1 | overall accuracy | est. change area |", "|---|---|---|---|---|---|"]
    f = lambda e, c, k: f"{e[k]:.2f} ({c[k][0]:.2f}-{c[k][1]:.2f})" if np.isfinite(e[k]) else "n/a"
    for name, v in r["maps"].items():
        e, c = v["estimate"], v["ci95"]
        L.append(f"| {name} | {f(e, c, 'precision')} | {f(e, c, 'recall')} | {f(e, c, 'f1')} | "
                 f"{f(e, c, 'overall_accuracy')} | {e['change_area_fraction']:.1%} |")
    L += ["", "Values in brackets: 95% bootstrap intervals. Labels are one analyst's interpretation of 10 m imagery."]
    return "\n".join(L) + "\n"
