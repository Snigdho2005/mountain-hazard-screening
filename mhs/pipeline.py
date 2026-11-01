"""End-to-end screening run on prepared inputs (paper Table 1 and Section 4.2)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from . import CLASSES, IGNORE_LABEL, __version__
from .baseline import difference_change, rule_classes
from .change import (alternative_explanations, class_change, index_differences, transition_summary)
from .config import save_config
from .exposure import asset_masks, exposure_in_corridor, load_assets
from .instability import zone_record
from .io import Raster, read_raster, same_grid, write_raster
from .physics import (Dist, area_thickness_samples, changed_volume, energy_interval, mass_interval,
                      thickness_scenario_ok)
from .preprocess import (build_features, coregister, season_gap_days, slope_aspect, spectral_indices,
                         valid_mask)
from .review import suggest_state
from .illumination import acquisition_sun, shadow_state
from .routing import FlowGrid, linked_sources, route_from_zone
from .zones import finalise_zones, form_zones

STATE_ORDER = {"review_candidate_for_escalation": 0, "review": 1, "routine": 2}


def _sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _date(r: Raster, fallback: str) -> str:
    return r.tags.get("date", fallback)


def classify(img: Raster, slope, model=None, meta=None, device="cpu"):
    """Return (class map uint8, probabilities or None)."""
    if model is None:
        return rule_classes(img, slope), None
    from .model.train import predict_scene

    feats = build_features(img, slope, meta["features"])
    probs = predict_scene(model, meta, feats, device)
    cls = np.nan_to_num(probs, nan=-1).argmax(0).astype(np.uint8)
    cls[~valid_mask(img)] = IGNORE_LABEL
    return cls, probs


def run_pipeline(cfg: dict, model_path: str | None = None, out_dir: str | None = None,
                 log=print, device: str = "cpu") -> dict:
    inp = cfg["inputs"]
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(out_dir or Path(cfg["output"]["dir"]) / f"{cfg['case']['name']}_{run_id}")
    out.mkdir(parents=True, exist_ok=True)
    log(f"run directory: {out}")

    # ---------------------------------------------------------------- inputs
    t1, t2, dem_r = read_raster(inp["t1"]), read_raster(inp["t2"]), read_raster(inp["dem"])
    t3 = read_raster(inp["t3"]) if inp.get("t3") else None
    for name, r in (("t2", t2), ("dem", dem_r)) + ((("t3", t3),) if t3 else ()):
        if not same_grid(t1, r):
            raise ValueError(f"{name} is not on the t1 grid; resample all inputs to one grid first")
    d1, d2 = _date(t1, "t1"), _date(t2, "t2")
    pp = cfg["preprocess"]
    gap = season_gap_days(d1, d2) if d1[:4].isdigit() and d2[:4].isdigit() else 0
    qc = {"dates": {"t1": d1, "t2": d2, "t3": _date(t3, "t3") if t3 else None}, "season_gap_days": gap}
    if gap > pp["max_season_gap_days"]:
        log(f"WARNING: dates are {gap} days apart in season; seasonal snow may appear as change")
    t2, qc["coreg_t2"] = coregister(t1, t2, max_shift_px=pp["max_shift_px"], reject_shift_px=pp["reject_shift_px"])
    if qc["coreg_t2"]["rejected"]:
        raise RuntimeError(f"t2 misregistration {qc['coreg_t2']['shift_px']:.2f} px: pair rejected")
    if t3 is not None:
        t3, qc["coreg_t3"] = coregister(t1, t3, max_shift_px=pp["max_shift_px"],
                                        reject_shift_px=pp["reject_shift_px"])
        if qc["coreg_t3"]["rejected"]:
            log("third date rejected for misregistration; persistence cannot be confirmed from it")
            t3 = None
    dem = dem_r.data[0].astype(np.float32)
    slope, aspect = slope_aspect(dem, t1.res)
    from rasterio.warp import transform as _tr

    cx, cy = t1.xy(t1.shape[0] / 2, t1.shape[1] / 2)
    lon, lat = (v[0] for v in _tr(t1.crs, "EPSG:4326", [float(cx)], [float(cy)]))
    ccfg = cfg["change"]
    log("illumination and cast shadows ...")
    sun1, sun2 = acquisition_sun(t1, lat, lon), acquisition_sun(t2, lat, lon)
    sh1, _ = shadow_state(dem, t1.res, slope, aspect, sun1, ccfg["min_cos_incidence"])
    sh2, _ = shadow_state(dem, t1.res, slope, aspect, sun2, ccfg["min_cos_incidence"])
    pseudo = sh1 != sh2
    both_shadow = sh1 & sh2
    qc["illumination"] = {"sun_t1": sun1, "sun_t2": sun2, "shadow_t1": float(sh1.mean()),
                          "shadow_t2": float(sh2.mean()), "shadow_state_differs": float(pseudo.mean()),
                          "shadow_both": float(both_shadow.mean())}
    if sun1.get("elevation") is not None and sun2.get("elevation") is not None:
        log(f"   sun t1 el {sun1['elevation']:.1f} az {sun1['azimuth']:.0f}; t2 el {sun2['elevation']:.1f} "
            f"az {sun2['azimuth']:.0f}; shadow t1 {sh1.mean():.0%}, t2 {sh2.mean():.0%}, "
            f"state differs {pseudo.mean():.0%}")
    log("routing grid ...")
    fg = FlowGrid(dem, t1.res)

    # ---------------------------------------------------------------- surface mapping
    model = meta = None
    if model_path:
        from .model.train import load_unet

        model, meta = load_unet(model_path, device)
        log(f"surface mapping with U-Net {model_path}")
    else:
        log("no model supplied: surface mapping with the rule baseline")
    cls1, p1 = classify(t1, slope, model, meta, device)
    cls2, p2 = classify(t2, slope, model, meta, device)
    cls3 = classify(t3, slope, model, meta, device)[0] if t3 is not None else None

    # ---------------------------------------------------------------- change
    ch = class_change(cls1, cls2, p1, p2, ccfg["min_confidence"], ccfg["use_probability_gate"])
    if ccfg["illumination_gate"]:
        removed_px = int((ch["changed"] & pseudo).sum())
        ch["changed"] &= ~pseudo
        ch["transition"][pseudo] = 0
        qc["illumination"]["changed_px_removed"] = removed_px
        log(f"   illumination gate removed {removed_px * t1.pixel_area / 1e6:.2f} km2 of class change")
    pairs = ccfg.get("ignore_transitions") or []
    if pairs:
        drop = np.zeros(cls1.shape, bool)
        for a, b in pairs:
            ia, ib = CLASSES.index(a), CLASSES.index(b)
            drop |= ((cls1 == ia) & (cls2 == ib)) | ((cls1 == ib) & (cls2 == ia))
        drop &= ch["changed"]
        ch["changed"] &= ~drop
        ch["transition"][drop] = 0
        qc["ignored_transitions"] = {"pairs": pairs, "pixels": int(drop.sum())}
        log(f"   ignored transitions {pairs}: {drop.sum() * t1.pixel_area / 1e6:.2f} km2 removed")
    idx1, idx2 = spectral_indices(t1), spectral_indices(t2)
    didx = index_differences(idx1, idx2)
    # baseline: rule-class change confirmed by direct image differencing
    b1, b2 = rule_classes(t1, slope), rule_classes(t2, slope)
    diff_mask, cva = difference_change(t1, t2)
    base_changed = (b1 != b2) & (b1 != IGNORE_LABEL) & (b2 != IGNORE_LABEL) & diff_mask

    # ---------------------------------------------------------------- zones
    zcfg = cfg["zones"]
    confirm = None
    if inp.get("independent_confirmation"):
        from .exposure import asset_masks as _am, load_assets as _la

        ms = _am(_la(inp["independent_confirmation"], t1), t1)
        confirm = np.any(ms, axis=0) if ms else None
    lab, comps = form_zones(ch, slope, dem, t1, zcfg, fg.acc_km2)
    d3 = _date(t3, "t3") if t3 is not None else None
    t3_same = True
    if d3 and d3[:4].isdigit() and d2[:4].isdigit():
        qc["season_gap_t2_t3_days"] = season_gap_days(d2, d3)
        t3_same = qc["season_gap_t2_t3_days"] <= pp["max_season_gap_days"]
        if not t3_same:
            log(f"NOTE: t3 is {qc['season_gap_t2_t3_days']} days from t2 in season; "
                "snow transitions cannot be confirmed by it")
    lab, zones, removed = finalise_zones(lab, comps, zcfg, cls3, confirm, t1.pixel_area, t3_same)
    log(f"{len(zones)} candidate source zones ({len(removed)} removed by slope rules)")
    for t in transition_summary(ch["transition"], t1.pixel_area)[:6]:
        log(f"   {t['from']:>10s} -> {t['to']:<10s} {t['area_m2'] / 1e6:8.2f} km2")
    n_valid = int(np.sum(ch["valid"]))
    log(f"   usable on both dates: {n_valid * t1.pixel_area / 1e6:.1f} km2 "
        f"({n_valid / ch['valid'].size:.0%} of scene)")

    # ---------------------------------------------------------------- physics, routes, exposure
    phy, rcfg = cfg["physics"], cfg["routing"]
    dz = None
    if inp.get("dem_t1") and inp.get("dem_t2"):
        a, b = read_raster(inp["dem_t1"]), read_raster(inp["dem_t2"])
        if same_grid(t1, a) and same_grid(t1, b):
            dz = (b.data[0] - a.data[0]).astype(np.float32)
        else:
            log("WARNING: multi-temporal DEMs not on grid; changed volume skipped")
    assets = masks = None
    if inp.get("assets"):
        assets = load_assets(inp["assets"], t1)
        masks = asset_masks(assets, t1)

    routes, records = {}, []
    corridor_any = np.zeros(t1.shape, bool)
    for z in zones:
        zmask = lab == z.id
        rec = z.to_dict()
        b1v = float(np.nanmean(idx1["BRIGHT"][zmask]))
        b2v = float(np.nanmean(idx2["BRIGHT"][zmask]))
        rec["alt_explanations"] = alternative_explanations(z.transition_from, z.transition_to, b1v, b2v, gap,
                                                           pp["max_season_gap_days"])
        rec["evidence"] = {k: float(np.nanmean(v[zmask])) for k, v in didx.items()}
        rec["evidence"]["cva_mean"] = float(np.nanmean(cva[zmask]))
        rec["evidence"]["baseline_agreement"] = float(base_changed[zmask].mean())
        rec["evidence"]["shadow_both_dates_fraction"] = float(both_shadow[zmask].mean())
        if both_shadow[zmask].mean() > 0.5:
            rec["alt_explanations"].append("shadowed_on_both_dates")
        rec["instability"] = zone_record(z.persistence, 1.0, z.slope_median_deg)

        rt = route_from_zone(fg, zmask, slope, rcfg)
        routes[z.id] = rt
        corridor_any |= rt.corridor
        h_ref = phy["reference_elevation_m"] if phy["reference_elevation_m"] is not None else rt.reference_elevation_m
        drop = max(0.0, z.elev_mean_m - h_ref)
        rec["route"] = {"length_m": rt.length_m, "reference_elevation_m": float(h_ref),
                        "reference_distance_m": float(rt.distance_m[rt.reference_index]),
                        "drop_m": drop, "qc_flags": rt.qc_flags,
                        "end_xy": [float(v) for v in t1.xy(rt.path[-1] // t1.shape[1], rt.path[-1] % t1.shape[1])]}

        rho_lo, rho_hi = phy["density_kg_m3"].get(z.source_type, [1800, 2600])
        rho = Dist("uniform", rho_lo, rho_hi)
        p = {"changed": None, "available": None}
        if dz is not None:
            cv = changed_volume(dz, zmask, t1.pixel_area, phy["dz_sigma_m"], phy["dz_correlation_m"])
            if cv.get("available"):
                vol = abs(cv["loss_m3"]) if cv["loss_m3"] < 0 else abs(cv["net_m3"])
                cv["mass"] = mass_interval(Dist("normal", vol, cv["sigma_m3"]), rho, phy["n_samples"],
                                           phy["interval"], phy["seed"])
            p["changed"] = cv
        scen = phy["thickness_scenarios_m"].get(z.source_type)
        if scen is None:
            p["available"] = {"status": "area_change_record_only",
                              "reason": "no independently supported thickness/depth declared for this source type"}
        elif not thickness_scenario_ok(scen, phy["max_thickness_ratio"]):
            p["available"] = {"status": "area_change_record_only",
                              "reason": f"thickness range {scen} too broad (ratio > {phy['max_thickness_ratio']})"}
        else:
            vs = area_thickness_samples(z.area_m2, phy["area_rel_sigma"], scen, phy["n_samples"], phy["seed"])
            dh_sig = phy["dz_sigma_m"] * np.sqrt(2)
            e = energy_interval(vs, rho, Dist("normal", drop, dh_sig), phy["n_samples"], phy["interval"],
                                phy["energy_threshold_J"], phy["seed"], phy["max_interval_ratio"])
            e["status"] = "inconclusive" if e["inconclusive"] else "estimated"
            e["thickness_scenario_m"] = scen
            e["volume_m3"] = {"median": float(np.median(vs)), "low": float(np.quantile(vs, 0.05)),
                              "high": float(np.quantile(vs, 0.95))}
            p["available"] = e
        rec["physics"] = p
        if assets:
            rec["exposure"] = exposure_in_corridor(assets, masks, rt.corridor, rt.path, rt.distance_m, t1.shape[1])
        else:
            rec["exposure"] = None
        rec["review"] = suggest_state(rec)
        records.append(rec)

    links = linked_sources(routes, lab)
    for rec in records:
        rec["linked_downstream_zones"] = links.get(rec["id"], [])

    def _key(r):
        e = (r["physics"]["available"] or {}).get("energy_J", {}).get("median", 0.0)
        n_exp = len(r["exposure"]["assets"]) if r["exposure"] else 0
        pers = {"persistent": 0, "unconfirmed": 1, "not_persistent": 2}[r["persistence"]]
        chg = abs((r["physics"]["changed"] or {}).get("loss_m3") or 0.0)
        # without energy or exposure the order falls back to persistence, DEM loss, then size
        n_alt = len(r["alt_explanations"])  # fewer competing explanations first
        return (STATE_ORDER[r["review"]["suggested_state"]], -n_exp, -e, pers, -chg, n_alt, -r["area_m2"])

    records.sort(key=_key)
    for k, r in enumerate(records, start=1):
        r["review_order"] = k

    # ---------------------------------------------------------------- outputs
    from .report import write_outputs

    result = {
        "case": cfg["case"], "run_dir": str(out), "method": "unet" if model is not None else "rule_baseline",
        "qc": qc, "transitions": transition_summary(ch["transition"], t1.pixel_area),
        "n_zones": len(records), "removed_components": removed, "zones": records,
        "baseline": {"changed_pixels": int(base_changed.sum()), "method": "rule classes + CVA threshold"},
        "disclaimer": ("Screening product for specialist review. Energy is available gravitational potential "
                       "energy only; it is not a failure probability, runout, impact force or loss estimate. "
                       "Routes are screening corridors, not runout or inundation boundaries."),
    }
    layers = {"cls1": cls1, "cls2": cls2, "cls3": cls3, "change": ch["transition"], "zones": lab,
              "corridors": corridor_any, "baseline_change": base_changed, "base1": b1, "base2": b2, "slope": slope, "dem": dem,
              "acc_km2": fg.acc_km2, "routes": routes, "t1": t1, "t2": t2, "assets": assets}
    write_outputs(result, layers, t1, out, cfg, figures=cfg["output"]["figures"])

    manifest = {
        "run_id": run_id, "mhs_version": __version__, "python": platform.python_version(),
        "platform": platform.platform(), "numpy": np.__version__,
        "inputs": {k: ({"path": v, "sha256": _sha256(v)} if isinstance(v, str) and Path(v).exists() else v)
                   for k, v in inp.items()},
        "model": {"path": model_path, "sha256": _sha256(model_path) if model_path else None,
                  "meta": meta},
        "classes": CLASSES,
    }
    try:
        import torch

        manifest["torch"] = torch.__version__
    except ImportError:
        pass
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    save_config(cfg, out / "config_used.yaml")
    log(f"done: {len(records)} zones; report at {out / 'report.md'}")
    return result
