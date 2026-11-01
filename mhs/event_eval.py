"""Retrospective event evaluation (paper Section 8).

Three tests against a documented event, declared in the config under `evaluation.event`
before any of them is run:

  ranking     Does a zone at the documented source appear among the highest-priority zones?
  removal     Re-rank with energy, instability, pathway or exposure removed, one at a time,
              and compare with simple baselines (area only; change confidence then area).
  pathway     Does the DEM route from the documented source pass the documented downstream
              points, and at what along-route distance compared with published distances?
  exposure    Which assets of an independent inventory fall in that route's corridor, and do
              they include the documented damaged assets?

Energy. When no thickness scenario is declared (the default; see physics config) the pipeline
assigns no available energy. Ranking then uses an energy *index*: rho * g * area * drop, i.e.
available potential energy per metre of source thickness. It orders zones as energy would if
all sources had the same thickness. It is labelled as such and is never reported as energy.

Nothing here changes the pipeline outputs; it only reads a finished run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rasterio.transform import rowcol
from rasterio.warp import transform as _tr

from . import CLASSES, G, IGNORE_LABEL
from .io import read_raster

COMPONENTS = ("energy", "instability", "route", "exposure")
PERS = {"persistent": 0, "unconfirmed": 1, "not_persistent": 2}
VARIANTS = {
    "full": set(COMPONENTS),
    "no_energy": {"instability", "route", "exposure"},
    "no_instability": {"energy", "route", "exposure"},
    "no_exposure": {"energy", "instability", "route"},
    # exposure is counted inside the route corridor, so removing the route removes it too
    "no_route_(and_corridor_exposure)": {"energy", "instability"},
}


def _rc(grid, lon, lat):
    xs, ys = _tr("EPSG:4326", grid.crs, [float(lon)], [float(lat)])
    r, c = rowcol(grid.transform, xs[0], ys[0])
    return int(r), int(c)


def _disk(shape, r, c, radius_px):
    rr, cc = np.ogrid[:shape[0], :shape[1]]
    return (rr - r) ** 2 + (cc - c) ** 2 <= radius_px ** 2


def zone_components(rec: dict, cfg: dict) -> dict:
    dens = cfg["physics"]["density_kg_m3"].get(rec["source_type"], [1800, 2600])
    drop = float(rec["route"]["drop_m"])
    e = ((rec["physics"]["available"] or {}).get("energy_J") or {}).get("median")
    flags = rec["route"]["qc_flags"]
    route_ok = (drop > 0 and "route_too_short" not in flags
                and not any(f.startswith("no_valley_floor") for f in flags))
    return {
        "energy": float(e) if e is not None else 0.5 * (dens[0] + dens[1]) * G * rec["area_m2"] * drop,
        "energy_is_index_J_per_m": e is None,
        "persistence": rec["persistence"],
        "signals": (rec.get("instability") or {}).get("independent_signals", 0),
        "route_ok": route_ok,
        "n_exp": len(rec["exposure"]["assets"]) if rec.get("exposure") else 0,
        "confidence": float(rec["confidence"]),
        "n_alt": len(rec.get("alt_explanations") or []),
        "area": float(rec["area_m2"]),
    }


def rank_key(c: dict, use: set):
    """Same order as the pipeline (review state, exposure, energy, persistence, ...), restricted to `use`."""
    pers_used = "instability" in use
    weak = c["confidence"] < 0.6 or (pers_used and c["persistence"] == "not_persistent")
    route_ok = c["route_ok"] if "route" in use else True
    exposed = "exposure" in use and c["n_exp"] > 0
    if weak or not route_ok:
        state = 2
    elif pers_used and c["persistence"] == "persistent" and exposed and c["signals"] >= 2:
        state = 0
    else:
        state = 1
    return (state,
            -c["n_exp"] if "exposure" in use else 0,
            -c["energy"] if "energy" in use else 0,
            PERS[c["persistence"]] if pers_used else 0,
            c["n_alt"], -c["area"])


def rankings(records: list[dict], cfg: dict) -> dict[str, dict[int, int]]:
    comps = {r["id"]: zone_components(r, cfg) for r in records}
    out = {}
    for name, use in VARIANTS.items():
        order = sorted(comps, key=lambda i: rank_key(comps[i], use))
        out[name] = {zid: k for k, zid in enumerate(order, start=1)}
    for name, key in (("baseline_area_only", lambda i: -comps[i]["area"]),
                      ("baseline_confidence_then_area", lambda i: (-comps[i]["confidence"], -comps[i]["area"]))):
        order = sorted(comps, key=key)
        out[name] = {zid: k for k, zid in enumerate(order, start=1)}
    out["pipeline_as_run"] = {r["id"]: r["review_order"] for r in records}
    return out


def _route_checks(rt, grid, checkpoints, tol_m):
    w = grid.shape[1]
    pr, pc = rt.path // w, rt.path % w
    out = []
    for cp in checkpoints:
        r, c = _rc(grid, cp["lon"], cp["lat"])
        d = np.hypot(pr - r, pc - c) * grid.res
        k = int(np.argmin(d))
        item = {"name": cp["name"], "closest_m": round(float(d[k]), 1), "reached": bool(d[k] <= tol_m),
                "along_route_km": round(float(rt.distance_m[k]) / 1000, 2), "path_index": k,
                "published_km": cp.get("published_km"), "ref": cp.get("ref")}
        if item["reached"] and cp.get("published_km"):
            item["difference_km"] = round(item["along_route_km"] - cp["published_km"], 2)
        out.append(item)
    idx = [c["path_index"] for c in out if c["reached"]]
    return {"length_km": round(rt.length_m / 1000, 2), "qc_flags": rt.qc_flags,
            "reference_elevation_m": round(rt.reference_elevation_m, 1), "checkpoints": out,
            "n_reached": len(idx), "n_checkpoints": len(out),
            "reached_in_order": bool(idx == sorted(idx))}


def _exposure_checks(cfg, grid, rt, damaged, tol_m=500.0):
    from .exposure import asset_masks, exposure_in_corridor, load_assets

    path = cfg["inputs"].get("assets")
    out = {"inventory": path}
    if not path or not Path(path).exists():
        out["status"] = "no asset inventory supplied"
        return out
    assets = load_assets(path, grid)
    hits = exposure_in_corridor(assets, asset_masks(assets, grid), rt.corridor, rt.path, rt.distance_m,
                                grid.shape[1])["assets"]
    by_id = {a["id"]: a for a in assets}
    w = grid.shape[1]
    out["inventory_size"] = len(assets)
    out["assets_in_corridor"] = [{k: h[k] for k in ("name", "type", "along_route_m")} for h in hits]
    dmg = []
    for d in damaged:
        r, c = _rc(grid, d["lon"], d["lat"])
        in_corr = bool(0 <= r < grid.shape[0] and 0 <= c < w and rt.corridor[r, c])
        near = []
        for h in hits:
            from shapely.geometry import shape
            g = shape(by_id[h["id"]]["geometry"]).centroid
            ar, ac = rowcol(grid.transform, g.x, g.y)
            if np.hypot(ar - r, ac - c) * grid.res <= tol_m:
                near.append(h["name"])
        dmg.append({"name": d["name"], "location_in_corridor": in_corr, "inventory_assets_within_500m": near})
    out["documented_damage"] = dmg
    out["damaged_sites_in_corridor"] = sum(d["location_in_corridor"] for d in dmg)
    out["damaged_sites_matched_by_inventory"] = sum(bool(d["inventory_assets_within_500m"]) for d in dmg)
    return out


def evaluate_event(cfg: dict, run_dir: str | Path, log=print) -> dict:
    from .illumination import acquisition_sun, shadow_state
    from .preprocess import slope_aspect
    from .routing import FlowGrid, route_from_zone

    ev_cfg = cfg["evaluation"]
    ev = ev_cfg.get("event")
    if not ev:
        raise ValueError("no evaluation.event declared in the config")
    run = Path(run_dir)
    result = json.loads((run / "report.json").read_text())
    records = result["zones"]
    zones = read_raster(run / "maps" / "zones.tif").data[0]
    cls1 = read_raster(run / "maps" / "cls1.tif").data[0]
    cls2 = read_raster(run / "maps" / "cls2.tif").data[0]
    trans = read_raster(run / "maps" / "transition.tif").data[0]
    t1, t2 = read_raster(cfg["inputs"]["t1"]), read_raster(cfg["inputs"]["t2"])
    dem = read_raster(cfg["inputs"]["dem"]).data[0].astype(np.float32)
    slope, aspect = slope_aspect(dem, t1.res)
    n = len(records)
    out = {"event": ev["name"], "run_dir": str(run), "method": result.get("method"), "n_zones": n}

    # ------------------------------------------------------------------ source diagnostics
    sr, sc = _rc(t1, ev["source"]["lon"], ev["source"]["lat"])
    if not (0 <= sr < t1.shape[0] and 0 <= sc < t1.shape[1]):
        raise ValueError("documented source lies outside the scene")
    cx, cy = t1.xy(t1.shape[0] / 2, t1.shape[1] / 2)
    lon_c, lat_c = (v[0] for v in _tr(t1.crs, "EPSG:4326", [float(cx)], [float(cy)]))
    sh1, _ = shadow_state(dem, t1.res, slope, aspect, acquisition_sun(t1, lat_c, lon_c), cfg["change"]["min_cos_incidence"])
    sh2, _ = shadow_state(dem, t1.res, slope, aspect, acquisition_sun(t2, lat_c, lon_c), cfg["change"]["min_cos_incidence"])
    disk = _disk(t1.shape, sr, sc, 1000.0 / t1.res)
    valid = (cls1 != IGNORE_LABEL) & (cls2 != IGNORE_LABEL)

    def _frac(cls):
        v = cls[disk & (cls != IGNORE_LABEL)]
        return {CLASSES[k]: round(float((v == k).mean()), 3) for k in range(len(CLASSES)) if v.size and (v == k).any()}

    out["source_diagnostics_1km"] = {
        "elevation_m": round(float(dem[sr, sc]), 1), "slope_deg": round(float(slope[sr, sc]), 1),
        "aspect_deg": round(float(aspect[sr, sc]), 1),
        "usable_both_dates": round(float(valid[disk].mean()), 3),
        "shadow_t1": round(float(sh1[disk].mean()), 3), "shadow_t2": round(float(sh2[disk].mean()), 3),
        "shadow_both": round(float((sh1 & sh2)[disk].mean()), 3),
        "class_fractions_t1": _frac(cls1), "class_fractions_t2": _frac(cls2),
        "changed_pixels_before_zone_rules": int((trans[disk] != 0).sum()),
        "in_zones_pixels": int((zones[disk] > 0).sum()),
    }

    # ------------------------------------------------------------------ ranking + removal tests
    ranks = rankings(records, cfg)
    rec_by_id = {r["id"]: r for r in records}
    by_radius = {}
    for rad in ev_cfg["source_radius_m"]:
        d = _disk(t1.shape, sr, sc, rad / t1.res)
        hit_ids = sorted(int(v) for v in np.unique(zones[d]) if v > 0)
        m = len(hit_ids)
        row = {"zones_matched": m,
               "matched": [{"id": i, "transition": f"{rec_by_id[i]['transition_from']}->{rec_by_id[i]['transition_to']}",
                            "area_m2": rec_by_id[i]["area_m2"], "persistence": rec_by_id[i]["persistence"]}
                           for i in hit_ids],
               "random_expected_best_rank": round((n + 1) / (m + 1), 1) if m else None, "variants": {}}
        for name, rk in ranks.items():
            if not m:
                row["variants"][name] = {"best_rank": None}
                continue
            best = min(rk[i] for i in hit_ids)
            row["variants"][name] = {"best_rank": best, "percentile": round(best / n, 3),
                                     **{f"top_{k}": best <= k for k in ev_cfg["top_k"]}}
        by_radius[str(rad)] = row
    out["ranking"] = by_radius
    out["energy_note"] = ("energy term is an index (rho*g*area*drop, J per metre of thickness): no thickness "
                          "scenario declared" if all(zone_components(r, cfg)["energy_is_index_J_per_m"] for r in records)
                          else "available-energy median where estimated")

    # ------------------------------------------------------------------ pathway + exposure
    log("routing from the documented source ...")
    fg = FlowGrid(dem, t1.res)
    rt = route_from_zone(fg, _disk(t1.shape, sr, sc, max(1.0, 50.0 / t1.res)), slope, cfg["routing"])
    tol = ev_cfg["checkpoint_tolerance_m"]
    out["pathway_from_documented_source"] = _route_checks(rt, t1, ev.get("checkpoints", []), tol)
    out["exposure_from_documented_source"] = _exposure_checks(cfg, t1, rt, ev.get("damaged_assets", []))
    rad = str(ev_cfg["source_radius_m"][len(ev_cfg["source_radius_m"]) // 2])
    if by_radius[rad]["zones_matched"]:
        full = ranks["full"]
        best_id = min((m["id"] for m in by_radius[rad]["matched"]), key=lambda i: full[i])
        rz = route_from_zone(fg, zones == best_id, slope, cfg["routing"])
        out["pathway_from_best_matched_zone"] = {"zone_id": best_id, **_route_checks(rz, t1, ev.get("checkpoints", []), tol)}
    out["_route_path"] = rt.path.tolist()
    return out


def summary_markdown(ev: dict) -> str:
    L = [f"# Event evaluation: {ev['event']}", f"run: `{ev['run_dir']}` ({ev['method']}), {ev['n_zones']} zones", ""]
    sd = ev["source_diagnostics_1km"]
    L += ["## Documented source (within 1 km)",
          f"elevation {sd['elevation_m']} m, slope {sd['slope_deg']} deg, aspect {sd['aspect_deg']} deg; "
          f"usable on both dates {sd['usable_both_dates']:.0%}; shadow t1 {sd['shadow_t1']:.0%}, "
          f"t2 {sd['shadow_t2']:.0%}, both {sd['shadow_both']:.0%}",
          f"classes t1 {sd['class_fractions_t1']}; t2 {sd['class_fractions_t2']}",
          f"changed pixels before zone rules {sd['changed_pixels_before_zone_rules']}; in zones {sd['in_zones_pixels']}", ""]
    L.append(f"## Ranking of the documented source ({ev['energy_note']})")
    for rad, row in ev["ranking"].items():
        L.append(f"\n### zones within {rad} m: {row['zones_matched']} "
                 f"(random expectation for best rank: {row['random_expected_best_rank']})")
        if not row["zones_matched"]:
            L.append("no zone at the documented source")
            continue
        L.append("| ranking | best rank | percentile | top 5 | top 10 | top 25 |\n|---|---|---|---|---|---|")
        for name, v in row["variants"].items():
            L.append(f"| {name} | {v['best_rank']} | {v['percentile']:.1%} | {v.get('top_5')} | "
                     f"{v.get('top_10')} | {v.get('top_25')} |")
    for key in ("pathway_from_documented_source", "pathway_from_best_matched_zone"):
        p = ev.get(key)
        if not p:
            continue
        L += ["", f"## {key.replace('_', ' ')}",
              f"length {p['length_km']} km; reached {p['n_reached']}/{p['n_checkpoints']} checkpoints "
              f"(in order: {p['reached_in_order']}); flags {p['qc_flags']}",
              "| checkpoint | closest (m) | reached | along route (km) | published (km) |\n|---|---|---|---|---|"]
        for c in p["checkpoints"]:
            L.append(f"| {c['name']} | {c['closest_m']} | {c['reached']} | {c['along_route_km']} | {c['published_km']} |")
    x = ev["exposure_from_documented_source"]
    L += ["", "## Exposure along the route from the documented source"]
    if "documented_damage" in x:
        L.append(f"inventory {x['inventory_size']} assets; {len(x['assets_in_corridor'])} in corridor; documented "
                 f"damaged sites in corridor {x['damaged_sites_in_corridor']}/{len(x['documented_damage'])}, "
                 f"matched by inventory {x['damaged_sites_matched_by_inventory']}/{len(x['documented_damage'])}")
        for a in x["assets_in_corridor"]:
            L.append(f"- {a['name']} ({a['type']}) at {a['along_route_m'] / 1000:.1f} km")
    else:
        L.append(x["status"])
    return "\n".join(L) + "\n"


def event_figure(cfg: dict, ev: dict, run_dir: str | Path, path: str | Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .report import _hillshade

    t1 = read_raster(cfg["inputs"]["t1"])
    dem = read_raster(cfg["inputs"]["dem"]).data[0]
    zones = read_raster(Path(run_dir) / "maps" / "zones.tif").data[0]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(_hillshade(dem, t1.res), cmap="gray")
    ax.imshow(np.ma.masked_where(zones == 0, zones > 0), cmap="autumn", alpha=0.6)
    w = t1.shape[1]
    p = np.array(ev["_route_path"])
    ax.plot(p % w, p // w, color="deepskyblue", lw=2, label="DEM route from documented source")
    e = cfg["evaluation"]["event"]
    r, c = _rc(t1, e["source"]["lon"], e["source"]["lat"])
    ax.plot(c, r, "*", color="yellow", ms=18, mec="k", label="documented source")
    for cp in e.get("checkpoints", []):
        r, c = _rc(t1, cp["lon"], cp["lat"])
        ax.plot(c, r, "o", color="lime", ms=8, mec="k")
        ax.annotate(cp["name"], (c, r), xytext=(5, 5), textcoords="offset points", color="white", fontsize=8)
    ax.set_title("Documented source, DEM route and checkpoints; mapped zones in red")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
