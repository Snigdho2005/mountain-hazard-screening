"""Follow-up experiment A: source detection, volume and energy from a DEM difference.

Optical change detection missed the 2021 Chamoli source (shadowed north-east face). This module
asks whether elevation change finds it, using a published post-minus-pre DEM difference:

  1. Resample the difference onto the 10 m image grid (block average).
  2. Estimate DEM-difference noise on stable terrain (gentle, snow-free on both image dates, away from
     the documented event route): median (vertical bias, removed) and NMAD.
  3. "Lowered" pixels: dz <= -max(loss_threshold_m, k * NMAD). Clean, connect, keep >= min area.
  4. For every lowered zone: loss volume with spatially correlated error, mass-weighted source
     elevation, D8 route and valley reference, drop, Monte Carlo mass and energy (one declared density
     range for all zones, material being unknown), exposure along the corridor.
  5. Rank zones (same rules as the event evaluation; energy is now measured, not an index) and run the
     component-removal tests. Compare with simple baselines (largest area, largest volume).
  6. Against the documented event: is there a lowered zone at the source, where does it rank, does its
     volume interval contain the published volume, and where does its route go?

All thresholds are declared in config `dem_change` before the difference map is opened.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import G
from .event_eval import VARIANTS, _disk, _rc, _route_checks, _exposure_checks, rank_key
from .io import read_raster, regrid, write_raster


def stable_stats(dz: np.ndarray, stable: np.ndarray) -> dict:
    v = dz[stable & np.isfinite(dz)]
    if v.size < 1000:
        raise RuntimeError(f"only {v.size} stable pixels with DEM difference; cannot estimate noise")
    med = float(np.median(v))
    nmad = float(1.4826 * np.median(np.abs(v - med)))
    return {"n_pixels": int(v.size), "median_m": med, "nmad_m": nmad,
            "p05_m": float(np.percentile(v, 5)), "p95_m": float(np.percentile(v, 95))}


def lowered_zones(dz: np.ndarray, threshold_m: float, min_px: int, opening_px: int = 1):
    m = np.isfinite(dz) & (dz <= -threshold_m)
    if opening_px:
        m = ndimage.binary_opening(m, structure=np.ones((3, 3)), iterations=opening_px)
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    if n:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= min_px
        lab, n = ndimage.label(keep[lab], structure=np.ones((3, 3)))
    return lab, n


def dem_change(cfg: dict, out_dir: str | Path, diff_path: str | None = None, log=print) -> dict:
    from .exposure import asset_masks, exposure_in_corridor, load_assets
    from .physics import Dist, changed_volume, energy_interval
    from .preprocess import slope_aspect, spectral_indices
    from .routing import FlowGrid, route_from_zone

    dc, ev_cfg, phy = cfg["dem_change"], cfg["evaluation"], cfg["physics"]
    ev = ev_cfg.get("event")
    diff_path = diff_path or dc.get("diff")
    if not diff_path or not Path(diff_path).exists():
        raise FileNotFoundError(f"DEM difference not found: {diff_path}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t1, t2 = read_raster(cfg["inputs"]["t1"]), read_raster(cfg["inputs"]["t2"])
    dem = read_raster(cfg["inputs"]["dem"]).data[0].astype(np.float32)
    slope, _ = slope_aspect(dem, t1.res)
    log("resampling the DEM difference to the image grid ...")
    dz_raw = regrid(diff_path, t1, "average").data[0].astype(np.float32)
    coverage = float(np.isfinite(dz_raw).mean())
    log(f"   difference covers {coverage:.0%} of the scene")

    log("routing grid ...")
    fg = FlowGrid(dem, t1.res)
    corridor_ev = np.zeros(t1.shape, bool)
    src_rc = None
    if ev:
        src_rc = _rc(t1, ev["source"]["lon"], ev["source"]["lat"])
        rt_ev = route_from_zone(fg, _disk(t1.shape, *src_rc, max(1.0, 50.0 / t1.res)), slope, cfg["routing"])
        on = np.zeros(t1.shape[0] * t1.shape[1], bool)
        on[rt_ev.path] = True
        corridor_ev = ndimage.distance_transform_edt(~on.reshape(t1.shape)) * t1.res <= dc["stable_exclude_corridor_m"]

    ndsi1, ndsi2 = spectral_indices(t1)["NDSI"], spectral_indices(t2)["NDSI"]
    stable = ((slope < dc["stable_max_slope_deg"]) & ~corridor_ev
              & (np.nan_to_num(ndsi1, nan=1) < dc["stable_max_ndsi"])
              & (np.nan_to_num(ndsi2, nan=1) < dc["stable_max_ndsi"]))
    st = stable_stats(dz_raw, stable)
    dz = dz_raw - st["median_m"]                     # remove vertical bias measured on stable terrain
    thr = max(dc["loss_threshold_m"], dc["loss_sigma_factor"] * st["nmad_m"])
    log(f"   stable terrain: {st['n_pixels']} px, bias {st['median_m']:+.2f} m, NMAD {st['nmad_m']:.2f} m; "
        f"lowering threshold {thr:.2f} m")
    lab, n = lowered_zones(dz, thr, int(np.ceil(dc["min_area_m2"] / t1.pixel_area)), dc["opening_px"])
    log(f"{n} lowered zones")
    write_raster(out / "dem_loss_zones.tif", t1.like(lab[None].astype(np.int32), ["zone_id"], 0), "int32")
    write_raster(out / "dz_10m.tif", t1.like(dz[None], ["dz_m"], np.nan), "float32")

    assets = masks = None
    if cfg["inputs"].get("assets") and Path(cfg["inputs"]["assets"]).exists():
        assets = load_assets(cfg["inputs"]["assets"], t1)
        masks = asset_masks(assets, t1)
    rho = Dist("uniform", *dc["density_kg_m3"])
    zones, comps, routes = [], {}, {}
    for zid in range(1, n + 1):
        zm = lab == zid
        cv = changed_volume(dz, zm, t1.pixel_area, st["nmad_m"], phy["dz_correlation_m"])
        w = -dz[zm]
        w = np.where(np.isfinite(w) & (w > 0), w, 0)
        z_pre = dem[zm]
        zc = float(np.sum((z_pre + 0.5 * w) * w) / np.sum(w)) if w.sum() > 0 else float(np.nanmean(z_pre))
        rt = route_from_zone(fg, zm, slope, cfg["routing"])
        routes[zid] = rt
        drop = max(0.0, zc - rt.reference_elevation_m)
        vol = abs(cv["loss_m3"])
        e = energy_interval(Dist("normal", vol, cv["sigma_m3"]), rho, Dist("normal", drop, phy["dz_sigma_m"] * np.sqrt(2)),
                            phy["n_samples"], phy["interval"], phy["energy_threshold_J"], phy["seed"],
                            phy["max_interval_ratio"])
        n_exp = 0
        exp_names = []
        if assets:
            ex = exposure_in_corridor(assets, masks, rt.corridor, rt.path, rt.distance_m, t1.shape[1])
            n_exp, exp_names = len(ex["assets"]), [a["name"] for a in ex["assets"]]
        rr, cc = np.nonzero(zm)
        x, y = t1.xy(int(np.median(rr)), int(np.median(cc)))
        flags = rt.qc_flags
        route_ok = drop > 0 and "route_too_short" not in flags and not any(f.startswith("no_valley_floor") for f in flags)
        rec = {"id": zid, "area_m2": float(zm.sum() * t1.pixel_area), "loss_m3": vol, "sigma_m3": cv["sigma_m3"],
               "mean_dz_m": cv["mean_dz_m"], "max_lowering_m": float(np.nanmax(w)),
               "source_elevation_m": zc, "reference_elevation_m": rt.reference_elevation_m, "drop_m": drop,
               "route_km": round(rt.length_m / 1000, 2), "route_flags": flags,
               "mass_kg": e["mass_kg"], "energy_J": e["energy_J"], "energy_inconclusive": e["inconclusive"],
               "exposed_assets": n_exp, "exposed_names": exp_names, "centre_xy": [float(x), float(y)],
               "slope_median_deg": float(np.nanmedian(slope[zm]))}
        zones.append(rec)
        comps[zid] = {"energy": e["energy_J"]["median"], "persistence": "unconfirmed", "signals": 0,
                      "route_ok": route_ok, "n_exp": n_exp, "confidence": 1.0, "n_alt": 0, "area": rec["area_m2"]}

    ranks = {}
    for name, use in VARIANTS.items():
        order = sorted(comps, key=lambda i: rank_key(comps[i], use))
        ranks[name] = {zid: k for k, zid in enumerate(order, start=1)}
    by_rec = {z["id"]: z for z in zones}
    for name, key in (("baseline_area_only", lambda i: -by_rec[i]["area_m2"]),
                      ("baseline_volume_only", lambda i: -by_rec[i]["loss_m3"])):
        ranks[name] = {zid: k for k, zid in enumerate(sorted(comps, key=key), start=1)}
    for z in zones:
        z["rank_full"] = ranks["full"][z["id"]]
    zones.sort(key=lambda z: z["rank_full"])

    res = {"diff": str(diff_path), "coverage_of_scene": coverage, "stable_terrain": st, "threshold_m": thr,
           "n_zones": n, "density_kg_m3": dc["density_kg_m3"], "zones": zones,
           "note": ("Energy from measured lowering volume, one declared density range (ice to rock) and the drop "
                    "from the mass-weighted source elevation to the first valley-floor point on the D8 route. "
                    "Persistence is not available for DEM zones, so the instability component is constant.")}

    if ev:
        if not np.isfinite(dz_raw[src_rc]):
            res["source_covered"] = False
        else:
            res["source_covered"] = True
        res["source_dz_1km"] = {k: (round(float(f(dz[_disk(t1.shape, *src_rc, 1000 / t1.res)])), 2)
                                    if np.isfinite(dz[_disk(t1.shape, *src_rc, 1000 / t1.res)]).any() else None)
                                for k, f in (("min_m", np.nanmin), ("median_m", np.nanmedian))}
        by_radius = {}
        for rad in ev_cfg["source_radius_m"]:
            hit = sorted(int(v) for v in np.unique(lab[_disk(t1.shape, *src_rc, rad / t1.res)]) if v > 0)
            row = {"zones_matched": len(hit), "random_expected_best_rank": round((n + 1) / (len(hit) + 1), 1) if hit else None,
                   "variants": {}}
            for name, rk in ranks.items():
                if hit:
                    best = min(rk[i] for i in hit)
                    row["variants"][name] = {"best_rank": best, "percentile": round(best / n, 3),
                                             **{f"top_{k}": best <= k for k in ev_cfg["top_k"]}}
            by_radius[str(rad)] = row
        res["ranking"] = by_radius
        mid = str(ev_cfg["source_radius_m"][len(ev_cfg["source_radius_m"]) // 2])
        hit = sorted(int(v) for v in np.unique(lab[_disk(t1.shape, *src_rc, int(mid) / t1.res)]) if v > 0)
        if hit:
            best = min(hit, key=lambda i: by_rec[i]["loss_m3"] * -1)  # largest-volume zone at the source
            z = by_rec[best]
            pv = ev.get("published_volume_m3") or {}
            lo, hi = z["loss_m3"] - 1.96 * z["sigma_m3"], z["loss_m3"] + 1.96 * z["sigma_m3"]
            res["source_zone"] = {
                "id": best, "rank_full": z["rank_full"], "loss_m3": z["loss_m3"], "ci95_m3": [lo, hi],
                "area_m2": z["area_m2"], "max_lowering_m": z["max_lowering_m"],
                "source_elevation_m": z["source_elevation_m"], "drop_m": z["drop_m"],
                "energy_J": z["energy_J"], "mass_kg": z["mass_kg"],
                "published_volume_m3": pv.get("central"),
                "volume_ratio_to_published": (z["loss_m3"] / pv["central"]) if pv.get("central") else None,
                "published_in_ci95": (lo <= pv["central"] <= hi) if pv.get("central") else None,
                "published_source_elevation_m": ev.get("published_source_elevation_m"),
                "pathway": _route_checks(routes[best], t1, ev.get("checkpoints", []), ev_cfg["checkpoint_tolerance_m"]),
            }
        res["exposure_from_documented_source"] = _exposure_checks(cfg, t1, rt_ev, ev.get("damaged_assets", []))
        res["_route_path"] = rt_ev.path.tolist()
    return res


def summary_markdown(r: dict) -> str:
    st = r["stable_terrain"]
    L = ["# Experiment A: source detection from elevation change", "",
         f"difference: `{Path(r['diff']).name}`; covers {r['coverage_of_scene']:.0%} of the scene",
         f"stable terrain: {st['n_pixels']} px, bias {st['median_m']:+.2f} m (removed), NMAD {st['nmad_m']:.2f} m",
         f"lowering threshold: {r['threshold_m']:.2f} m; lowered zones: {r['n_zones']}; density {r['density_kg_m3']} kg/m3", ""]
    L += ["## Top 15 lowered zones (full ranking)",
          "| rank | zone | area (m2) | loss volume (m3) | max lowering (m) | drop (m) | energy median (J) | exposed |",
          "|---|---|---|---|---|---|---|---|"]
    for z in r["zones"][:15]:
        L.append(f"| {z['rank_full']} | {z['id']} | {z['area_m2']:,.0f} | {z['loss_m3']:.3g} | {z['max_lowering_m']:.1f} | "
                 f"{z['drop_m']:.0f} | {z['energy_J']['median']:.3g} | {z['exposed_assets']} |")
    if "ranking" in r:
        L += ["", f"## Documented source: covered by difference {r['source_covered']}; "
                  f"dz within 1 km {r['source_dz_1km']}"]
        for rad, row in r["ranking"].items():
            L.append(f"\n### lowered zones within {rad} m: {row['zones_matched']} "
                     f"(random expectation for best rank: {row['random_expected_best_rank']})")
            if not row["zones_matched"]:
                continue
            L.append("| ranking | best rank | percentile | top 5 | top 10 | top 25 |\n|---|---|---|---|---|---|")
            for name, v in row["variants"].items():
                L.append(f"| {name} | {v['best_rank']} | {v['percentile']:.1%} | {v['top_5']} | {v['top_10']} | {v['top_25']} |")
    s = r.get("source_zone")
    if s:
        L += ["", "## Source zone vs published values",
              f"zone {s['id']} (full rank {s['rank_full']}): loss volume {s['loss_m3']:.3g} m3 "
              f"(95% CI {s['ci95_m3'][0]:.3g}-{s['ci95_m3'][1]:.3g}); published {s['published_volume_m3']:.3g} m3; "
              f"ratio {s['volume_ratio_to_published']:.2f}; published inside CI: {s['published_in_ci95']}",
              f"area {s['area_m2']:,.0f} m2; max lowering {s['max_lowering_m']:.1f} m; source elevation "
              f"{s['source_elevation_m']:.0f} m (published ~{s['published_source_elevation_m']} m); drop {s['drop_m']:.0f} m",
              f"mass {s['mass_kg']['median']:.3g} kg (90% {s['mass_kg']['low']:.3g}-{s['mass_kg']['high']:.3g}); "
              f"energy {s['energy_J']['median']:.3g} J (90% {s['energy_J']['low']:.3g}-{s['energy_J']['high']:.3g})",
              "", "| checkpoint | closest (m) | reached | along route (km) | published (km) |", "|---|---|---|---|---|"]
        for c in s["pathway"]["checkpoints"]:
            L.append(f"| {c['name']} | {c['closest_m']} | {c['reached']} | {c['along_route_km']} | {c['published_km']} |")
    L += ["", r["note"]]
    return "\n".join(L) + "\n"


def dem_change_figure(cfg: dict, r: dict, out_dir: str | Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .report import _hillshade

    out = Path(out_dir)
    t1 = read_raster(cfg["inputs"]["t1"])
    dem = read_raster(cfg["inputs"]["dem"]).data[0]
    dz = read_raster(out / "dz_10m.tif").data[0]
    lab = read_raster(out / "dem_loss_zones.tif").data[0]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(_hillshade(dem, t1.res), cmap="gray")
    im = ax.imshow(np.ma.masked_invalid(np.clip(dz, -40, 40)), cmap="RdBu", vmin=-40, vmax=40, alpha=0.7)
    ax.contour(lab > 0, levels=[0.5], colors="k", linewidths=0.6)
    w = t1.shape[1]
    if r.get("_route_path"):
        p = np.array(r["_route_path"])
        ax.plot(p % w, p // w, color="gold", lw=1.5, label="route from documented source")
        e = cfg["evaluation"]["event"]
        rr, cc = _rc(t1, e["source"]["lon"], e["source"]["lat"])
        ax.plot(cc, rr, "*", color="yellow", ms=18, mec="k", label="documented source")
    fig.colorbar(im, ax=ax, shrink=0.7, label="elevation change (m), clipped to +/-40")
    ax.set_title("DEM difference (Feb 2021 - Sept 2015); lowered zones outlined")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "dem_change.png", dpi=130)
    plt.close(fig)
