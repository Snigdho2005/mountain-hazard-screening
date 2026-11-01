"""Write maps, tables, a readable report and an overview figure for a run."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from rasterio.features import shapes
from rasterio.warp import transform_geom

from . import CLASSES, IGNORE_LABEL
from .io import Raster, write_raster

CLASS_COLORS = ["#7a9e5a", "#2f6db5", "#bfe3ff", "#8c8c8c", "#b58b5a", "#5a3e2b"]
MAX_LABELS = 25


def _fmt(x, unit=""):
    if x is None:
        return "–"
    if isinstance(x, float) and (abs(x) >= 1e5 or (0 < abs(x) < 1e-2)):
        return f"{x:.2e}{unit}"
    return f"{x:,.1f}{unit}" if isinstance(x, float) else f"{x}{unit}"


def zones_geojson(lab: np.ndarray, grid: Raster, records: list[dict], to_wgs84=True) -> dict:
    by_id = {r["id"]: r for r in records}
    feats = []
    for geom, val in shapes(lab.astype(np.int32), mask=lab > 0, transform=grid.transform):
        r = by_id.get(int(val))
        if r is None:
            continue
        if to_wgs84:
            geom = transform_geom(grid.crs, "EPSG:4326", geom)
        avail = r["physics"]["available"] or {}
        props = {
            "zone_id": r["id"], "review_order": r["review_order"], "source_type": r["source_type"],
            "transition": f"{r['transition_from']}->{r['transition_to']}", "area_m2": r["area_m2"],
            "persistence": r["persistence"], "confidence": round(r["confidence"], 3),
            "suggested_state": r["review"]["suggested_state"],
            "energy_status": avail.get("status"),
            "energy_median_J": avail.get("energy_J", {}).get("median"),
            "energy_low_J": avail.get("energy_J", {}).get("low"),
            "energy_high_J": avail.get("energy_J", {}).get("high"),
            "exposed_assets": len(r["exposure"]["assets"]) if r["exposure"] else None,
        }
        feats.append({"type": "Feature", "geometry": geom, "properties": props})
    return {"type": "FeatureCollection", "features": feats}


def routes_geojson(routes, grid: Raster, to_wgs84=True) -> dict:
    feats = []
    w = grid.shape[1]
    for zid, rt in routes.items():
        x, y = grid.xy(rt.path // w, rt.path % w)
        geom = {"type": "LineString", "coordinates": [[float(a), float(b)] for a, b in zip(x, y)]}
        if to_wgs84:
            geom = transform_geom(grid.crs, "EPSG:4326", geom)
        feats.append({"type": "Feature", "geometry": geom,
                      "properties": {"zone_id": zid, "length_m": rt.length_m, "qc_flags": ",".join(rt.qc_flags),
                                     "note": "screening route, not a runout boundary"}})
    return {"type": "FeatureCollection", "features": feats}


def write_csv(records, path):
    cols = ["review_order", "id", "suggested_state", "source_type", "transition_from", "transition_to",
            "area_m2", "persistence", "confidence", "slope_median_deg", "elev_mean_m", "drop_m",
            "route_length_m", "changed_loss_m3", "changed_sigma_m3", "energy_status", "energy_median_J",
            "energy_low_J", "energy_high_J", "p_exceed_threshold", "dominant_input", "exposed_assets",
            "alt_explanations", "linked_downstream_zones"]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in records:
            av = r["physics"]["available"] or {}
            chg = r["physics"]["changed"] or {}
            wr.writerow({
                "review_order": r["review_order"], "id": r["id"], "suggested_state": r["review"]["suggested_state"],
                "source_type": r["source_type"], "transition_from": r["transition_from"],
                "transition_to": r["transition_to"], "area_m2": round(r["area_m2"], 1),
                "persistence": r["persistence"], "confidence": round(r["confidence"], 3),
                "slope_median_deg": round(r["slope_median_deg"], 1), "elev_mean_m": round(r["elev_mean_m"], 1),
                "drop_m": round(r["route"]["drop_m"], 1), "route_length_m": round(r["route"]["length_m"], 1),
                "changed_loss_m3": chg.get("loss_m3"), "changed_sigma_m3": chg.get("sigma_m3"),
                "energy_status": av.get("status"),
                "energy_median_J": av.get("energy_J", {}).get("median"),
                "energy_low_J": av.get("energy_J", {}).get("low"),
                "energy_high_J": av.get("energy_J", {}).get("high"),
                "p_exceed_threshold": av.get("p_exceed_threshold"), "dominant_input": av.get("dominant_input"),
                "exposed_assets": len(r["exposure"]["assets"]) if r["exposure"] else None,
                "alt_explanations": ";".join(r["alt_explanations"]),
                "linked_downstream_zones": ";".join(map(str, r["linked_downstream_zones"])),
            })


def write_markdown(result, path):
    L = [f"# Screening report: {result['case']['name']}", "",
         f"_{result['disclaimer']}_", "",
         f"- Surface mapping: **{result['method']}**",
         f"- Dates: t1 {result['qc']['dates']['t1']}, t2 {result['qc']['dates']['t2']}, "
         f"t3 {result['qc']['dates']['t3']}  (season gap {result['qc']['season_gap_days']} days)",
         f"- t2 residual offset: {result['qc']['coreg_t2']['shift_px']:.2f} px "
         f"(corrected: {result['qc']['coreg_t2']['corrected']})",
         f"- Candidate zones: **{result['n_zones']}**; removed by slope rules: {len(result['removed_components'])}",
         f"- Baseline changed pixels: {result['baseline']['changed_pixels']}", "",
         "## Zones in review order", "",
         "| # | Zone | Suggested state | Type | Transition | Area (m²) | Persistence | Drop (m) | Energy (J), 90% interval | Exposed assets | Flags |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["zones"]:
        av = r["physics"]["available"] or {}
        if av.get("energy_J"):
            e = av["energy_J"]
            en = f"{e['median']:.2e} [{e['low']:.2e}–{e['high']:.2e}]" + (" (inconclusive)" if av["inconclusive"] else "")
        else:
            en = "not assigned (area-change record)"
        nexp = len(r["exposure"]["assets"]) if r["exposure"] else "–"
        L.append(f"| {r['review_order']} | {r['id']} | {r['review']['suggested_state']} | {r['source_type']} | "
                 f"{r['transition_from']}→{r['transition_to']} | {r['area_m2']:,.0f} | {r['persistence']} | "
                 f"{r['route']['drop_m']:.0f} | {en} | {nexp} | {', '.join(r['alt_explanations'] + r['route']['qc_flags']) or '–'} |")
    L += ["", "## Zone details", ""]
    for r in result["zones"]:
        L.append(f"### Zone {r['id']} — {r['source_type']}")
        L.append(f"- Review reasons: {'; '.join(r['review']['reasons'])}")
        L.append(f"- Persistence evidence: {r['persistence_evidence']}")
        chg = r["physics"]["changed"]
        if chg and chg.get("available"):
            L.append(f"- Changed volume (DEM difference): loss {chg['loss_m3']:,.0f} m³, gain {chg['gain_m3']:,.0f} m³, "
                     f"±{chg['sigma_m3']:,.0f} m³ (1σ), detectable: {chg['detectable']}")
        av = r["physics"]["available"] or {}
        if av.get("energy_J"):
            sh = ", ".join(f"{k} {v:.0%}" for k, v in av["spread_share"].items())
            L.append(f"- Available energy: median {av['energy_J']['median']:.2e} J; spread from {sh}; "
                     f"P(E > {av['threshold_J']:.0e} J) = {av['p_exceed_threshold']:.2f}")
        else:
            L.append(f"- Available energy: {av.get('reason', 'not assessed')}")
        if r["exposure"]:
            ex = r["exposure"]
            L.append(f"- Exposure in corridor: {ex['counts_by_type'] or 'none mapped'}; population {ex['population_in_corridor']}")
            for a in ex["assets"][:10]:
                L.append(f"  - {a['type']} '{a['name']}' at {a['along_route_m'] / 1000:.1f} km along route")
        if r["linked_downstream_zones"]:
            L.append(f"- Route passes through zones {r['linked_downstream_zones']} (possible linked source)")
        L.append("")
    Path(path).write_text("\n".join(L))


def overview_figure(layers, grid: Raster, records, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    def rgb(img):
        a = np.stack([img.band("B04"), img.band("B03"), img.band("B02")], -1)
        lo, hi = np.nanpercentile(a, 1), np.nanpercentile(a, 99)
        return np.clip(np.nan_to_num((a - lo) / (hi - lo)), 0, 1) ** 0.5

    cmap = ListedColormap(CLASS_COLORS)
    cmap.set_bad("#4a4a4a")  # masked / unusable pixels
    fig, ax = plt.subplots(2, 3, figsize=(16, 10.5))
    ax[0, 0].imshow(rgb(layers["t1"])); ax[0, 0].set_title(f"t1 RGB")
    ax[0, 1].imshow(rgb(layers["t2"])); ax[0, 1].set_title(f"t2 RGB")
    hs = _hillshade(layers["dem"], grid.res)
    for k, key in ((0, "cls1"), (1, "cls2")):
        c = np.ma.masked_equal(layers[key], IGNORE_LABEL)
        a = ax[1, k]
        a.imshow(c, cmap=cmap, vmin=-0.5, vmax=len(CLASSES) - 0.5, interpolation="nearest")
        a.set_title(f"class map {key[-1]}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in CLASS_COLORS]
    ax[1, 1].legend(handles, CLASSES, loc="lower right", fontsize=7, framealpha=0.85)

    a = ax[0, 2]
    a.imshow(hs, cmap="gray")
    a.imshow(np.ma.masked_equal(layers["change"] > 0, False), cmap=ListedColormap(["#d62728"]), alpha=0.8)
    a.imshow(np.ma.masked_equal(layers["baseline_change"], False), cmap=ListedColormap(["#ff7f0e"]), alpha=0.35)
    a.set_title("change: model (red) vs baseline (orange)")

    a = ax[1, 2]
    a.imshow(hs, cmap="gray")
    a.imshow(np.ma.masked_equal(layers["corridors"], False), cmap=ListedColormap(["#ffbf00"]), alpha=0.35)
    w = grid.shape[1]
    top = {r["id"] for r in records[:MAX_LABELS]}
    for zid, rt in layers["routes"].items():
        if zid not in top:
            continue
        a.plot(rt.path % w, rt.path // w, color="#1f77b4", lw=1.2)
        a.plot(rt.path[rt.reference_index] % w, rt.path[rt.reference_index] // w, "v", color="#1f77b4", ms=6)
    a.contour(layers["zones"] > 0, levels=[0.5], colors="#d62728", linewidths=1.2)
    for r in records[:MAX_LABELS]:  # records are already in review order
        rr, cc = grid.rowcol(r["centroid_x"], r["centroid_y"])
        a.text(cc, rr, str(r["id"]), color="white", fontsize=9, weight="bold", ha="center", va="center",
               bbox=dict(boxstyle="round,pad=0.15", fc="#d62728", ec="none"))
    if layers.get("assets"):
        from shapely.geometry import shape

        for asset in layers["assets"]:
            g = shape(asset["geometry"])
            if g.geom_type == "Point":
                rr, cc = grid.rowcol(g.x, g.y)
                a.plot(cc, rr, "s", ms=5, color="black", mec="white")
            elif "Line" in g.geom_type:
                xs, ys = np.array(g.coords).T if g.geom_type == "LineString" else (None, None)
                if xs is not None:
                    rr, cc = grid.rowcol(xs, ys)
                    a.plot(cc, rr, "-", color="black", lw=1)
    a.set_title(f"zones (red; top {min(MAX_LABELS, len(records))} of {len(records)} numbered), routes, corridor")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Screening overview — not a runout, inundation or warning map", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _hillshade(dem, res, az=315, alt=45):
    z = np.nan_to_num(dem, nan=np.nanmean(dem))
    dy, dx = np.gradient(z, res)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    a, b = np.radians(az), np.radians(alt)
    return np.sin(b) * np.cos(slope) + np.cos(b) * np.sin(slope) * np.cos(a - aspect)


def write_outputs(result, layers, grid: Raster, out: Path, cfg, figures=True):
    out = Path(out)
    maps = out / "maps"
    maps.mkdir(exist_ok=True)
    for key in ("cls1", "cls2", "cls3"):
        if layers[key] is not None:
            write_raster(maps / f"{key}.tif", grid.like(layers[key][None], [key], IGNORE_LABEL), "uint8")
    write_raster(maps / "transition.tif", grid.like(layers["change"][None], ["transition_code"]), "int32")
    write_raster(maps / "zones.tif", grid.like(layers["zones"][None], ["zone_id"], 0), "int32")
    write_raster(maps / "corridors.tif", grid.like(layers["corridors"][None].astype(np.uint8), ["corridor"]), "uint8")
    for key in ("base1", "base2"):
        if layers.get(key) is not None:
            write_raster(maps / f"baseline_cls{key[-1]}.tif", grid.like(layers[key][None], [key], IGNORE_LABEL), "uint8")
    write_raster(maps / "baseline_change.tif", grid.like(layers["baseline_change"][None].astype(np.uint8),
                                                          ["baseline_change"]), "uint8")
    with open(out / "zones.geojson", "w") as f:
        json.dump(zones_geojson(layers["zones"], grid, result["zones"]), f)
    with open(out / "routes.geojson", "w") as f:
        json.dump(routes_geojson(layers["routes"], grid), f)
    write_csv(result["zones"], out / "zones.csv")
    with open(out / "report.json", "w") as f:
        json.dump(result, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    write_markdown(result, out / "report.md")
    if figures:
        overview_figure(layers, grid, result["zones"], out / "overview.png")
