"""Exposure along a screening corridor: settlements and infrastructure from GeoJSON.

Each feature should carry `type` (e.g. settlement, road, bridge, hydropower,
utility) and optionally `name`, `population`, `data_date`, `source`.
Coordinates may be lon/lat (EPSG:4326, GeoJSON default) or the grid CRS if the
file declares it under "crs". Monetary loss is never estimated here.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape

from .io import Raster


def load_assets(path: str | Path, grid: Raster) -> list[dict]:
    with open(path) as f:
        gj = json.load(f)
    src_crs = (gj.get("crs") or {}).get("properties", {}).get("name", "EPSG:4326")
    out = []
    for i, ft in enumerate(gj["features"]):
        geom = ft["geometry"]
        if str(src_crs) != str(grid.crs):
            geom = transform_geom(src_crs, grid.crs, geom)
        props = ft.get("properties") or {}
        out.append({"id": props.get("id", i), "type": props.get("type", "unknown"),
                    "name": props.get("name", f"asset_{i}"), "population": props.get("population"),
                    "data_date": props.get("data_date"), "source": props.get("source"),
                    "geometry": geom})
    return out


def asset_masks(assets: list[dict], grid: Raster) -> list[np.ndarray]:
    masks = []
    for a in assets:
        g = shape(a["geometry"])
        if g.geom_type == "Point":
            g = g.buffer(grid.res * 0.75)
        m = rasterize([(mapping(g), 1)], out_shape=grid.shape, transform=grid.transform,
                      all_touched=True, dtype="uint8").astype(bool)
        masks.append(m)
    return masks


def exposure_in_corridor(assets, masks, corridor: np.ndarray, route_path: np.ndarray,
                         route_dist: np.ndarray, width: int) -> dict:
    """Assets intersecting the corridor, ordered by along-route distance of nearest path cell."""
    pr, pc = route_path // width, route_path % width
    hits = []
    for a, m in zip(assets, masks):
        inter = m & corridor
        if not inter.any():
            continue
        rr, cc = np.nonzero(inter)
        d2 = (pr[:, None] - rr[None, :]) ** 2 + (pc[:, None] - cc[None, :]) ** 2
        k = int(np.unravel_index(np.argmin(d2), d2.shape)[0])
        hits.append({"id": a["id"], "type": a["type"], "name": a["name"], "population": a["population"],
                     "along_route_m": float(route_dist[k]), "data_date": a["data_date"],
                     "source": a["source"]})
    hits.sort(key=lambda h: h["along_route_m"])
    counts = Counter(h["type"] for h in hits)
    pops = [h["population"] for h in hits if h["population"] is not None]
    dates = sorted({h["data_date"] for h in hits if h["data_date"]})
    return {
        "assets": hits,
        "counts_by_type": dict(counts),
        "population_in_corridor": int(sum(pops)) if pops else None,
        "population_completeness": f"{len(pops)}/{sum(1 for h in hits if h['type'] == 'settlement')} settlements",
        "data_dates": dates,
        "note": "Screening corridor only; not a runout or inundation boundary. No monetary loss estimated.",
    }
