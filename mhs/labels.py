"""Labelling helpers: QGIS-friendly previews and polygon -> label raster.

Workflow
  1. mhs preview CONFIG --out previews/
       writes t1_rgb.tif, t2_rgb.tif (true colour), t1_swir.tif, t2_swir.tif
       (B12-B08-B04 false colour: snow cyan, rock/debris brown-pink, water black)
       and split.geojson (train / validation / gap / test strips).
  2. In QGIS draw polygons in ONE layer with two attribute fields:
       class : background | water | snow_ice | rock | soil | debris  (or 0-5)
       date  : t1 | t2 | both
     Save as GeoPackage or GeoJSON.
  3. mhs rasterize-labels CONFIG --polygons labels.gpkg --out data/chamoli/labels.tif
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping

from . import CLASS_INDEX, CLASSES, IGNORE_LABEL
from .io import Raster, read_raster, write_raster


def _stretch(bands: np.ndarray, lo_pct=1, hi_pct=99, gamma=0.7) -> np.ndarray:
    out = np.zeros(bands.shape, np.uint8)
    for k, b in enumerate(bands):
        lo, hi = np.nanpercentile(b, lo_pct), np.nanpercentile(b, hi_pct)
        x = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1) ** gamma
        out[k] = np.where(np.isfinite(b), 1 + x * 254, 0).astype(np.uint8)  # 0 = no data
    return out


def write_previews(cfg: dict, out_dir: str | Path) -> list[str]:
    from .model.train import SPLIT_GAP, SPLIT_TEST, SPLIT_VAL, geographic_split

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for key in ("t1", "t2"):
        img = read_raster(cfg["inputs"][key])
        for name, bands in (("rgb", ("B04", "B03", "B02")), ("swir", ("B12", "B08", "B04"))):
            arr = _stretch(np.stack([img.band(b) for b in bands]))
            p = write_raster(out / f"{key}_{name}.tif", img.like(arr, list(bands), 0), "uint8")
            written.append(str(p))
    # split strips as polygons so the user can see where test labels are needed
    split = geographic_split(img.shape, **cfg["model"]["split"])[0]
    names = {0: "train", SPLIT_VAL: "validation", SPLIT_GAP: "gap (unused)", SPLIT_TEST: "test"}
    feats, start = [], 0
    for c in range(1, len(split) + 1):
        if c == len(split) or split[c] != split[start]:
            x0, _ = img.xy(0, start)
            x1, _ = img.xy(0, c - 1)
            _, y0 = img.xy(0, 0)
            _, y1 = img.xy(img.shape[0] - 1, 0)
            half = img.res / 2
            g = box(x0 - half, y1 - half, x1 + half, y0 + half)
            geom = transform_geom(img.crs, "EPSG:4326", mapping(g))
            feats.append({"type": "Feature", "geometry": geom,
                          "properties": {"split": names[int(split[start])]}})
            start = c
    (out / "split.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    written.append(str(out / "split.geojson"))
    return written


def _class_value(v) -> int:
    if v is None:
        raise ValueError("polygon without a class")
    if isinstance(v, (int, float)) or str(v).strip().isdigit():
        k = int(v)
        if not 0 <= k < len(CLASSES):
            raise ValueError(f"class number {k} outside 0-{len(CLASSES) - 1}")
        return k
    s = str(v).strip().lower().replace(" ", "_").replace("/", "_")
    aliases = {"snow": "snow_ice", "ice": "snow_ice", "snow_and_ice": "snow_ice", "bg": "background",
               "vegetation": "background", "other": "background"}
    s = aliases.get(s, s)
    if s not in CLASS_INDEX:
        raise ValueError(f"unknown class {v!r}; use one of {CLASSES}")
    return CLASS_INDEX[s]


def read_polygons(path: str | Path) -> tuple[list[dict], str]:
    """Read features from GeoJSON or GeoPackage. Returns (features, crs string)."""
    path = Path(path)
    if path.suffix.lower() in (".geojson", ".json"):
        gj = json.loads(path.read_text())
        crs = (gj.get("crs") or {}).get("properties", {}).get("name", "EPSG:4326")
        return gj["features"], crs
    import fiona  # GeoPackage / Shapefile

    with fiona.open(path) as src:
        crs = src.crs_wkt or "EPSG:4326"
        feats = [{"geometry": dict(f["geometry"]), "properties": dict(f["properties"])} for f in src]
    return feats, crs


def rasterize_labels(cfg: dict, polygons: str | Path, out_path: str | Path) -> dict:
    grid = read_raster(cfg["inputs"]["t1"])
    feats, crs = read_polygons(polygons)
    layers = {"t1": np.full(grid.shape, IGNORE_LABEL, np.uint8), "t2": np.full(grid.shape, IGNORE_LABEL, np.uint8)}
    counts = {d: {c: 0 for c in CLASSES} for d in layers}
    shapes = {"t1": [], "t2": []}
    for f in feats:
        p = {str(k).lower(): v for k, v in (f.get("properties") or {}).items()}
        k = _class_value(p.get("class"))
        when = str(p.get("date") or "both").strip().lower()
        if when not in ("t1", "t2", "both"):
            raise ValueError(f"date must be t1, t2 or both, got {when!r}")
        geom = transform_geom(crs, grid.crs, f["geometry"]) if str(crs) != str(grid.crs) else f["geometry"]
        for d in (("t1", "t2") if when == "both" else (when,)):
            shapes[d].append((geom, k))
    for d in layers:
        # later polygons overwrite earlier ones where they overlap
        for geom, k in shapes[d]:
            m = rasterize([(geom, 1)], out_shape=grid.shape, transform=grid.transform, dtype="uint8").astype(bool)
            layers[d][m] = k
        img = read_raster(cfg["inputs"][d])
        layers[d][~np.all(np.isfinite(img.data), axis=0)] = IGNORE_LABEL  # never label masked pixels
        v, n = np.unique(layers[d][layers[d] != IGNORE_LABEL], return_counts=True)
        for a, b in zip(v, n):
            counts[d][CLASSES[a]] = int(b)
    write_raster(out_path, grid.like(np.stack([layers["t1"], layers["t2"]]), ["labels_t1", "labels_t2"],
                                     IGNORE_LABEL), "uint8")
    return {"out": str(out_path), "polygons": len(feats), "pixels_per_class": counts}


def label_coverage_by_split(cfg: dict, labels_path: str | Path) -> dict:
    """How many labelled pixels fall in each split strip (test strip needs labels too)."""
    from .model.train import SPLIT_GAP, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, geographic_split

    lab = read_raster(labels_path).data
    split = geographic_split(lab.shape[1:], **cfg["model"]["split"])
    out = {}
    for name, s in (("train", SPLIT_TRAIN), ("validation", SPLIT_VAL), ("test", SPLIT_TEST), ("gap", SPLIT_GAP)):
        m = split == s
        out[name] = {CLASSES[c]: int(((lab[:, m] == c)).sum()) for c in range(len(CLASSES))}
    return out


def label_aid(cfg: dict, out_dir: str | Path, polygons: str | Path | None = None, n_points: int = 30,
              seed: int = 11, min_sep_m: float = 150.0, steep_only: bool = False, prefix: str = "P",
              avoid_points: str | Path | None = None, clean_px: int = 0) -> dict:
    """Aids for unbiased rock/soil labelling in the TEST strip.

    Writes
      slope_class.tif   1 = gentle (<25 deg), 2 = moderate, 3 = steep (>35 deg)
      t1_shadow.tif     1 where t1 is too dark to judge the surface (label those with date t2 only)
      test_points.geojson  stratified-random points on bare ground in the test strip, away from
                           existing polygons. Points are drawn at random (not by any class rule), so the
                           labeller decides rock / soil / skip by eye and the test set is not tuned to
                           either method.

    Rock stratum (steep_only=True, e.g. prefix "R", clean_px=1): random points are drawn only on
    steep (>35 deg) bare ground that is not shadowed on t2 and whose (2*clean_px+1)^2 neighbourhood is
    entirely bare, unmasked and unshadowed. Plain random points rarely land on usable rock, because
    clean rock in winter is mostly on steep faces next to snow and shadow. The labeller still decides
    rock / soil / skip by eye. Report this stratum separately: it measures rock vs soil on steep slopes.
    Written to <prefix>_points.geojson so the first set is not overwritten.
    """
    from .model.train import SPLIT_TEST, geographic_split
    from .preprocess import slope_aspect, spectral_indices

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t1, t2 = read_raster(cfg["inputs"]["t1"]), read_raster(cfg["inputs"]["t2"])
    dem = read_raster(cfg["inputs"]["dem"]).data[0]
    slope, _ = slope_aspect(dem, t2.res)
    sc = np.where(slope < 25, 1, np.where(slope > 35, 3, 2)).astype(np.uint8)
    sc[~np.isfinite(slope)] = 0
    write_raster(out / "slope_class.tif", t2.like(sc[None], ["slope_class"], 0), "uint8")

    b1 = np.mean(np.stack([t1.band(b) for b in ("B02", "B03", "B04", "B08")]), axis=0)
    dark1 = (~np.isfinite(b1)) | (b1 < 0.06)
    write_raster(out / "t1_shadow.tif", t2.like(dark1[None].astype(np.uint8), ["t1_dark"], 0), "uint8")

    idx = spectral_indices(t2)
    valid2 = np.all(np.isfinite(t2.data), axis=0)
    bare = valid2 & (idx["NDVI"] < 0.3) & (idx["NDSI"] < 0.2) & (idx["NDWI"] < 0.0)
    test = geographic_split(t2.shape, **cfg["model"]["split"]) == SPLIT_TEST
    # keep a 3-pixel margin from the strip edge
    from scipy import ndimage
    test = ndimage.binary_erosion(test, iterations=3)
    cand = bare & test
    if steep_only:
        b2 = np.mean(np.stack([t2.band(b) for b in ("B02", "B03", "B04", "B08")]), axis=0)
        cand &= (sc == 3) & np.isfinite(b2) & (b2 >= 0.06)
    if clean_px > 0:
        size = 2 * clean_px + 1
        cand &= ndimage.minimum_filter(cand.astype(np.uint8), size=size, mode="constant", cval=0).astype(bool)
    if avoid_points:
        pf, pcrs = read_polygons(avoid_points)
        m = np.zeros(t2.shape, bool)
        for f in pf:
            g = transform_geom(pcrs, t2.crs, f["geometry"]) if str(pcrs) != str(t2.crs) else f["geometry"]
            from rasterio.transform import rowcol
            r, c = rowcol(t2.transform, *g["coordinates"][:2])
            if 0 <= r < t2.shape[0] and 0 <= c < t2.shape[1]:
                m[r, c] = True
        if m.any():
            cand &= ndimage.distance_transform_edt(~m) * t2.res >= min_sep_m
    if polygons:
        feats, crs = read_polygons(polygons)
        geoms = [transform_geom(crs, t2.crs, f["geometry"]) if str(crs) != str(t2.crs) else f["geometry"]
                 for f in feats]
        if geoms:
            m = rasterize([(g, 1) for g in geoms], out_shape=t2.shape, transform=t2.transform,
                          dtype="uint8").astype(bool)
            near = ndimage.distance_transform_edt(~m) * t2.res < min_sep_m
            cand &= ~near
    rr, cc = np.nonzero(cand)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rr))
    pts, sep_px = [], min_sep_m / t2.res
    for k in order:
        r, c = rr[k], cc[k]
        if all((r - a) ** 2 + (c - b) ** 2 >= sep_px ** 2 for a, b in pts):
            pts.append((r, c))
        if len(pts) >= n_points:
            break
    pts.sort(key=lambda p: (p[0], p[1]))  # number from north to south
    feats_out = []
    for i, (r, c) in enumerate(pts, start=1):
        x, y = t2.xy(r, c)
        g = transform_geom(t2.crs, "EPSG:4326", {"type": "Point", "coordinates": [float(x), float(y)]})
        feats_out.append({"type": "Feature", "geometry": g, "properties": {
            "id": f"{prefix}{i}", "slope_deg": round(float(slope[r, c]), 1),
            "slope_class": ["", "gentle", "moderate", "steep"][int(sc[r, c])],
            "t1_dark": bool(dark1[r, c]), "suggested_date": "t2" if dark1[r, c] else "both"}})
    name = "test_points.geojson" if prefix == "P" else f"{prefix.lower()}_points.geojson"
    (out / name).write_text(json.dumps({"type": "FeatureCollection", "features": feats_out}))
    return {"points": len(feats_out), "file": name, "candidate_pixels": int(cand.sum()),
            "steep_only": steep_only, "clean_px": clean_px,
            "t1_dark_fraction_in_test": float(dark1[test].mean()), "out": str(out)}
