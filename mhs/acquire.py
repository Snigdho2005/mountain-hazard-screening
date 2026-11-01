"""Download Sentinel-2 L2A and Copernicus DEM onto a common grid.

Uses the Microsoft Planetary Computer STAC API (free, no account needed for
these collections). Run this where the internet is reachable, e.g. Kaggle with
"Internet on" or Colab:

    pip install pystac-client planetary-computer
    mhs acquire configs/chamoli.yaml

Each epoch window in `acquire.dates` yields one stack: the least-cloudy
acquisition day in that window, mosaicked from all tiles of that day.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from .io import Raster, write_raster
from .preprocess import dn_to_reflectance, quality_mask

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def utm_crs_for(lon: float, lat: float) -> CRS:
    zone = int(math.floor((lon + 180) / 6) + 1)
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def target_grid(bbox, crs=None, res=10.0):
    """Snapped transform and shape covering a lon/lat bbox in the target CRS."""
    lon0, lat0, lon1, lat1 = bbox
    crs = CRS.from_user_input(crs) if crs else utm_crs_for((lon0 + lon1) / 2, (lat0 + lat1) / 2)
    xmin, ymin, xmax, ymax = transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21)
    xmin, ymin = math.floor(xmin / res) * res, math.floor(ymin / res) * res
    xmax, ymax = math.ceil(xmax / res) * res, math.ceil(ymax / res) * res
    width, height = int(round((xmax - xmin) / res)), int(round((ymax - ymin) / res))
    return crs, from_origin(xmin, ymax, res, res), (height, width)


def _read_to_grid(href, crs, transform, shape, resampling):
    with rasterio.open(href) as src:
        with WarpedVRT(src, crs=crs, transform=transform, width=shape[1], height=shape[0],
                       resampling=resampling, src_nodata=src.nodata if src.nodata is not None else 0,
                       nodata=0) as vrt:
            return vrt.read(1)


def _client():
    try:
        import planetary_computer
        import pystac_client
    except ImportError as e:  # pragma: no cover
        raise ImportError("acquisition needs: pip install pystac-client planetary-computer") from e
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def acquire_sentinel2(bbox, window: str, out_path, crs=None, res=10.0, max_cloud=20.0,
                      bands=("B02", "B03", "B04", "B08", "B11", "B12"),
                      collection="sentinel-2-l2a") -> dict:
    """Least-cloudy day in `window`, mosaicked, cloud-masked, as reflectance GeoTIFF."""
    crs, transform, shape = target_grid(bbox, crs, res)
    items = list(_client().search(collections=[collection], bbox=bbox, datetime=window,
                                  query={"eo:cloud_cover": {"lt": max_cloud}}).items())
    if not items:
        raise RuntimeError(f"no {collection} scenes in {window} under {max_cloud}% cloud")
    by_day = defaultdict(list)
    for it in items:
        by_day[it.datetime.date().isoformat()].append(it)
    day = min(by_day, key=lambda d: np.mean([i.properties.get("eo:cloud_cover", 100) for i in by_day[d]]))
    day_items = sorted(by_day[day], key=lambda i: i.properties.get("eo:cloud_cover", 100))

    stack = np.full((len(bands), *shape), np.nan, np.float32)
    scl_out = np.zeros(shape, np.uint8)
    mask_stats = []
    for it in day_items:
        filled = np.all(np.isfinite(stack), axis=0)
        if filled.all():
            break
        scl = _read_to_grid(it.assets["SCL"].href, crs, transform, shape, Resampling.nearest)
        pb = it.properties.get("s2:processing_baseline")
        refl = {b: dn_to_reflectance(_read_to_grid(it.assets[b].href, crs, transform, shape,
                                                   Resampling.bilinear), pb) for b in bands}
        valid, st = quality_mask(scl, refl["B03"], refl["B11"], res)
        mask_stats.append({"item": it.id, **st})
        ok = valid & ~filled
        for k, b in enumerate(bands):
            stack[k][ok] = refl[b][ok]
        scl_out[ok | (scl_out == 0)] = scl[ok | (scl_out == 0)]
    meta = {"date": day, "window": window, "item_ids": [i.id for i in day_items],
            "cloud_cover": [i.properties.get("eo:cloud_cover") for i in day_items],
            "processing_baseline": [i.properties.get("s2:processing_baseline") for i in day_items],
            "mask": mask_stats,
            "sun_elevation": (90 - float(day_items[0].properties["s2:mean_solar_zenith"])
                              if "s2:mean_solar_zenith" in day_items[0].properties else None),
            "sun_azimuth": (float(day_items[0].properties["s2:mean_solar_azimuth"])
                            if "s2:mean_solar_azimuth" in day_items[0].properties else None),
            "valid_fraction": float(np.mean(np.all(np.isfinite(stack), axis=0)))}
    scl_path = Path(out_path).with_name(Path(out_path).stem + "_scl.tif")
    write_raster(scl_path, Raster(scl_out, transform, crs, ["SCL"], 0), dtype="uint8")
    write_raster(out_path, Raster(stack, transform, crs, list(bands), np.nan,
                                  tags={"date": day, "source": json.dumps(meta)}), dtype="float32")
    return meta


AWS_COP30 = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
             "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")


def cop30_aws_urls(bbox) -> list[str]:
    """1-degree Copernicus GLO-30 tiles covering a bbox, from the public AWS open-data bucket."""
    w, s, e, n = bbox
    urls = []
    for la in range(math.floor(s), math.floor(n - 1e-9) + 1):
        for lo in range(math.floor(w), math.floor(e - 1e-9) + 1):
            urls.append(AWS_COP30.format(ns="N" if la >= 0 else "S", lat=abs(la),
                                         ew="E" if lo >= 0 else "W", lon=abs(lo)))
    return urls


def acquire_dem(bbox, out_path, crs=None, res=10.0, collection="cop-dem-glo-30") -> dict:
    """Copernicus GLO-30 from Planetary Computer; if that fails, the same product from AWS open data."""
    crs, transform, shape = target_grid(bbox, crs, res)
    source, note = "planetary_computer", None
    try:
        items = list(_client().search(collections=[collection], bbox=bbox).items())
        if not items:
            raise RuntimeError("no DEM tiles found")
        hrefs, ids = [it.assets["data"].href for it in items], [i.id for i in items]
    except Exception as e:  # e.g. Planetary Computer token endpoint unavailable
        note = f"Planetary Computer DEM failed ({type(e).__name__}: {str(e)[:120]}); used AWS open data"
        print(note)
        hrefs = cop30_aws_urls(bbox)
        ids, source = [Path(h).stem for h in hrefs], "aws_open_data_copernicus-dem-30m"
    dem = np.full(shape, np.nan, np.float32)
    for href in hrefs:
        with rasterio.open(href) as src:
            with WarpedVRT(src, crs=crs, transform=transform, width=shape[1], height=shape[0],
                           resampling=Resampling.bilinear, nodata=np.nan, dtype="float32") as vrt:
                z = vrt.read(1)
        m = np.isfinite(z) & ~np.isfinite(dem)
        dem[m] = z[m]
    meta = {"collection": collection, "source": source, "item_ids": ids, "note": note,
            "valid_fraction": float(np.isfinite(dem).mean())}
    if meta["valid_fraction"] < 0.99:
        raise RuntimeError(f"DEM covers only {meta['valid_fraction']:.0%} of the scene")
    write_raster(out_path, Raster(dem, transform, crs, ["elevation"], np.nan,
                                  tags={"source": json.dumps(meta)}), dtype="float32")
    return meta


def acquire_case(cfg: dict, out_dir: str | Path, only: list[str] | None = None) -> dict:
    """Download all epochs, or only those named in `only` (e.g. ["t3"]), keeping the rest."""
    case, acq = cfg["case"], cfg["acquire"]
    if not case["bbox"] or len(acq["dates"]) < 2:
        raise ValueError("config needs case.bbox and at least two acquire.dates windows")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / "acquisition.json"
    record = json.loads(rec_path.read_text()) if (only and rec_path.exists()) else {}
    if not only or "dem" in only:
        record["dem"] = acquire_dem(case["bbox"], out / "dem.tif", case["crs"], case["resolution"],
                                    acq["dem_collection"])
    for k, window in enumerate(acq["dates"], start=1):
        if only and f"t{k}" not in only:
            continue
        record[f"t{k}"] = acquire_sentinel2(case["bbox"], window, out / f"t{k}.tif", case["crs"],
                                            case["resolution"], acq["max_cloud"], acq["bands"],
                                            acq["collection"])
    with open(out / "acquisition.json", "w") as f:
        json.dump(record, f, indent=2)
    return record
