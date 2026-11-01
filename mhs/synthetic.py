"""Synthetic mountain catchment with known truth, for offline end-to-end testing.

Scene (3.84 x 3.84 km, 10 m): a main valley draining east, a northern
tributary fed by a cirque lake, a glaciated peak. Between t1 and t2:
  1. the cirque lake grows (water_storage)
  2. a landslide opens a scar on the main-valley side slope and leaves a deposit
  3. a glacier tongue thins and exposes rock
  4. the snow line drops slightly (seasonal; reverts by t3)
Settlements, a hydropower plant, a bridge and a road sit on the valley floor.
The spectra are idealised; this tests the software, not the science.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from rasterio.crs import CRS
from rasterio.transform import from_origin
from scipy import ndimage

from . import CLASS_INDEX, IGNORE_LABEL
from .io import Raster, write_raster

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
SPECTRA = {
    "background": [0.040, 0.070, 0.050, 0.300, 0.180, 0.090],
    "water":      [0.060, 0.050, 0.030, 0.020, 0.010, 0.005],
    "snow_ice":   [0.850, 0.820, 0.800, 0.700, 0.100, 0.070],
    "rock":       [0.140, 0.160, 0.180, 0.240, 0.300, 0.260],
    "soil":       [0.100, 0.130, 0.170, 0.220, 0.330, 0.290],
    "debris":     [0.120, 0.140, 0.150, 0.190, 0.240, 0.200],
}
X0, Y0, RES, N = 390000.0, 3385000.0, 10.0, 384
CRS_ = CRS.from_epsg(32644)


def _layout(seed: int) -> dict:
    """Seed-dependent positions so dev and test catchments differ geographically."""
    r = np.random.default_rng(seed + 1000)
    trib = float(r.uniform(1300, 1900))
    return {"trib_x": trib, "lake_c": (trib, float(r.uniform(850, 1050))),
            "peak_x": float(r.uniform(700, 1000)), "glacier_c": (float(r.uniform(650, trib - 550)), 1150.0),
            "scar_x": float(r.uniform(2300, 3200))}


def _terrain(rng, L):
    y, x = np.mgrid[0:N, 0:N] * RES
    yv = 2600.0                                   # main valley axis (m from top)
    floor = 2400.0 - 0.08 * x                     # drains east
    dy = np.abs(y - yv)
    z = floor + 0.5 * dy + 2.5e-5 * dy ** 2       # V-shaped valley steepening upwards
    # northern tributary along x = trib_x: a V notch whose depth grows downstream
    notch = np.clip(1 - np.abs(x - L['trib_x']) / 350, 0, None) * (y < yv)
    z -= notch * 0.45 * np.clip(yv - y, 0, 1700)
    # glaciated peak
    z += 1300 * np.exp(-(((x - L['peak_x']) / 650) ** 2 + ((y - 400) / 500) ** 2))
    z += ndimage.gaussian_filter(rng.normal(0, 1, (N, N)), 6) * 60
    # cirque lake basin at the head of the tributary: scoop, then flatten to lake level
    lake_c = L['lake_c']
    r = np.hypot(x - lake_c[0], y - lake_c[1])
    z -= 90 * np.exp(-(r / 300) ** 2)
    lake_level = float(np.percentile(z[(r > 240) & (r < 270)], 10)) - 3
    z = np.where(r < 260, np.minimum(z, lake_level), z)
    return x, y, z.astype(np.float32), yv, lake_c, lake_level


def _slope(z):
    gy, gx = np.gradient(z, RES)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def _classes(x, y, z, slope, yv, lake_c, lake_r, snow_line, rng_field, trib_x):
    c = np.full(z.shape, CLASS_INDEX["background"], np.uint8)
    c[(slope > 22) & (z > 2600) & (rng_field > 0.0)] = CLASS_INDEX["soil"]
    c[slope > 38] = CLASS_INDEX["rock"]
    c[z > 3700] = CLASS_INDEX["rock"]
    c[np.abs(y - yv) < 70] = CLASS_INDEX["debris"]
    c[(np.abs(x - trib_x) < 40) & (y < yv) & (y > lake_c[1] + 300)] = CLASS_INDEX["debris"]
    c[z > snow_line] = CLASS_INDEX["snow_ice"]
    c[np.abs(y - yv) < 15] = CLASS_INDEX["water"]
    c[np.hypot(x - lake_c[0], y - lake_c[1]) < lake_r] = CLASS_INDEX["water"]
    return c


def _render(cls, z, rng, noise=0.008):
    gy, gx = np.gradient(z, RES)
    sl = np.arctan(np.hypot(gx, gy))
    asp = np.arctan2(-gx, gy)
    illum = np.cos(sl) * np.sin(np.radians(40)) + np.sin(sl) * np.cos(np.radians(40)) * np.cos(np.radians(135) - asp)
    illum = np.clip(0.55 + 0.6 * illum, 0.3, 1.25)
    names = list(SPECTRA)
    spec = np.array([SPECTRA[n] for n in names], np.float32)  # (6 classes, 6 bands)
    img = spec[cls].transpose(2, 0, 1) * illum[None]
    # within-class variability (moisture, lithology, grain size) as a smooth field
    var = ndimage.gaussian_filter(rng.normal(0, 1, cls.shape), 4)
    img *= (1 + 0.07 * var / (var.std() + 1e-9))[None]
    deb = cls == CLASS_INDEX["debris"]
    img[:, deb] *= 1 + rng.normal(0, 0.12, deb.sum())[None]
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0.001, 1.2).astype(np.float32)


def generate(out_dir: str | Path, seed: int = 0) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    L = _layout(seed)
    x, y, z1, yv, lake_c, lake_level = _terrain(rng, L)
    soil_field = ndimage.gaussian_filter(rng.normal(0, 1, z1.shape), 8)
    soil_field = soil_field / soil_field.std()
    tr = from_origin(X0, Y0, RES, RES)

    # ---- t2 terrain change: landslide scar and deposit, glacier thinning
    scar_c, dep_c = (L['scar_x'], yv - 420.0), (L['scar_x'] + 50, yv - 160.0)
    scar = ((x - scar_c[0]) / 200) ** 2 + ((y - scar_c[1]) / 120) ** 2 < 1
    dep = ((x - dep_c[0]) / 220) ** 2 + ((y - dep_c[1]) / 90) ** 2 < 1
    gx, gy = L['glacier_c']
    glacier = ((x - gx) / 300) ** 2 + ((y - gy) / 150) ** 2 < 1
    dz = np.zeros_like(z1)
    dz -= 12 * np.clip(1 - (((x - scar_c[0]) / 200) ** 2 + ((y - scar_c[1]) / 120) ** 2), 0, 1) * scar
    dz += 6 * np.clip(1 - (((x - dep_c[0]) / 220) ** 2 + ((y - dep_c[1]) / 90) ** 2), 0, 1) * dep
    dz -= 8 * glacier
    z2 = z1 + dz

    slope1 = _slope(z1)
    snow1 = 3900.0
    c1 = _classes(x, y, z1, slope1, yv, lake_c, 150, snow1, soil_field, L['trib_x'])
    # make sure glacier tongue is ice at t1 (it extends below the snow line)
    c1[glacier] = CLASS_INDEX["snow_ice"]

    def later(lake_r, snow_line):
        c = _classes(x, y, z1, slope1, yv, lake_c, lake_r, snow_line, soil_field, L['trib_x'])
        c[glacier] = CLASS_INDEX["rock"]
        c[scar] = CLASS_INDEX["soil"]
        c[scar & (slope1 > 30)] = CLASS_INDEX["rock"]
        c[dep & ~scar & (c != CLASS_INDEX["water"])] = CLASS_INDEX["debris"]
        return c

    c2 = later(230, 3820.0)   # snow line drops at t2 (seasonal)
    c3 = later(232, 3900.0)   # snow line back at t3; other changes persist

    imgs = [_render(c, z1, np.random.default_rng(seed + k + 10)) for k, c in enumerate((c1, c2, c3), 1)]
    dates = ["2023-10-12", "2024-10-06", "2024-11-05"]
    paths = {}
    for k, (img, d) in enumerate(zip(imgs, dates), 1):
        paths[f"t{k}"] = write_raster(out / f"t{k}.tif", Raster(img, tr, CRS_, BANDS, tags={"date": d}), "float32")
    write_raster(out / "dem.tif", Raster(z1, tr, CRS_, ["elevation"]), "float32")
    write_raster(out / "dem_t1.tif", Raster(z1, tr, CRS_, ["elevation"]), "float32")
    noise = ndimage.gaussian_filter(np.random.default_rng(seed + 99).normal(0, 1, z1.shape), 3)
    write_raster(out / "dem_t2.tif", Raster(z2 + noise.astype(np.float32), tr, CRS_, ["elevation"]), "float32")

    # ---- sparse reference labels (15% of pixels) for t1 and t2; truth kept separately
    lrng = np.random.default_rng(seed + 5)
    labels = []
    for c in (c1, c2):
        lab = np.full(c.shape, IGNORE_LABEL, np.uint8)
        pick = lrng.random(c.shape) < 0.15
        lab[pick] = c[pick]
        labels.append(lab)
    write_raster(out / "labels.tif", Raster(np.stack(labels), tr, CRS_, ["labels_t1", "labels_t2"], IGNORE_LABEL), "uint8")
    write_raster(out / "truth.tif", Raster(np.stack([c1, c2, c3]), tr, CRS_, ["truth_t1", "truth_t2", "truth_t3"]), "uint8")

    # ---- assets on the valley floor (grid CRS)
    def pt(px, py):
        return [X0 + px, Y0 - py]

    feats = [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": pt(L['trib_x'] + 500, yv + 40)},
         "properties": {"type": "settlement", "name": "Village A", "population": 640, "data_date": "2024-06", "source": "synthetic"}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": pt(min(L['scar_x'] + 700, 3750), yv - 40)},
         "properties": {"type": "settlement", "name": "Village B", "population": 1200, "data_date": "2024-06", "source": "synthetic"}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": pt(L['scar_x'] + 350, yv + 30)},
         "properties": {"type": "hydropower", "name": "Small HEP", "data_date": "2024-06", "source": "synthetic"}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": pt(L['trib_x'] + 30, yv)},
         "properties": {"type": "bridge", "name": "Tributary bridge", "data_date": "2024-06", "source": "synthetic"}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": pt(600, yv - 30)},
         "properties": {"type": "settlement", "name": "Village C (upstream)", "population": 300, "data_date": "2024-06", "source": "synthetic"}},
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [pt(px, yv + 90) for px in range(0, 3841, 160)]},
         "properties": {"type": "road", "name": "Valley road", "data_date": "2024-06", "source": "synthetic"}},
    ]
    gj = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:32644"}}, "features": feats}
    with open(out / "assets.geojson", "w") as f:
        json.dump(gj, f, indent=1)

    cfg = {
        "case": {"name": "synthetic_catchment", "crs": "EPSG:32644", "resolution": RES,
                 "notes": "Synthetic scene with known truth (mhs.synthetic)."},
        "inputs": {"t1": "t1.tif", "t2": "t2.tif", "t3": "t3.tif", "dem": "dem.tif", "dem_t1": "dem_t1.tif",
                   "dem_t2": "dem_t2.tif", "labels": "labels.tif", "assets": "assets.geojson"},
        "model": {"epochs": 12, "samples_per_epoch": 480, "batch_size": 12, "patch_size": 96,
                  "base_channels": 16, "depth": 3, "patience": 5},
        "physics": {"thickness_scenarios_m": {"rock_soil_storage": [3, 6, 12], "water_storage": [8, 15, 30],
                                              "snow_ice_storage": [10, 25, 50]},
                    "dz_sigma_m": 1.0},
        "output": {"dir": str(out / "runs")},
    }
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    truth = {"lake_growth_m2": float((np.hypot(x - lake_c[0], y - lake_c[1]) < 230).sum() * RES ** 2 -
                                     (np.hypot(x - lake_c[0], y - lake_c[1]) < 150).sum() * RES ** 2),
             "scar_volume_change_m3": float(dz[scar].sum() * RES ** 2),
             "deposit_volume_gain_m3": float(dz[dep & ~scar].sum() * RES ** 2),
             "glacier_volume_loss_m3": float(dz[glacier].sum() * RES ** 2),
             "scar_area_m2": float(scar.sum() * RES ** 2), "glacier_area_m2": float(glacier.sum() * RES ** 2)}
    truth["layout"] = L
    with open(out / "truth.json", "w") as f:
        json.dump(truth, f, indent=2)
    return {"config": str(out / "config.yaml"), "truth": truth}
