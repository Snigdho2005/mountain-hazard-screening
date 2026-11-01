"""Rule-based baseline: spectral-index classes and direct image differencing.

This is the simple comparison method required by the paper (Sections 4.1, 8).
It uses the same six classes as the U-Net so the two can be scored identically.
Rock / soil / debris are not well separated spectrally; the rules use slope and
texture as declared heuristics, which is exactly why the baseline is weak there.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import CLASS_INDEX, IGNORE_LABEL
from .io import Raster
from .preprocess import spectral_indices

DEFAULT_RULES = {
    "water_ndwi": 0.15,
    "water_max_nir": 0.15,
    "water_max_slope": 12.0,   # lakes and rivers are flat; dark shadowed snow is not
    "snow_ndsi": 0.40,
    "snow_min_green": 0.25,
    "veg_ndvi": 0.30,
    "rock_min_slope": 35.0,
    "soil_max_slope": 20.0,
    "debris_texture": 0.025,
}


def rule_classes(img: Raster, slope: np.ndarray, rules: dict | None = None) -> np.ndarray:
    r = {**DEFAULT_RULES, **(rules or {})}
    idx = spectral_indices(img)
    g, n = img.band("B03"), img.band("B08")
    out = np.full(img.shape, CLASS_INDEX["background"], np.uint8)

    bare = (idx["NDVI"] < r["veg_ndvi"])
    texture = ndimage.generic_filter(np.nan_to_num(img.band("B11")), np.std, size=3)
    rock = bare & (slope >= r["rock_min_slope"])
    soil = bare & (slope < r["soil_max_slope"]) & (texture < r["debris_texture"])
    debris = bare & ~rock & ~soil
    out[debris] = CLASS_INDEX["debris"]
    out[soil] = CLASS_INDEX["soil"]
    out[rock] = CLASS_INDEX["rock"]

    snow = (idx["NDSI"] > r["snow_ndsi"]) & (g > r["snow_min_green"])
    water = (idx["NDWI"] > r["water_ndwi"]) & (n < r["water_max_nir"]) & ~snow & (slope < r["water_max_slope"])
    out[snow] = CLASS_INDEX["snow_ice"]
    out[water] = CLASS_INDEX["water"]
    out[~np.all(np.isfinite(img.data), axis=0)] = IGNORE_LABEL
    return out


def weak_labels(img: Raster, slope: np.ndarray, sample_fraction: float = 0.2, seed: int = 0) -> np.ndarray:
    """High-margin pixels only, as a STARTING POINT for reference labels.

    Covers water, snow/ice, vegetated background and steep rock. Soil and debris
    are left unlabelled because spectral rules cannot separate them reliably;
    they must be digitised by hand (e.g. in QGIS) before training. Never use
    weak labels as the evaluation reference.
    """
    idx = spectral_indices(img)
    g, n = img.band("B03"), img.band("B08")
    lab = np.full(img.shape, IGNORE_LABEL, np.uint8)
    lab[(idx["NDVI"] > 0.5)] = CLASS_INDEX["background"]
    lab[(idx["NDVI"] < 0.15) & (slope > 45) & (idx["NDSI"] < 0.1)] = CLASS_INDEX["rock"]
    lab[(idx["NDSI"] > 0.6) & (g > 0.4)] = CLASS_INDEX["snow_ice"]
    lab[(idx["NDWI"] > 0.3) & (n < 0.08) & (g < 0.2) & (slope < 8)] = CLASS_INDEX["water"]
    lab[~np.all(np.isfinite(img.data), axis=0)] = IGNORE_LABEL
    rng = np.random.default_rng(seed)
    lab[rng.random(lab.shape) > sample_fraction] = IGNORE_LABEL
    return lab


def difference_change(img1: Raster, img2: Raster, k: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    """Change-vector magnitude over the reflectance bands with a robust threshold.

    Returns (changed mask, magnitude). Threshold = median + k * 1.4826 * MAD.
    """
    bands = [b for b in img1.names if b in img2.names]
    d = np.stack([img2.band(b) - img1.band(b) for b in bands])
    mag = np.sqrt(np.nansum(d ** 2, axis=0))
    valid = np.all(np.isfinite(d), axis=0)
    m = mag[valid]
    med = np.median(m)
    mad = np.median(np.abs(m - med)) * 1.4826
    changed = valid & (mag > med + k * mad)
    return changed, mag.astype(np.float32)
