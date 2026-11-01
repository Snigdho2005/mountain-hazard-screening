"""Input preparation: reflectance, quality masks, co-registration, indices, terrain."""
from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
from scipy import ndimage

from .io import Raster

# Sentinel-2 L2A scene classification (SCL) codes treated as invalid.
# 11 (snow) is deliberately kept valid: snow and ice are a mapped class.
SCL_INVALID = {0: "no_data", 1: "saturated", 3: "cloud_shadow", 8: "cloud_medium",
               9: "cloud_high", 10: "thin_cirrus"}


def dn_to_reflectance(dn: np.ndarray, processing_baseline: str | float | None) -> np.ndarray:
    """Convert L2A digital numbers to surface reflectance.

    From processing baseline 04.00 (25 Jan 2022) ESA adds BOA_ADD_OFFSET = -1000.
    """
    dn = dn.astype(np.float32)
    offset = 0.0
    try:
        if processing_baseline is not None and float(processing_baseline) >= 4.0:
            offset = -1000.0
    except ValueError:
        pass
    refl = (dn + offset) / 10000.0
    refl[dn == 0] = np.nan
    return np.clip(refl, 0.0, 1.5)


def scl_valid_mask(scl: np.ndarray) -> np.ndarray:
    return ~np.isin(scl, list(SCL_INVALID))


def quality_mask(scl, green, swir1, res=10.0, ndsi_snow=0.4, swir_snow_max=0.2,
                 shadow_buffer_m=3000.0, cloud_edge_px=2):
    """Snow-aware validity mask for mountain scenes.

    SCL often labels bright snow/ice as cloud and terrain shadow as cloud
    shadow. Here:
      * no data / saturated (SCL 0, 1) are always invalid;
      * SCL cloud classes (8, 9, 10) are invalid only where the pixel does not
        look like snow (NDSI > ndsi_snow and SWIR1 < swir_snow_max); real water
        clouds are bright in SWIR1, snow is dark;
      * SCL cloud shadow (3) is invalid only within shadow_buffer_m of a
        retained cloud; elsewhere it is terrain shadow and kept (the change
        step flags dark zones as possible shadow).
    Returns (valid mask, statistics).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ndsi = (green - swir1) / (green + swir1)
    nodata = np.isin(scl, [0, 1]) | ~np.isfinite(green) | ~np.isfinite(swir1)
    cloudlike = np.isin(scl, [8, 9, 10])
    snowlike = (ndsi > ndsi_snow) & (swir1 < swir_snow_max)
    cloud = cloudlike & ~snowlike
    if cloud_edge_px and cloud.any():
        cloud = ndimage.binary_opening(cloud, iterations=1)          # drop speckle
        cloud = ndimage.binary_dilation(cloud, iterations=cloud_edge_px)  # cover edges
    shadow = scl == 3
    if cloud.any() and shadow.any():
        shadow &= ndimage.distance_transform_edt(~cloud) * res <= shadow_buffer_m
    else:
        shadow = np.zeros_like(shadow)
    valid = ~(nodata | cloud | shadow)
    n = scl.size
    stats = {"nodata": float(nodata.mean()), "scl_cloud": float(cloudlike.mean()),
             "scl_cloud_kept_as_snow": float((cloudlike & snowlike).sum() / n),
             "cloud_masked": float(cloud.mean()), "scl_shadow": float((scl == 3).mean()),
             "shadow_masked": float(shadow.mean()), "valid": float(valid.mean())}
    return valid, stats


def _nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / (a + b)
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32)


def spectral_indices(img: Raster) -> dict[str, np.ndarray]:
    """Standard indices from a stack with bands B02 B03 B04 B08 B11 (B12 optional).

    NDSI and MNDWI share a formula (green vs SWIR1); water and snow are then
    separated by visible brightness in the rule baseline.
    """
    b, g, r, n, s1 = (img.band(k) for k in ("B02", "B03", "B04", "B08", "B11"))
    out = {
        "NDWI": _nd(g, n),          # McFeeters water index
        "NDSI": _nd(g, s1),         # snow index (= MNDWI)
        "NDVI": _nd(n, r),
        "BSI": _nd(s1 + r, n + b),  # bare soil index
        "BRIGHT": np.mean(np.stack([b, g, r]), axis=0).astype(np.float32),  # NaN where masked
    }
    return out


def slope_aspect(dem: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope (degrees) and aspect (degrees clockwise from north) by central differences."""
    z = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    dzdy, dzdx = np.gradient(z, res)
    # rows increase southwards, so north-facing gradient is -dzdy
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = (np.degrees(np.arctan2(-dzdx, dzdy)) + 360.0) % 360.0
    slope[~np.isfinite(dem)] = np.nan
    return slope.astype(np.float32), aspect.astype(np.float32)


def estimate_shift(ref: np.ndarray, mov: np.ndarray, upsample: int = 20) -> tuple[float, float, float]:
    """Sub-pixel (row, col) shift of `mov` relative to `ref` by phase correlation."""
    from skimage.registration import phase_cross_correlation

    a = np.nan_to_num(ref - np.nanmean(ref))
    b = np.nan_to_num(mov - np.nanmean(mov))
    shift, error, _ = phase_cross_correlation(a, b, upsample_factor=upsample)
    return float(shift[0]), float(shift[1]), float(error)


def apply_shift(img: Raster, drow: float, dcol: float) -> Raster:
    data = np.stack([
        ndimage.shift(np.nan_to_num(band, nan=0.0), (drow, dcol), order=1, mode="constant", cval=np.nan)
        for band in img.data
    ]).astype(np.float32)
    valid = ndimage.shift(np.isfinite(img.data[0]).astype(np.float32), (drow, dcol), order=0, cval=0) > 0.5
    data[:, ~valid] = np.nan
    return img.like(data, img.names, img.nodata)


def coregister(ref: Raster, mov: Raster, band: str = "B08", max_shift_px: float = 0.5,
               reject_shift_px: float = 2.0) -> tuple[Raster, dict]:
    """Check and correct residual misregistration of `mov` against `ref`."""
    drow, dcol, err = estimate_shift(ref.band(band), mov.band(band))
    mag = float(np.hypot(drow, dcol))
    record = {"shift_row_px": drow, "shift_col_px": dcol, "shift_px": mag, "error": err,
              "corrected": False, "rejected": False}
    if mag > reject_shift_px:
        record["rejected"] = True
        warnings.warn(f"Residual offset {mag:.2f} px exceeds {reject_shift_px} px; pair should be excluded")
        return mov, record
    if mag > max_shift_px:
        mov = apply_shift(mov, drow, dcol)
        record["corrected"] = True
    return mov, record


def season_gap_days(d1: str, d2: str) -> int:
    """Circular day-of-year distance between two acquisition dates."""
    a = dt.date.fromisoformat(d1[:10]).timetuple().tm_yday
    b = dt.date.fromisoformat(d2[:10]).timetuple().tm_yday
    d = abs(a - b)
    return int(min(d, 365 - d))


def build_features(img: Raster, slope: np.ndarray | None, names: list[str]) -> np.ndarray:
    """Assemble the model input stack (C, H, W) in the declared feature order."""
    idx = spectral_indices(img)
    layers = []
    for n in names:
        if img.has(n):
            layers.append(img.band(n))
        elif n in idx:
            layers.append(idx[n])
        elif n == "slope":
            if slope is None:
                raise ValueError("feature 'slope' requested but no DEM slope supplied")
            layers.append(slope / 90.0)
        else:
            raise KeyError(f"unknown feature {n!r}")
    return np.stack(layers).astype(np.float32)


def valid_mask(img: Raster) -> np.ndarray:
    return np.all(np.isfinite(img.data), axis=0)
