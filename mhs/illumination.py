"""Sun geometry, terrain illumination and cast shadows.

Different sun positions on the two dates make slopes switch between lit and
shadowed. A rule or learned classifier then reports a class change that is
only an illumination change (pseudo-change). This module finds pixels whose
shadow state differs between dates so the change step can drop them, and
records pixels shadowed on both dates so zones there can be flagged.
"""
from __future__ import annotations

import datetime as dt
import json
import re

import numpy as np

from .routing import _shifted

S2_OVERPASS_UTC = dt.time(5, 20)  # approximate Sentinel-2 overpass at ~79 E


def sun_position(when: dt.datetime, lat: float, lon: float) -> tuple[float, float]:
    """Solar elevation and azimuth (degrees, azimuth clockwise from north). NOAA approximation."""
    doy = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60 + when.second / 3600
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hour * 60 + eqtime + 4 * lon
    ha = np.radians(tst / 4 - 180)
    la = np.radians(lat)
    cos_z = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    zen = np.arccos(np.clip(cos_z, -1, 1))
    az = np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * np.sin(la) - np.tan(decl) * np.cos(la))) + 180
    return float(90 - np.degrees(zen)), float(az % 360)


def acquisition_sun(raster, lat: float, lon: float) -> dict:
    """Sun angles for a stack: from saved metadata if present, else computed from its timestamp."""
    tags = raster.tags or {}
    src = {}
    try:
        src = json.loads(tags.get("source", "{}"))
    except (TypeError, ValueError):
        pass
    if src.get("sun_elevation") is not None and src.get("sun_azimuth") is not None:
        return {"elevation": float(src["sun_elevation"]), "azimuth": float(src["sun_azimuth"]),
                "method": "scene_metadata"}
    when = None
    for iid in src.get("item_ids", []):
        m = re.search(r"_(\d{8}T\d{6})_", iid)
        if m:
            when = dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
            break
    method = "computed_from_item_time"
    if when is None:
        date = tags.get("date", "")
        if not date[:4].isdigit():
            return {"elevation": None, "azimuth": None, "method": "unknown_date"}
        when = dt.datetime.combine(dt.date.fromisoformat(date[:10]), S2_OVERPASS_UTC)
        method = "computed_from_date_assumed_overpass"
    el, az = sun_position(when, lat, lon)
    return {"elevation": el, "azimuth": az, "method": method, "time_utc": when.isoformat()}


def cos_incidence(slope_deg, aspect_deg, sun_el, sun_az) -> np.ndarray:
    z = np.radians(90 - sun_el)
    s = np.radians(slope_deg)
    return (np.cos(z) * np.cos(s) + np.sin(z) * np.sin(s) * np.cos(np.radians(sun_az - aspect_deg))).astype(np.float32)


def cast_shadow(dem: np.ndarray, res: float, sun_el: float, sun_az: float, max_dist_m: float = 8000.0,
                n_steps: int = 250) -> np.ndarray:
    """Pixels hidden from the sun by terrain between them and the sun (horizon tracing)."""
    z = np.where(np.isfinite(dem), dem, np.nanmin(dem)).astype(np.float32)
    dr, dc = -np.cos(np.radians(sun_az)), np.sin(np.radians(sun_az))  # toward the sun; north = -row
    tan_el = np.tan(np.radians(sun_el))
    max_px = max_dist_m / res
    shadow = np.zeros(z.shape, bool)
    seen = set()
    for d in np.geomspace(1, max_px, n_steps):
        oy, ox = int(round(d * dr)), int(round(d * dc))
        if (oy, ox) in seen or (oy == 0 and ox == 0):
            continue
        seen.add((oy, ox))
        dist = np.hypot(oy, ox) * res
        zs = _shifted(z, oy, ox, -np.inf)
        shadow |= (zs - z) > dist * tan_el
    return shadow


def shadow_state(dem, res, slope, aspect, sun: dict, min_cos: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """(shadow mask, cos incidence). Shadow = self-shadowed / grazing light or cast shadow."""
    if sun.get("elevation") is None:
        return np.zeros(dem.shape, bool), np.ones(dem.shape, np.float32)
    ci = cos_incidence(slope, aspect, sun["elevation"], sun["azimuth"])
    return (ci < min_cos) | cast_shadow(dem, res, sun["elevation"], sun["azimuth"]), ci
