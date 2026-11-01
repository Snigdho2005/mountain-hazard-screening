"""Instability evidence record (paper Section 7, Table 3).

Indicators are kept as measured values with units and uncertainty. A 0-1 score is
added only when the reference set supports it; missing values stay missing. No
combined index is produced in the minimum pipeline.
"""
from __future__ import annotations

import numpy as np

BARRIER_CLASSES = {"no_observed_concern": 0.0, "uncertain": 0.33, "deteriorating": 0.67, "urgent_concern": 1.0}
MIN_REFERENCE_N = 30


def percentile_score(x: float, reference: np.ndarray) -> dict:
    """s = (x - P10) / (P90 - P10) clipped to [0, 1], only if >= 30 independent reference values."""
    ref = np.asarray(reference, float)
    ref = ref[np.isfinite(ref)]
    if x is None or not np.isfinite(x):
        return {"value": None, "score": None, "method": "missing"}
    if ref.size < MIN_REFERENCE_N:
        z = float((x - ref.mean()) / ref.std()) if ref.size >= 3 and ref.std() > 0 else None
        return {"value": float(x), "score": None, "standardised_anomaly": z,
                "method": f"insufficient_reference_n={ref.size}"}
    p10, p90 = np.percentile(ref, [10, 90])
    s = float(np.clip((x - p10) / (p90 - p10), 0, 1)) if p90 > p10 else None
    return {"value": float(x), "score": s, "p10": float(p10), "p90": float(p90), "method": "percentile"}


def precipitation_anomaly(daily_mm: np.ndarray, end_index: int, climatology: dict[int, np.ndarray],
                          windows=(1, 3, 7), min_completeness=0.8) -> dict:
    """Accumulated totals over 1/3/7 days vs. same-period totals from a 30-year reference.

    `climatology[w]` holds reference totals for window w (one per reference year).
    """
    out = {}
    for w in windows:
        seg = np.asarray(daily_mm[max(0, end_index - w + 1): end_index + 1], float)
        comp = np.isfinite(seg).mean() if seg.size else 0.0
        if comp < min_completeness:
            out[f"{w}d"] = {"value": None, "method": f"completeness={comp:.2f}"}
            continue
        total = float(np.nansum(seg))
        out[f"{w}d"] = {"total_mm": total, **percentile_score(total, climatology.get(w, np.array([])))}
    return out


def temperature_trend(days: np.ndarray, temps_c: np.ndarray) -> dict:
    """OLS slope (degC per decade-equivalent per year scale given in days) and freeze-thaw crossings."""
    d, t = np.asarray(days, float), np.asarray(temps_c, float)
    m = np.isfinite(d) & np.isfinite(t)
    if m.sum() < 3:
        return {"slope_c_per_year": None, "freeze_thaw_crossings": None}
    slope = np.polyfit(d[m], t[m], 1)[0] * 365.25
    s = np.sign(t[m])
    crossings = int(np.sum((s[:-1] < 0) & (s[1:] > 0)))
    return {"slope_c_per_year": float(slope), "freeze_thaw_crossings": crossings, "n": int(m.sum())}


def displacement_record(times_days, displacement_m, sigma_m) -> dict:
    """Cumulative movement, rate and acceleration, with significance against measurement error."""
    t, d = np.asarray(times_days, float), np.asarray(displacement_m, float)
    if len(t) < 2:
        return {"cumulative_m": None}
    cum = float(d[-1] - d[0])
    rate = np.polyfit(t, d, 1)[0] * 365.25
    acc = float(np.polyfit(t, d, 2)[0] * 2 * 365.25 ** 2) if len(t) >= 4 else None
    return {"cumulative_m": cum, "rate_m_per_year": float(rate), "acceleration_m_per_year2": acc,
            "exceeds_error": bool(abs(cum) > 2 * np.sqrt(2) * sigma_m), "sigma_m": sigma_m}


def barrier_condition(cls: str | None) -> dict:
    if cls is None:
        return {"class": None, "display_value": None}
    if cls not in BARRIER_CLASSES:
        raise ValueError(f"barrier class must be one of {list(BARRIER_CLASSES)}")
    return {"class": cls, "display_value": BARRIER_CLASSES[cls]}


def zone_record(persistent: str, area_change_fraction: float, median_slope: float,
                extra: dict | None = None) -> dict:
    """The per-zone instability record produced from imagery alone; other indicators are
    attached when the user supplies them (radar/GNSS displacement, weather, field notes)."""
    rec = {
        "persistent_area_change": {"status": persistent, "changed_fraction": float(area_change_fraction)},
        "slope_deg_median": float(median_slope),
        "displacement": None,
        "precipitation_anomaly": None,
        "temperature_trend": None,
        "barrier_condition": None,
    }
    if extra:
        rec.update(extra)
    rec["independent_signals"] = sum(
        1 for k in ("displacement", "precipitation_anomaly", "temperature_trend", "barrier_condition")
        if rec.get(k)
    ) + (1 if persistent == "persistent" else 0)
    return rec
