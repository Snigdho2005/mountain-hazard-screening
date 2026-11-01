"""Follow-up experiment D: rainfall and temperature indicators for the instability record (paper §7).

Daily ERA5 reanalysis at the documented source location via the Open-Meteo archive API
(no key needed; cite Open-Meteo, Zippenfenig 2023, doi:10.5281/zenodo.7970649, and Copernicus C3S ERA5).

For each window w (days), the total precipitation and mean temperature over the w days ending the day
before the event are compared with the same calendar window in every reference year (default
1991-2020). With >= 30 reference values the percentile score of paper §7 is reported; otherwise only
the value and a standardised anomaly. Daily mean temperature is (max + min) / 2.

ERA5 cells are ~31 km wide and their mean elevation is far below a 5,500 m source, so absolute values
are not local conditions; anomalies relative to the cell's own climate are the meaningful quantity.
One location cannot separate zones within a scene: this is event-level context, not a ranking input.
"""
from __future__ import annotations

import datetime as dt
import json
import urllib.parse
import urllib.request

import numpy as np

from .instability import percentile_score, temperature_trend

URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_daily(lat, lon, start, end, timeout=120) -> dict:
    base = {"latitude": f"{lat:.5f}", "longitude": f"{lon:.5f}", "start_date": start, "end_date": end,
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min", "timezone": "GMT"}
    last = None
    for extra in ({"models": "era5"}, {}):  # ERA5 explicitly; fall back to the API default reanalysis
        try:
            q = urllib.parse.urlencode({**base, **extra})
            with urllib.request.urlopen(f"{URL}?{q}", timeout=timeout) as r:
                d = json.loads(r.read())
            if "daily" in d:
                d["_model"] = extra.get("models", "api default")
                return d
            last = d
        except Exception as e:
            last = e
    raise RuntimeError(f"Open-Meteo request failed: {last}")


def window_indicators(dates, precip, tmean, event_date: dt.date, windows, ref_years) -> dict:
    idx = {d: i for i, d in enumerate(dates)}
    end = event_date - dt.timedelta(days=1)
    out = {}
    for w in windows:
        def window(year):
            try:
                e = end.replace(year=year)
            except ValueError:                       # 29 Feb in a non-leap year
                e = end.replace(year=year, day=28)
            days = [e - dt.timedelta(days=k) for k in range(w)]
            ii = [idx.get(d) for d in days]
            if any(i is None for i in ii):
                return None, None
            p, t = precip[ii], tmean[ii]
            if np.isnan(p).mean() > 0.2 or np.isnan(t).mean() > 0.2:
                return None, None
            return float(np.nansum(p)), float(np.nanmean(t))
        p_ev, t_ev = window(end.year)
        ref = [window(y) for y in range(ref_years[0], ref_years[1] + 1)]
        rp = np.array([r[0] for r in ref if r[0] is not None])
        rt = np.array([r[1] for r in ref if r[1] is not None])
        out[f"{w}d"] = {
            "precip_total_mm": None if p_ev is None else round(p_ev, 1),
            "precip_reference_median_mm": round(float(np.median(rp)), 1) if rp.size else None,
            "precip": percentile_score(p_ev, rp) if p_ev is not None else {"value": None, "method": "missing"},
            "temp_mean_c": None if t_ev is None else round(t_ev, 2),
            "temp_reference_mean_c": round(float(rt.mean()), 2) if rt.size else None,
            "temp_anomaly_c": None if t_ev is None or not rt.size else round(t_ev - float(rt.mean()), 2),
            "temp": percentile_score(t_ev, rt) if t_ev is not None else {"value": None, "method": "missing"},
            "n_reference_years": int(rp.size),
        }
    return out


def climate_record(cfg: dict) -> dict:
    ev = cfg["evaluation"]["event"]
    if not ev or not ev.get("date"):
        raise ValueError("evaluation.event with source and date is required")
    c = cfg["climate"]
    event_date = dt.date.fromisoformat(str(ev["date"]))
    start = f"{c['reference_years'][0]}-01-01"
    d = fetch_daily(ev["source"]["lat"], ev["source"]["lon"], start, event_date.isoformat())
    dates = [dt.date.fromisoformat(s) for s in d["daily"]["time"]]
    f = lambda k: np.array([np.nan if v is None else v for v in d["daily"][k]], float)
    precip, tmax, tmin = f("precipitation_sum"), f("temperature_2m_max"), f("temperature_2m_min")
    tmean = (tmax + tmin) / 2
    rec = {"event": ev["name"], "location": {"lat": ev["source"]["lat"], "lon": ev["source"]["lon"]},
           "reanalysis": d["_model"], "cell_elevation_m": d.get("elevation"),
           "reference_years": c["reference_years"],
           "windows": window_indicators(dates, precip, tmean, event_date, c["windows_days"], c["reference_years"])}
    # long-term winter (Dec-Feb) warming at the cell, reference period through the event winter
    yrs, wt = [], []
    for y in range(c["reference_years"][0] + 1, event_date.year + 1):
        m = np.array([(dd.year == y and dd.month in (1, 2)) or (dd.year == y - 1 and dd.month == 12) for dd in dates])
        if m.sum() > 60 and np.isfinite(tmean[m]).mean() > 0.8 and not (y == event_date.year):
            yrs.append(y)
            wt.append(float(np.nanmean(tmean[m])))
    tr = temperature_trend(np.array(yrs, float) * 365.25, np.array(wt))
    rec["winter_mean_temperature_trend"] = {"c_per_decade": None if tr["slope_c_per_year"] is None
                                            else round(tr["slope_c_per_year"] * 10, 3), "n_winters": len(yrs)}
    rec["note"] = ("ERA5 grid-cell values; anomalies relative to the cell's own 1991-2020 climate. Event-level "
                   "context for the instability record; not a ranking input and not evidence of causation.")
    return rec


def summary_markdown(r: dict) -> str:
    L = [f"# Experiment D: rainfall and temperature before the event ({r['reanalysis']})",
         f"location {r['location']}, cell elevation {r['cell_elevation_m']} m; reference {r['reference_years']}", "",
         "| window | precip (mm) | ref. median | precip percentile score | mean T (C) | anomaly (C) | T percentile score | n ref |",
         "|---|---|---|---|---|---|---|---|"]
    r2 = lambda x: "–" if x is None else f"{x:.2f}"
    for w, v in r["windows"].items():
        L.append(f"| {w} | {v['precip_total_mm']} | {v['precip_reference_median_mm']} | {r2(v['precip'].get('score'))} | "
                 f"{v['temp_mean_c']} | {v['temp_anomaly_c']} | {r2(v['temp'].get('score'))} | {v['n_reference_years']} |")
    t = r["winter_mean_temperature_trend"]
    L += ["", f"winter (DJF) mean temperature trend: {t['c_per_decade']} C per decade over {t['n_winters']} winters",
          "", r["note"]]
    return "\n".join(L) + "\n"
