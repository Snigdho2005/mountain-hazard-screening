"""Source-zone formation (paper Table 1 'Source-zone formation', Section 5)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage

from . import CLASSES
from .change import decode_transition, source_type_for


@dataclass
class Zone:
    id: int
    source_type: str
    transition_from: str
    transition_to: str
    transition_share: float
    area_m2: float
    n_pixels: int
    centroid_x: float
    centroid_y: float
    elev_mean_m: float
    elev_max_m: float
    slope_median_deg: float
    confidence: float
    persistence: str                 # persistent | not_persistent | unconfirmed
    persistence_evidence: str
    in_channel: bool
    alt_explanations: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def form_zones(change: dict, slope: np.ndarray, elevation: np.ndarray, grid, zcfg: dict,
               accum_km2: np.ndarray | None = None) -> tuple[np.ndarray, list[dict]]:
    """Join adjoining changed pixels, remove isolated patches, apply area rule.

    Returns a label image and raw component records; slope rules and
    persistence are applied in `finalise_zones` once source type is known.
    """
    m = change["changed"]
    if zcfg["opening_px"]:
        m = ndimage.binary_opening(m, structure=np.ones((3, 3)), iterations=zcfg["opening_px"])
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    min_px = int(np.ceil(zcfg["min_area_m2"] / grid.pixel_area))
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_px
    lab = np.where(keep[lab], lab, 0)
    lab, n = ndimage.label(lab > 0, structure=np.ones((3, 3)))
    comps = []
    objs = ndimage.find_objects(lab)
    for zid, sl in enumerate(objs, start=1):
        sub = lab[sl] == zid
        codes = change["transition"][sl][sub]
        codes = codes[codes > 0]
        vals, cnt = np.unique(codes, return_counts=True)
        dom = int(vals[np.argmax(cnt)]) if len(vals) else 0
        f, t = decode_transition(dom) if dom else ("background", "background")
        rr, cc = np.nonzero(sub)
        rr, cc = rr + sl[0].start, cc + sl[1].start
        x, y = grid.xy(rr.mean(), cc.mean())
        in_ch = bool(accum_km2 is not None and np.nanpercentile(accum_km2[rr, cc], 90) >= zcfg["channel_accum_km2"])
        comps.append({
            "id": zid, "rows": rr, "cols": cc, "from": f, "to": t,
            "share": float(cnt.max() / cnt.sum()) if len(cnt) else 0.0,
            "x": float(x), "y": float(y), "in_channel": in_ch,
            "elev_mean": float(np.nanmean(elevation[rr, cc])),
            "elev_max": float(np.nanmax(elevation[rr, cc])),
            "slope_median": float(np.nanmedian(slope[rr, cc])),
            "confidence": float(np.nanmean(change["confidence"][rr, cc])),
        })
    return lab, comps


def persistence_of(comp: dict, cls3: np.ndarray | None, confirm_mask: np.ndarray | None,
                   frac: float, t3_same_season: bool = True) -> tuple[str, str]:
    """Persistent only if a third usable date or an independent observation confirms the change.

    A change into or out of snow/ice cannot be confirmed by a third date from a
    different season (seasonal melt would look 'persistent'), so it stays unconfirmed.
    """
    rr, cc = comp["rows"], comp["cols"]
    if confirm_mask is not None and confirm_mask[rr, cc].mean() >= frac:
        return "persistent", "independent_observation"
    if cls3 is not None and not t3_same_season and "snow_ice" in (comp["from"], comp["to"]):
        return "unconfirmed", "third_date_in_different_season_cannot_confirm_snow_transition"
    if cls3 is not None:
        to_idx = CLASSES.index(comp["to"])
        v = cls3[rr, cc]
        usable = v != 255
        if usable.mean() < 0.5:
            return "unconfirmed", "third_date_unusable"
        same = (v[usable] == to_idx).mean()
        return ("persistent" if same >= frac else "not_persistent"), f"third_date_agreement={same:.2f}"
    return "unconfirmed", "no_third_date_or_independent_observation"


def finalise_zones(lab, comps, zcfg, cls3=None, confirm_mask=None, pixel_area=1.0, t3_same_season=True):
    """Assign source type, apply declared slope rules, record persistence. Renumbers kept zones."""
    zones, new_lab = [], np.zeros_like(lab)
    removed = []
    for c in comps:
        st = source_type_for(c["from"], c["to"], c["in_channel"])
        min_slope = zcfg["min_median_slope_deg"].get(st, 0.0)
        if c["slope_median"] < min_slope:
            removed.append({"component": c["id"], "reason": f"median slope {c['slope_median']:.1f} < {min_slope}"})
            continue
        pers, ev = persistence_of(c, cls3, confirm_mask, zcfg["persistence_fraction"], t3_same_season)
        zid = len(zones) + 1
        new_lab[c["rows"], c["cols"]] = zid
        zones.append(Zone(
            id=zid, source_type=st, transition_from=c["from"], transition_to=c["to"],
            transition_share=c["share"], area_m2=len(c["rows"]) * pixel_area, n_pixels=len(c["rows"]),
            centroid_x=c["x"], centroid_y=c["y"], elev_mean_m=c["elev_mean"], elev_max_m=c["elev_max"],
            slope_median_deg=c["slope_median"], confidence=c["confidence"], persistence=pers,
            persistence_evidence=ev, in_channel=c["in_channel"],
        ))
    return new_lab, zones, removed
