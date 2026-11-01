"""Change mapping from two class maps (paper Table 1, 'Change mapping')."""
from __future__ import annotations

import numpy as np

from . import CLASS_INDEX, CLASSES, IGNORE_LABEL, N_CLASSES

NO_CHANGE = 0  # transition code reserved for unchanged pixels


def transition_code(c1, c2):
    """Encode from->to as 1 + from*N + to (0 is reserved for no change)."""
    return 1 + c1.astype(np.int32) * N_CLASSES + c2.astype(np.int32)


def decode_transition(code: int) -> tuple[str, str]:
    k = int(code) - 1
    return CLASSES[k // N_CLASSES], CLASSES[k % N_CLASSES]


def class_change(cls1, cls2, prob1=None, prob2=None, min_confidence=0.6, use_gate=True) -> dict:
    """Compare class maps directly. Low-confidence pixels on either date are not called changed."""
    valid = (cls1 != IGNORE_LABEL) & (cls2 != IGNORE_LABEL)
    changed = valid & (cls1 != cls2)
    conf = np.ones(cls1.shape, np.float32)
    if prob1 is not None and prob2 is not None:
        with np.errstate(invalid="ignore"):
            p1 = np.where(np.isfinite(prob1), prob1, 0).max(0)
            p2 = np.where(np.isfinite(prob2), prob2, 0).max(0)
        conf = np.minimum(p1, p2)
        conf = np.nan_to_num(conf, nan=0.0).astype(np.float32)
        if use_gate:
            changed &= conf >= min_confidence
    code = np.where(changed, transition_code(cls1, cls2), NO_CHANGE).astype(np.int32)
    return {"changed": changed, "transition": code, "confidence": conf, "valid": valid}


def index_differences(idx1: dict, idx2: dict) -> dict[str, np.ndarray]:
    """Supporting evidence: per-pixel index differences t2 - t1."""
    return {f"d{k}": (idx2[k] - idx1[k]).astype(np.float32) for k in ("NDWI", "NDSI", "NDVI", "BSI")}


def alternative_explanations(transition_from: str, transition_to: str, bright1: float, bright2: float,
                             season_gap: int, max_season_gap: int) -> list[str]:
    """Flags that a specialist must rule out before a change is treated as real (Table 2)."""
    flags = []
    pair = {transition_from, transition_to}
    if "snow_ice" in pair:
        flags.append("seasonal_snow" if season_gap > max_season_gap // 2 else "snow_variability")
    if "water" in pair:
        flags.append("water_level_change")
    if min(bright1, bright2) < 0.04:
        flags.append("possible_shadow")
    if season_gap > max_season_gap:
        flags.append("season_mismatch")
    return flags


def transition_summary(code_map: np.ndarray, pixel_area: float) -> list[dict]:
    codes, counts = np.unique(code_map[code_map != NO_CHANGE], return_counts=True)
    out = []
    for c, n in sorted(zip(codes, counts), key=lambda t: -t[1]):
        f, t = decode_transition(c)
        out.append({"from": f, "to": t, "pixels": int(n), "area_m2": float(n * pixel_area)})
    return out


SOURCE_TYPE_RULES = {
    # (to-class) -> source type, applied after the from-class rules below
    "water": "water_storage",
    "snow_ice": "snow_ice_storage",
    "rock": "rock_soil_storage",
    "soil": "rock_soil_storage",
    "debris": "rock_soil_storage",
}


def source_type_for(transition_from: str, transition_to: str, in_channel: bool) -> str:
    """Assign a paper Section 5 source category to a dominant class transition."""
    if transition_to == "debris" and in_channel:
        return "channel_storage"
    if transition_from == "water" or transition_to == "water":
        return "water_storage"
    if transition_from == "snow_ice" and transition_to in ("rock", "soil", "debris") and not in_channel:
        # snow/ice loss exposing ground: thinning front or new scar
        return "snow_ice_storage" if transition_to == "rock" else "rock_soil_storage"
    return SOURCE_TYPE_RULES.get(transition_to, "rock_soil_storage")


__all__ = ["class_change", "decode_transition", "index_differences", "alternative_explanations",
           "transition_summary", "source_type_for", "CLASS_INDEX"]
