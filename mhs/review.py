"""Review states (paper Section 11, Table 4).

The software only *suggests* a state. 'escalated_technical_review' can be set
only by a named specialist; the most the pipeline proposes is
'review_candidate_for_escalation'. No state authorises warnings, closures,
evacuation or engineering action.
"""
from __future__ import annotations

ROUTINE = "routine"
REVIEW = "review"
ESCALATION_CANDIDATE = "review_candidate_for_escalation"
ESCALATED = "escalated_technical_review"


def suggest_state(zone: dict) -> dict:
    reasons = []
    route_ok = zone.get("route") and "route_too_short" not in zone["route"]["qc_flags"]
    exposed = bool(zone.get("exposure") and zone["exposure"]["assets"])
    persistent = zone["persistence"] == "persistent"
    signals = zone.get("instability", {}).get("independent_signals", 0)
    weak = zone["confidence"] < 0.6 or zone["persistence"] == "not_persistent"

    if weak or not route_ok:
        state = ROUTINE
        reasons.append("evidence too weak or no credible pathway")
    else:
        state = REVIEW
        reasons.append("new or changed surface signal present but uncertain")
        if persistent and exposed and signals >= 2:
            state = ESCALATION_CANDIDATE
            reasons.append("persistent change with >=2 independent signals and exposed assets on route")
        elif not persistent:
            reasons.append("check another date or sensor for persistence")
    if zone.get("alt_explanations"):
        reasons.append("rule out: " + ", ".join(zone["alt_explanations"]))
    return {"suggested_state": state, "reasons": reasons, "specialist_state": None,
            "reviewer": None, "note": "Suggested by software; requires specialist review."}


def apply_specialist_review(zone: dict, state: str, reviewer: str, note: str) -> dict:
    if state not in (ROUTINE, REVIEW, ESCALATED):
        raise ValueError("specialist state must be routine, review or escalated_technical_review")
    if not reviewer:
        raise ValueError("a named reviewer is required")
    if not note:
        raise ValueError("a written uncertainty note is required")
    zone["review"].update({"specialist_state": state, "reviewer": reviewer, "uncertainty_note": note})
    return zone
