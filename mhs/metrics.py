"""Evaluation measures (paper Section 8)."""
from __future__ import annotations

import numpy as np

from . import CLASSES, IGNORE_LABEL, N_CLASSES


def confusion(y_true, y_pred, n=N_CLASSES) -> np.ndarray:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    m = (y_true != IGNORE_LABEL) & (y_pred != IGNORE_LABEL) & (y_true < n) & (y_pred < n)
    return np.bincount(n * y_true[m].astype(int) + y_pred[m].astype(int), minlength=n * n).reshape(n, n)


def segmentation_metrics(y_true, y_pred, names=CLASSES) -> dict:
    """Class-wise IoU, precision, recall, F1, plus macro F1 and mIoU over present classes."""
    cm = confusion(y_true, y_pred, len(names))
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    present = cm.sum(1) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)
    per = {n: {"iou": float(iou[i]), "precision": float(prec[i]), "recall": float(rec[i]),
               "f1": float(f1[i]), "support": int(cm.sum(1)[i])} for i, n in enumerate(names)}
    return {
        "per_class": per,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "miou": float(iou[present].mean()) if present.any() else 0.0,
        "overall_accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "confusion": cm.tolist(),
    }


def change_metrics(true_change, pred_change, valid=None) -> dict:
    """Change-class F1, false-alarm rate and missed-change rate."""
    t = np.asarray(true_change, bool)
    p = np.asarray(pred_change, bool)
    if valid is not None:
        t, p = t[valid], p[valid]
    tp = int((t & p).sum())
    fp = int((~t & p).sum())
    fn = int((t & ~p).sum())
    tn = int((~t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "precision": prec,
        "recall": rec,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
        "missed_change_rate": fn / (fn + tp) if fn + tp else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def interval_coverage(intervals, truths) -> float:
    """Fraction of reference values that fall inside reported (low, high) intervals."""
    hits = [lo <= t <= hi for (lo, hi), t in zip(intervals, truths) if t is not None]
    return float(np.mean(hits)) if hits else float("nan")


def hits_at_k(ranked_ids, event_ids, k) -> float:
    """Fraction of documented event sources found among the top-k ranked zones."""
    top = set(list(ranked_ids)[:k])
    ev = set(event_ids)
    return len(top & ev) / len(ev) if ev else float("nan")


def zone_overlap(pred_mask, true_mask) -> dict:
    p, t = np.asarray(pred_mask, bool), np.asarray(true_mask, bool)
    inter, union = (p & t).sum(), (p | t).sum()
    return {"iou": float(inter / union) if union else 0.0,
            "dice": float(2 * inter / (p.sum() + t.sum())) if p.sum() + t.sum() else 0.0}
