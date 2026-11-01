"""Training and inference for the six-class surface U-Net.

Geographic separation: the scene is cut into vertical strips of `block_px`
columns. The eastmost strips form the test area, then a gap that is never used,
then validation strips; the rest is for training. Training patches are drawn
only from inside the training columns, so no test pixel is ever seen as input
or label during development.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import CLASSES, IGNORE_LABEL, N_CLASSES
from ..metrics import segmentation_metrics
from .unet import UNet

SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, SPLIT_GAP = 0, 1, 2, 3


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def geographic_split(shape, block_px=64, test_fraction=0.25, val_fraction=0.15, gap_blocks=1,
                     val_side: str = "east") -> np.ndarray:
    """Vertical strips: [train | val | gap | test] (val_side='east') or [val | train | gap | test] ('west')."""
    h, w = shape
    nb = math.ceil(w / block_px)
    n_test = max(1, round(nb * test_fraction)) if test_fraction > 0 else 0
    n_val = max(1, round(nb * val_fraction))
    gap = gap_blocks if n_test else 0
    if n_test + gap + n_val >= nb:
        raise ValueError("scene too narrow for the requested geographic split")
    col_split = np.full(nb, SPLIT_TRAIN)
    if n_test:  # test_fraction = 0 when testing uses a separate catchment
        col_split[nb - n_test:] = SPLIT_TEST
        col_split[nb - n_test - gap: nb - n_test] = SPLIT_GAP
    gap_blocks = gap
    if val_side == "west":
        col_split[:n_val] = SPLIT_VAL
    else:
        col_split[nb - n_test - gap_blocks - n_val: nb - n_test - gap_blocks] = SPLIT_VAL
    cols = np.repeat(col_split, block_px)[:w]
    return np.broadcast_to(cols, shape).copy()


@dataclass
class Scene:
    """One date: features (C,H,W) and labels (H,W) with IGNORE_LABEL where unlabelled."""
    features: np.ndarray
    labels: np.ndarray


def _norm_stats(scenes, split):
    c = scenes[0].features.shape[0]
    vals = [s.features[:, split == SPLIT_TRAIN].reshape(c, -1) for s in scenes]
    v = np.concatenate(vals, axis=1)
    mean = np.nanmean(v, axis=1)
    std = np.nanstd(v, axis=1) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def normalise(feat, mean, std):
    x = (feat - mean[:, None, None]) / std[:, None, None]
    return np.nan_to_num(x, nan=0.0).astype(np.float32)


def _class_weights(scenes, split):
    counts = np.zeros(N_CLASSES)
    for s in scenes:
        lab = s.labels[(split == SPLIT_TRAIN) & (s.labels != IGNORE_LABEL)]
        counts += np.bincount(lab, minlength=N_CLASSES)[:N_CLASSES]
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    w[counts == 0] = 0.0
    return (w / w[w > 0].mean()).astype(np.float32), counts


def _sample_patch(scene_x, scene_y, split, p, rng):
    h, w = scene_y.shape
    train_cols = np.where(split[0] == SPLIT_TRAIN)[0]
    c0, c1 = train_cols.min(), train_cols.max() + 1
    for _ in range(50):
        r = rng.integers(0, max(1, h - p + 1))
        c = rng.integers(c0, max(c0 + 1, c1 - p + 1))
        y = scene_y[r:r + p, c:c + p].copy()
        y[split[r:r + p, c:c + p] != SPLIT_TRAIN] = IGNORE_LABEL
        if (y != IGNORE_LABEL).sum() >= 0.01 * p * p:
            return scene_x[:, r:r + p, c:c + p], y
    return scene_x[:, r:r + p, c:c + p], y


def _augment(x, y, rng):
    k = int(rng.integers(0, 4))
    x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


@torch.no_grad()
def predict_probs(model, x: np.ndarray, patch=256, overlap=32, device="cpu", batch=4) -> np.ndarray:
    """Sliding-window softmax probabilities (n_classes, H, W) with linear blending."""
    model.eval()
    c, h, w = x.shape
    patch = min(patch, max(h, w))
    step = max(1, patch - overlap)
    rows = list(range(0, max(h - patch, 0) + 1, step))
    cols = list(range(0, max(w - patch, 0) + 1, step))
    if rows[-1] + patch < h:
        rows.append(h - patch)
    if cols[-1] + patch < w:
        cols.append(w - patch)
    ramp = np.minimum(np.arange(1, patch + 1), np.arange(patch, 0, -1)).astype(np.float32)
    wt = np.minimum.outer(ramp, ramp)
    wt = wt / wt.max()
    acc = np.zeros((N_CLASSES, h, w), np.float32)
    norm = np.zeros((h, w), np.float32)
    coords = [(r, cc) for r in rows for cc in cols]
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        tiles = np.stack([x[:, r:r + patch, cc:cc + patch] for r, cc in chunk])
        ph, pw = tiles.shape[-2:]
        logits = model(torch.from_numpy(tiles).to(device))
        prob = F.softmax(logits, 1).cpu().numpy()
        for (r, cc), pr in zip(chunk, prob):
            ww = wt[:ph, :pw]
            acc[:, r:r + ph, cc:cc + pw] += pr * ww
            norm[r:r + ph, cc:cc + pw] += ww
    return acc / np.maximum(norm, 1e-6)


def train_unet(scenes: list[Scene], mcfg: dict, out_dir: str | Path, device: str | None = None,
               log=print) -> dict:
    """Train one U-Net on all supplied dates; returns a record of the run."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(mcfg["seed"])
    rng = np.random.default_rng(mcfg["seed"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shape = scenes[0].labels.shape
    split = geographic_split(shape, **mcfg["split"])
    mean, std = _norm_stats(scenes, split)
    xs = [normalise(s.features, mean, std) for s in scenes]
    weights, counts = _class_weights(scenes, split)
    log(f"train pixels per class: {dict(zip(CLASSES, counts.astype(int).tolist()))}")

    in_ch = xs[0].shape[0]
    model = UNet(in_ch, N_CLASSES, mcfg["base_channels"], mcfg["depth"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=mcfg["lr"], weight_decay=mcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=mcfg["epochs"])
    cw = torch.tensor(weights, device=device) if mcfg["class_weighting"] else None

    p, bs = mcfg["patch_size"], mcfg["batch_size"]
    steps = max(1, mcfg["samples_per_epoch"] // bs)
    best, best_epoch, history, bad, best_f1 = -np.inf, -1, [], 0, float("nan")
    meta = {"in_channels": in_ch, "features": mcfg["features"], "classes": CLASSES,
            "base_channels": mcfg["base_channels"], "depth": mcfg["depth"],
            "mean": mean.tolist(), "std": std.tolist(), "seed": mcfg["seed"],
            "split": mcfg["split"]}
    ckpt = out_dir / "unet.pt"
    t0 = time.time()
    for epoch in range(mcfg["epochs"]):
        model.train()
        losses = []
        for _ in range(steps):
            bx, by = [], []
            for _ in range(bs):
                k = int(rng.integers(0, len(xs)))
                x, y = _sample_patch(xs[k], scenes[k].labels, split, p, rng)
                x, y = _augment(x, y, rng)
                bx.append(x)
                by.append(y)
            xb = torch.from_numpy(np.stack(bx)).to(device)
            yb = torch.from_numpy(np.stack(by).astype(np.int64)).to(device)
            loss = F.cross_entropy(model(xb), yb, weight=cw, ignore_index=IGNORE_LABEL)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()

        yt, yp, nll = [], [], []
        for x, s in zip(xs, scenes):
            prob = predict_probs(model, x, device=device)
            pred = prob.argmax(0)
            m = (split == SPLIT_VAL) & (s.labels != IGNORE_LABEL)
            yt.append(s.labels[m])
            yp.append(pred[m])
            p_true = prob[s.labels[m].astype(int), np.nonzero(m)[0], np.nonzero(m)[1]]
            nll.append(-np.log(np.clip(p_true, 1e-6, 1)))
        vm = segmentation_metrics(np.concatenate(yt), np.concatenate(yp))
        vloss = float(np.mean(np.concatenate(nll))) if sum(len(a) for a in nll) else float("nan")
        n_val_classes = int(len(np.unique(np.concatenate(yt))))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_loss": vloss,
                        "val_macro_f1": vm["macro_f1"], "val_miou": vm["miou"]})
        log(f"epoch {epoch + 1:3d}  loss {np.mean(losses):.4f}  val loss {vloss:.4f}  "
            f"val macroF1 {vm['macro_f1']:.3f} ({n_val_classes} classes)")
        score = -vloss  # selection on validation cross-entropy: stable even with few val classes
        if score > best:
            best, best_epoch, bad = score, epoch + 1, 0
            best_f1 = vm["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "meta": meta}, ckpt)
        else:
            bad += 1
            if bad >= mcfg["patience"]:
                log(f"early stop: no improvement for {bad} epochs")
                break

    record = {"checkpoint": str(ckpt), "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
              "best_val_loss": -best, "selection": "min validation cross-entropy",
              "history": history, "train_seconds": round(time.time() - t0, 1), "device": device,
              "class_counts_train": counts.astype(int).tolist(), "class_weights": weights.tolist(),
              "hyperparameters": {k: mcfg[k] for k in ("patch_size", "batch_size", "epochs", "lr",
                                                       "weight_decay", "patience", "samples_per_epoch",
                                                       "class_weighting")},
              "augmentation": "random rot90 + horizontal flip", "loss": "weighted cross-entropy",
              "meta": meta}
    with open(out_dir / "training_record.json", "w") as f:
        json.dump(record, f, indent=2)
    return record


def load_unet(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    meta = ck["meta"]
    model = UNet(meta["in_channels"], len(meta["classes"]), meta["base_channels"], meta["depth"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), meta


def predict_scene(model, meta, features: np.ndarray, device="cpu") -> np.ndarray:
    mean, std = np.array(meta["mean"], np.float32), np.array(meta["std"], np.float32)
    probs = predict_probs(model, normalise(features, mean, std), device=device)
    probs[:, ~np.all(np.isfinite(features), axis=0)] = np.nan
    return probs
