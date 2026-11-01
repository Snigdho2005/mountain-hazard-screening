"""Command line: mhs <command> ...

  mhs demo [--out DIR] [--epochs N]      synthetic scene -> train -> screen -> evaluate
  mhs acquire CONFIG --out DIR            download Sentinel-2 + DEM (needs internet)
  mhs weak-labels CONFIG --out FILE       bootstrap partial labels (edit by hand before use)
  mhs train CONFIG --out DIR              train the U-Net on t1/t2 with reference labels
  mhs run CONFIG [--model unet.pt]        full screening run
  mhs evaluate RUN_DIR --truth FILE       score a run against reference class maps
  mhs chamoli [--pre --post --polygon]    Section 9 sensitivity / DEM recomputation
  mhs regrid SRC --like t1.tif --out F    resample a raster (e.g. 2 m DEM) onto the image grid
  mhs benchmark CONFIG --out DIR [--seeds 0 1 2]   U-Net per seed vs baseline (mean ± sd)
  mhs label-aid CONFIG --out DIR [--polygons F] [--steep --prefix R --clean 1 --avoid PTS]   random test-strip points
  mhs preview CONFIG --out DIR            small RGB / SWIR previews + split strips for QGIS labelling
  mhs rasterize-labels CONFIG --polygons F --out labels.tif   polygons (class, date) -> label raster
  mhs osm-assets CONFIG --out assets.geojson   settlements / hydropower / bridges from OpenStreetMap
  mhs event-eval CONFIG RUN_DIR [--out DIR]    ranking, removal tests, pathway, exposure vs documented event
  mhs dem-change CONFIG --out DIR [--diff F]   source detection, volume, energy, ranking from a DEM difference
  mhs climate CONFIG --out DIR                 ERA5 rainfall / temperature anomalies before the event
  mhs change-points CONFIG RUN_DIR --out DIR   blind stratified points for change-accuracy labelling
  mhs change-accuracy CONFIG --labels F --key F RUN_DIR [RUN_DIR ...]   area-weighted change accuracy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _scenes(cfg):
    from .io import read_raster
    from .model.train import Scene
    from .preprocess import build_features, slope_aspect

    inp = cfg["inputs"]
    if not inp.get("labels"):
        raise SystemExit("config inputs.labels is required for training")
    labels = read_raster(inp["labels"]).data
    dem = read_raster(inp["dem"]).data[0]
    scenes = []
    for k, key in enumerate(("t1", "t2")):
        if k >= labels.shape[0]:
            break
        img = read_raster(inp[key])
        slope, _ = slope_aspect(dem, img.res)
        scenes.append(Scene(build_features(img, slope, cfg["model"]["features"]), labels[k].astype(np.uint8)))
    return scenes


def cmd_train(cfg, out, log=print):
    from .model.train import train_unet

    return train_unet(_scenes(cfg), cfg["model"], out, log=log)


def cmd_evaluate(run_dir: Path, truth_path: str, cfg: dict, area: str = "test_strip") -> dict:
    """Segmentation and change metrics, U-Net vs baseline.

    area='test_strip': only the held-out geographic test strip of a single scene.
    area='full': the whole scene, for a catchment never used in development.
    """
    from . import IGNORE_LABEL
    from .io import read_raster
    from .metrics import change_metrics, segmentation_metrics
    from .model.train import SPLIT_TEST, geographic_split

    truth = read_raster(truth_path).data
    c1 = read_raster(run_dir / "maps/cls1.tif").data[0]
    c2 = read_raster(run_dir / "maps/cls2.tif").data[0]
    tr = read_raster(run_dir / "maps/transition.tif").data[0]
    base = read_raster(run_dir / "maps/baseline_change.tif").data[0].astype(bool)
    if area == "full":
        sel = np.ones(c1.shape, bool)
    else:
        sel = geographic_split(c1.shape, **cfg["model"]["split"]) == SPLIT_TEST
    seg = segmentation_metrics(np.concatenate([truth[0][sel], truth[1][sel]]),
                               np.concatenate([c1[sel], c2[sel]]))
    true_change = truth[0] != truth[1]
    # reference labels may be sparse: score change only where both dates are labelled
    valid = sel & (c1 != IGNORE_LABEL) & (c2 != IGNORE_LABEL) & (truth[0] != IGNORE_LABEL) & (truth[1] != IGNORE_LABEL)
    b1p, b2p = run_dir / "maps/baseline_cls1.tif", run_dir / "maps/baseline_cls2.tif"
    seg_base = None
    if b1p.exists() and b2p.exists():
        bb1, bb2 = read_raster(b1p).data[0], read_raster(b2p).data[0]
        seg_base = segmentation_metrics(np.concatenate([truth[0][sel], truth[1][sel]]),
                                        np.concatenate([bb1[sel], bb2[sel]]))
    res = {"area": area,
           "segmentation": {k: v for k, v in seg.items() if k != "confusion"},
           "segmentation_baseline": ({k: v for k, v in seg_base.items() if k != "confusion"} if seg_base else None),
           "confusion": seg["confusion"],
           "change_model": change_metrics(true_change, tr > 0, valid),
           "change_baseline": change_metrics(true_change, base, valid)}
    with open(run_dir / "evaluation.json", "w") as f:
        json.dump(res, f, indent=2)
    return res


def cmd_benchmark(cfg: dict, seeds: list[int], out: Path, log=print) -> dict:
    """Train one U-Net per seed and score each on the held-out test strip against the rule baseline.

    Only the surface maps are evaluated here (no zones/routing), so each seed is quick.
    Reports per-class F1 as mean, standard deviation, min and max across seeds.
    """
    import copy

    from . import CLASSES, IGNORE_LABEL
    from .baseline import rule_classes
    from .io import read_raster
    from .metrics import segmentation_metrics
    from .model.train import SPLIT_TEST, geographic_split, load_unet, predict_scene, train_unet
    from .preprocess import build_features, slope_aspect

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    inp = cfg["inputs"]
    labels = read_raster(inp["labels"]).data
    dem = read_raster(inp["dem"]).data[0]
    imgs = [read_raster(inp[k]) for k in ("t1", "t2")]
    slope, _ = slope_aspect(dem, imgs[0].res)
    test = geographic_split(labels.shape[1:], **cfg["model"]["split"]) == SPLIT_TEST
    truth = np.concatenate([labels[0][test], labels[1][test]])
    base = np.concatenate([rule_classes(im, slope)[test] for im in imgs])
    seg_base = segmentation_metrics(truth, base)
    scenes = _scenes(cfg)
    runs = []
    for s in seeds:
        log(f"--- seed {s} ---")
        mcfg = copy.deepcopy(cfg["model"])
        mcfg["seed"] = s
        rec = train_unet(scenes, mcfg, out / f"seed_{s}", log=lambda *a: None)
        model, meta = load_unet(rec["checkpoint"])
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(dev)
        preds = []
        for im in imgs:
            probs = predict_scene(model, meta, build_features(im, slope, meta["features"]), dev)
            c = np.nan_to_num(probs, nan=-1).argmax(0).astype(np.uint8)
            c[~np.all(np.isfinite(im.data), axis=0)] = IGNORE_LABEL
            preds.append(c[test])
        seg = segmentation_metrics(truth, np.concatenate(preds))
        runs.append({"seed": s, "best_epoch": rec["best_epoch"], "macro_f1": seg["macro_f1"],
                     "overall_accuracy": seg["overall_accuracy"],
                     "per_class_f1": {c: seg["per_class"][c]["f1"] for c in CLASSES}})
        log(f"seed {s}: best epoch {rec['best_epoch']}, test macro F1 {seg['macro_f1']:.3f}")
    present = [c for c in CLASSES if seg_base["per_class"][c]["support"] > 0]
    summary = {}
    for c in present + ["macro_f1", "overall_accuracy"]:
        v = np.array([r["per_class_f1"][c] if c in CLASSES else r[c] for r in runs])
        b = seg_base["per_class"][c]["f1"] if c in CLASSES else seg_base[c]
        summary[c] = {"unet_mean": float(v.mean()), "unet_sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                      "unet_min": float(v.min()), "unet_max": float(v.max()), "baseline": float(b),
                      "support": int(seg_base["per_class"][c]["support"]) if c in CLASSES else None}
    res = {"seeds": seeds, "area": "test_strip", "runs": runs, "summary": summary,
           "baseline_note": "rule baseline is deterministic (same value for every seed)"}
    with open(out / "benchmark.json", "w") as f:
        json.dump(res, f, indent=2)
    log(f"\nU-Net ({len(seeds)} seeds) vs rule baseline, held-out test strip")
    log(f"{'class':14s} {'support':>8s} {'U-Net F1 mean ± sd':>20s} {'min-max':>13s} {'baseline F1':>12s}")
    for c, m in summary.items():
        sup = f"{m['support']:8d}" if m["support"] is not None else " " * 8
        log(f"{c:14s} {sup} {m['unet_mean']:11.3f} ± {m['unet_sd']:.3f} "
            f"{m['unet_min']:6.3f}-{m['unet_max']:.3f} {m['baseline']:12.3f}")
    return res


def print_evaluation(res: dict) -> None:
    """Readable U-Net vs baseline table on the evaluation area."""
    seg, base = res["segmentation"], res.get("segmentation_baseline")
    print(f"\nEvaluation area: {res['area']}  (labelled pixels only)")
    print(f"{'class':12s} {'support':>8s} {'U-Net F1':>9s} {'base F1':>8s} {'U-Net IoU':>10s} {'base IoU':>9s}")
    for c, m in seg["per_class"].items():
        if m["support"] == 0:
            continue
        b = base["per_class"][c] if base else {"f1": float("nan"), "iou": float("nan")}
        print(f"{c:12s} {m['support']:8d} {m['f1']:9.3f} {b['f1']:8.3f} {m['iou']:10.3f} {b['iou']:9.3f}")
    print(f"{'macro F1':12s} {'':8s} {seg['macro_f1']:9.3f} {base['macro_f1'] if base else float('nan'):8.3f}")
    print(f"{'overall acc':12s} {'':8s} {seg['overall_accuracy']:9.3f} "
          f"{base['overall_accuracy'] if base else float('nan'):8.3f}")
    cm = res["change_model"]
    if cm["tp"] + cm["fn"] == 0:
        print("change scores: not measurable here (no labelled before/after change in this area)")
    else:
        for k in ("change_model", "change_baseline"):
            m = res[k]
            print(f"{k:16s} F1 {m['f1']:.3f}  false-alarm {m['false_alarm_rate']:.4f}  missed {m['missed_change_rate']:.3f}")


def cmd_demo(out: Path, epochs: int | None, log=print):
    """Develop on one synthetic catchment, test on a second, geographically different one."""
    from .config import load_config
    from .pipeline import run_pipeline
    from .synthetic import generate

    log("1/5 generating development catchment (seed 0) and test catchment (seed 1)")
    dev, test = generate(out / "dev_data", seed=0), generate(out / "test_data", seed=1)
    split = {"split": {"test_fraction": 0.0, "val_fraction": 0.2}}
    cfg_dev = load_config(dev["config"], {"model": split})
    cfg_test = load_config(test["config"], {"model": split})
    if epochs:
        cfg_dev["model"]["epochs"] = epochs
    log("2/5 training U-Net on the development catchment (east strip for validation)")
    rec = cmd_train(cfg_dev, out / "model", log)
    log(f"     best val macro F1 {rec['best_val_macro_f1']:.3f} at epoch {rec['best_epoch']}")
    log("3/5 screening run on the development catchment")
    run_pipeline(cfg_dev, rec["checkpoint"], out / "run_dev", log=log)
    log("4/5 screening run on the unseen test catchment (no retraining, no threshold changes)")
    result = run_pipeline(cfg_test, rec["checkpoint"], out / "run_test", log=log)
    log("5/5 evaluation on the test catchment against its truth")
    ev = cmd_evaluate(out / "run_test", str(out / "test_data/truth.tif"), cfg_test, area="full")
    log(f"     surface map: macro F1 {ev['segmentation']['macro_f1']:.3f}, mIoU {ev['segmentation']['miou']:.3f}")
    for k in ("change_model", "change_baseline"):
        m = ev[k]
        log(f"     {k:16s} F1 {m['f1']:.3f}  false-alarm {m['false_alarm_rate']:.4f}  missed {m['missed_change_rate']:.3f}")
    t = test["truth"]
    log(f"     truth: scar dV {t['scar_volume_change_m3']:,.0f} m3, glacier dV {t['glacier_volume_loss_m3']:,.0f} m3, "
        f"lake growth {t['lake_growth_m2']:,.0f} m2")
    return result, ev


def main(argv=None):
    from .config import load_config

    ap = argparse.ArgumentParser(prog="mhs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("demo"); p.add_argument("--out", default="demo_output"); p.add_argument("--epochs", type=int)
    p = sub.add_parser("acquire"); p.add_argument("config"); p.add_argument("--out", required=True)
    p.add_argument("--only", nargs="+", help="epochs to (re)download, e.g. t3 or dem t1")
    p = sub.add_parser("weak-labels"); p.add_argument("config"); p.add_argument("--out", required=True)
    p.add_argument("--fraction", type=float, default=0.2)
    p = sub.add_parser("train"); p.add_argument("config"); p.add_argument("--out", required=True)
    p = sub.add_parser("run"); p.add_argument("config"); p.add_argument("--model"); p.add_argument("--out")
    p = sub.add_parser("evaluate"); p.add_argument("run_dir"); p.add_argument("--truth", required=True)
    p.add_argument("--config"); p.add_argument("--area", choices=["test_strip", "full"], default="test_strip")
    p = sub.add_parser("chamoli"); p.add_argument("--pre"); p.add_argument("--post"); p.add_argument("--polygon")
    p.add_argument("--ref-elev", type=float); p.add_argument("--ref-x", type=float); p.add_argument("--ref-y", type=float)
    p = sub.add_parser("benchmark"); p.add_argument("config"); p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p = sub.add_parser("preview"); p.add_argument("config"); p.add_argument("--out", required=True)
    p = sub.add_parser("label-aid"); p.add_argument("config"); p.add_argument("--out", required=True)
    p.add_argument("--polygons"); p.add_argument("--n", type=int, default=30)
    p.add_argument("--steep", action="store_true", help="only steep (>35 deg), unshadowed bare ground")
    p.add_argument("--prefix", default="P"); p.add_argument("--avoid", help="existing points to stay away from")
    p.add_argument("--clean", type=int, default=0, help="require a clean (2k+1)^2 neighbourhood")
    p.add_argument("--seed", type=int, default=11)
    p = sub.add_parser("rasterize-labels"); p.add_argument("config"); p.add_argument("--polygons", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("osm-assets"); p.add_argument("config"); p.add_argument("--out", required=True)
    p = sub.add_parser("event-eval"); p.add_argument("config"); p.add_argument("run_dir"); p.add_argument("--out")
    p = sub.add_parser("dem-change"); p.add_argument("config"); p.add_argument("--out", required=True)
    p.add_argument("--diff")
    p = sub.add_parser("climate"); p.add_argument("config"); p.add_argument("--out", required=True)
    p = sub.add_parser("change-points"); p.add_argument("config"); p.add_argument("run_dir")
    p.add_argument("--out", required=True); p.add_argument("--n", type=int, nargs=4, default=[15, 15, 15, 25])
    p = sub.add_parser("change-accuracy"); p.add_argument("config"); p.add_argument("run_dirs", nargs="+")
    p.add_argument("--labels", required=True); p.add_argument("--key", required=True); p.add_argument("--out")
    p = sub.add_parser("regrid"); p.add_argument("src"); p.add_argument("--like", required=True)
    p.add_argument("--out", required=True); p.add_argument("--resampling", default="average")
    a = ap.parse_args(argv)

    if a.cmd == "demo":
        cmd_demo(Path(a.out), a.epochs)
    elif a.cmd == "acquire":
        from .acquire import acquire_case

        print(json.dumps(acquire_case(load_config(a.config), a.out, a.only), indent=2))
    elif a.cmd == "weak-labels":
        from .baseline import weak_labels
        from .io import Raster, read_raster, write_raster
        from .preprocess import slope_aspect

        cfg = load_config(a.config)
        dem = read_raster(cfg["inputs"]["dem"]).data[0]
        labs = []
        for k in ("t1", "t2"):
            img = read_raster(cfg["inputs"][k])
            labs.append(weak_labels(img, slope_aspect(dem, img.res)[0], a.fraction))
        write_raster(a.out, img.like(np.stack(labs), ["labels_t1", "labels_t2"], 255), "uint8")
        print(f"wrote {a.out}. Soil and debris are unlabelled: digitise them before training.")
    elif a.cmd == "train":
        rec = cmd_train(load_config(a.config), a.out)
        print(f"checkpoint {rec['checkpoint']}  best val macro F1 {rec['best_val_macro_f1']:.3f}")
    elif a.cmd == "run":
        from .pipeline import run_pipeline

        run_pipeline(load_config(a.config), a.model, a.out)
    elif a.cmd == "evaluate":
        res = cmd_evaluate(Path(a.run_dir), a.truth, load_config(a.config), a.area)
        print_evaluation(res)
    elif a.cmd == "chamoli":
        from .chamoli import from_dems, sensitivity

        if a.pre:
            ref_xy = (a.ref_x, a.ref_y) if a.ref_x is not None else None
            res = from_dems(a.pre, a.post, a.polygon, ref_xy, a.ref_elev)
        else:
            res = sensitivity()
        print(json.dumps(res, indent=2, default=float))
    elif a.cmd == "benchmark":
        cmd_benchmark(load_config(a.config), a.seeds, Path(a.out))
    elif a.cmd == "label-aid":
        from .labels import label_aid

        print(json.dumps(label_aid(load_config(a.config), a.out, a.polygons, a.n, seed=a.seed,
                                   steep_only=a.steep, prefix=a.prefix, avoid_points=a.avoid,
                                   clean_px=a.clean), indent=2))
    elif a.cmd == "preview":
        from .labels import write_previews

        for f in write_previews(load_config(a.config), a.out):
            print("wrote", f)
    elif a.cmd == "rasterize-labels":
        from .labels import label_coverage_by_split, rasterize_labels

        cfg = load_config(a.config)
        res = rasterize_labels(cfg, a.polygons, a.out)
        print(json.dumps(res, indent=2))
        cov = label_coverage_by_split(cfg, a.out)
        print("labelled pixels per split strip (t1 + t2):")
        for split, cnt in cov.items():
            print(f"  {split:10s} " + "  ".join(f"{k} {v}" for k, v in cnt.items()))
        missing = [k for k, v in cov["test"].items() if v == 0]
        if missing:
            print(f"WARNING: no test-strip labels for {missing}; draw some in the east strip")
    elif a.cmd == "osm-assets":
        from .assets_osm import fetch_osm_assets

        print(json.dumps(fetch_osm_assets(load_config(a.config), a.out), indent=2))
    elif a.cmd == "event-eval":
        from .event_eval import evaluate_event, event_figure, summary_markdown

        cfg = load_config(a.config)
        ev = evaluate_event(cfg, a.run_dir)
        out = Path(a.out or Path(a.run_dir) / "event_eval")
        out.mkdir(parents=True, exist_ok=True)
        md = summary_markdown(ev)
        (out / "event_eval.md").write_text(md)
        event_figure(cfg, ev, a.run_dir, out / "event_route.png")
        ev.pop("_route_path", None)
        (out / "event_eval.json").write_text(json.dumps(ev, indent=2, default=float))
        print(md)
        print(f"wrote {out}/event_eval.md, event_eval.json, event_route.png")
    elif a.cmd == "dem-change":
        from .dem_change import dem_change, dem_change_figure, summary_markdown as md_dc

        cfg = load_config(a.config)
        r = dem_change(cfg, a.out, a.diff)
        dem_change_figure(cfg, r, a.out)
        r.pop("_route_path", None)
        Path(a.out, "dem_change.json").write_text(json.dumps(r, indent=2, default=float))
        md = md_dc(r)
        Path(a.out, "dem_change.md").write_text(md)
        print(md)
        print(f"wrote {a.out}/dem_change.md, dem_change.json, dem_change.png, dem_loss_zones.tif, dz_10m.tif")
    elif a.cmd == "climate":
        from .climate import climate_record, summary_markdown as md_cl

        r = climate_record(load_config(a.config))
        Path(a.out).mkdir(parents=True, exist_ok=True)
        Path(a.out, "climate.json").write_text(json.dumps(r, indent=2, default=float))
        md = md_cl(r)
        Path(a.out, "climate.md").write_text(md)
        print(md)
    elif a.cmd == "change-points":
        from .change_accuracy import change_points

        print(json.dumps(change_points(load_config(a.config), a.run_dir, a.out, tuple(a.n)), indent=2))
    elif a.cmd == "change-accuracy":
        from .change_accuracy import change_accuracy, summary_markdown as md_ca

        r = change_accuracy(load_config(a.config), a.labels, a.key, a.run_dirs)
        md = md_ca(r)
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            Path(a.out, "change_accuracy.json").write_text(json.dumps(r, indent=2, default=float))
            Path(a.out, "change_accuracy.md").write_text(md)
        print(md)
    elif a.cmd == "regrid":
        from .io import read_raster, regrid, write_raster

        write_raster(a.out, regrid(a.src, read_raster(a.like), a.resampling), "float32")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
