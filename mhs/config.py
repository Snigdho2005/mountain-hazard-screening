"""Run configuration.

Every threshold, parameter range and rule the pipeline uses is declared here and
archived with each run, so the protocol is fixed before outcomes are examined
(paper Section 12). Values below are defaults for a first case study; change
them in a YAML file rather than in code.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "case": {
        "name": "unnamed_case",
        "bbox": None,            # [min_lon, min_lat, max_lon, max_lat]
        "crs": None,             # e.g. "EPSG:32644"; None -> UTM zone of bbox centre
        "resolution": 10.0,      # metres
        "notes": "",
    },
    "inputs": {
        "t1": "data/t1.tif",     # prepared reflectance stacks on a common grid
        "t2": "data/t2.tif",
        "t3": None,              # optional third date for persistence
        "dem": "data/dem.tif",   # elevation used for slope and routing
        "dem_t1": None,          # optional multi-temporal DEMs for changed volume
        "dem_t2": None,
        "labels": None,          # optional reference labels (uint8, 255 = unlabelled)
        "assets": None,          # optional GeoJSON of settlements / infrastructure
        "independent_confirmation": None,  # optional GeoJSON polygons confirming change
    },
    "acquire": {
        "collection": "sentinel-2-l2a",
        "dem_collection": "cop-dem-glo-30",
        "dates": [],             # list of "YYYY-MM-DD/YYYY-MM-DD" windows, one per epoch
        "max_cloud": 20.0,
        "bands": ["B02", "B03", "B04", "B08", "B11", "B12"],
    },
    "preprocess": {
        "max_shift_px": 0.5,     # residual misregistration above this is corrected
        "reject_shift_px": 2.0,  # ... and above this the image pair is rejected
        "max_season_gap_days": 45,
    },
    "model": {
        "features": ["B02", "B03", "B04", "B08", "B11", "B12", "NDWI", "NDSI", "NDVI", "slope"],
        "base_channels": 32,
        "depth": 4,
        "patch_size": 128,
        "batch_size": 16,
        "epochs": 30,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 6,           # early stopping on validation macro F1
        "seed": 42,
        "class_weighting": True,
        "samples_per_epoch": 800,
        "split": {               # geographic split in pixel blocks
            "block_px": 64,
            "test_fraction": 0.25,   # eastmost blocks held out for testing
            "val_fraction": 0.15,    # next blocks used for validation / early stopping
            "gap_blocks": 1,         # unused buffer between development and test areas
            "val_side": "east",      # "west" puts validation at the west edge instead
        },
    },
    "change": {
        "min_confidence": 0.6,   # min class probability on both dates
        "use_probability_gate": True,
        "illumination_gate": True,   # drop pixels lit on one date and shadowed on the other
        "min_cos_incidence": 0.15,   # below this a slope counts as self-shadowed
        # Class pairs whose swaps are not counted as change (spectrally ambiguous,
        # not hazard sources). Either direction is ignored. Recorded in each run.
        "ignore_transitions": [],
    },
    "zones": {
        "min_area_m2": 2000.0,
        "opening_px": 1,
        "persistence_fraction": 0.5,
        "min_median_slope_deg": {  # declared slope rules per source type
            "water_storage": 0.0,
            "snow_ice_storage": 15.0,
            "rock_soil_storage": 15.0,
            "channel_storage": 0.0,
        },
        "channel_accum_km2": 1.0,
    },
    "physics": {
        "n_samples": 20000,
        "interval": 0.90,
        "seed": 7,
        "dz_sigma_m": 2.0,          # DEM-difference uncertainty (1 sigma)
        "dz_correlation_m": 100.0,  # spatial correlation range of DEM error
        "area_rel_sigma": 0.10,     # relative error of mapped area
        "density_kg_m3": {          # bulk density ranges (low, high), uniform
            "water_storage": [1000, 1000],
            "snow_ice_storage": [400, 900],
            "rock_soil_storage": [1800, 2600],
            "channel_storage": [1800, 2300],
        },
        # Thickness scenarios (low, mid, high) in metres for AVAILABLE volume.
        # Leave a source type out to keep it as an area-change record only.
        "thickness_scenarios_m": {},
        "max_thickness_ratio": 10.0,   # wider scenario range -> no energy assigned
        "max_interval_ratio": 10.0,    # wider energy interval -> inconclusive
        "reference_elevation_m": None, # fixed h_r; None -> first valley-floor point on route
        "energy_threshold_J": 1.0e13,  # screening threshold for exceedance probability
    },
    "routing": {
        "max_route_km": 30.0,
        "corridor_halfwidth_m": 150.0,
        "valley_slope_deg": 8.0,
        "valley_accum_km2": 2.0,
        "fill_depth_flag_m": 30.0,   # 30 m GLO-30 has pits in deep gorges
    },
    # Retrospective event evaluation (paper Section 8: ranking, removal tests, pathway).
    # The reference event is declared here, from published sources, before outcomes are examined.
    "evaluation": {
        "event": None,
        "source_radius_m": [500, 1000, 2000],   # zones within this distance of the source count as a hit
        "top_k": [5, 10, 25],
        "checkpoint_tolerance_m": 250.0,         # route counts as reaching a checkpoint within this distance
    },
    "output": {
        "dir": "runs",
        "figures": True,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
        base = Path(path).resolve().parent
        for k, v in cfg["inputs"].items():
            if isinstance(v, str) and not Path(v).is_absolute():
                cfg["inputs"][k] = str((base / v).resolve())
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
