import json

import numpy as np
import pytest
import torch

from mhs import CLASS_INDEX, G, IGNORE_LABEL
from mhs.change import class_change, decode_transition, transition_code
from mhs.chamoli import sensitivity
from mhs.instability import percentile_score
from mhs.metrics import change_metrics, segmentation_metrics
from mhs.model.train import SPLIT_GAP, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, geographic_split
from mhs.model.unet import UNet
from mhs.physics import (Dist, changed_volume, column_energy, delta_energy, energy_from_hypsometry,
                         energy_interval, mass_hypsometry, route_energy_balance, thickness_scenario_ok)
from mhs.routing import FlowGrid, fill_depressions


# ---------------------------------------------------------------- physics
def test_point_mass_energy_matches_mgh():
    e = energy_interval(Dist("fixed", 1000.0), Dist("fixed", 2000.0), Dist("fixed", 100.0), n=100)
    assert e["energy_J"]["median"] == pytest.approx(1000 * 2000 * G * 100)


def test_column_energy_equals_hypsometry_and_reduces_to_point_mass():
    rng = np.random.default_rng(0)
    surf = 3000 + rng.uniform(0, 50, (20, 20))
    thick = np.full((20, 20), 4.0)
    mask = np.ones((20, 20), bool)
    E = column_energy(surf, thick, mask, 2400, 100.0, 2000.0)
    z, mu = mass_hypsometry(surf, thick, mask, 2400, 100.0, bin_m=0.5)
    assert energy_from_hypsometry(z, mu, 2000.0, 0.5) == pytest.approx(E, rel=1e-3)
    m = 2400 * 4 * 100 * mask.sum()
    hbar = (surf - 2).mean()  # mass-weighted centre of each column
    assert E == pytest.approx(m * G * (hbar - 2000), rel=1e-9)


def test_delta_energy_eq3():
    assert delta_energy(1e6, 3000, 1.2e6, 3000, 2000) == pytest.approx(G * 0.2e6 * 1000)


def test_changed_volume_signs_and_sigma():
    dz = np.zeros((10, 10)); dz[:5] = -2.0; dz[5:] = 1.0
    cv = changed_volume(dz, np.ones_like(dz, bool), 100.0, 0.5, 10.0)
    assert cv["loss_m3"] == pytest.approx(-10000) and cv["gain_m3"] == pytest.approx(5000)
    assert cv["sigma_m3"] > 0


def test_inconclusive_and_thickness_rules():
    e = energy_interval(Dist("uniform", 1e3, 1e6), Dist("fixed", 2000), Dist("fixed", 100), n=5000,
                        max_interval_ratio=10)
    assert e["inconclusive"]
    assert thickness_scenario_ok([2, 5, 10], 10) and not thickness_scenario_ok([0.5, 5, 20], 10)


def test_chamoli_sensitivity_reproduces_paper():
    r = sensitivity(n=20000)
    assert r["check_against_paper"]["mass_matches_paper"]
    assert r["check_against_paper"]["energy_matches_paper"]
    assert r["corners"]["energy_J"][1] == pytest.approx(5.4e14, rel=0.01)


def test_energy_balance_stops_material():
    s = np.linspace(0, 1000, 101)
    z = 1000 - 0.1 * s
    K = route_energy_balance(s, z, 1e6, loss_per_m=2e6 * G)  # loss exceeds gain -> stays at 0
    assert np.all(K == 0)


# ---------------------------------------------------------------- routing
def test_fill_and_flow_to_outlet():
    y, x = np.mgrid[0:40, 0:40].astype(float)
    dem = 100 + x * 2 + np.abs(y - 20) * 3      # valley along row 20 draining west
    dem[20, 15] -= 30                           # a pit
    filled = fill_depressions(dem)
    assert filled[20, 15] >= dem[20, 14] - 1e-9
    fg = FlowGrid(dem, 10.0)
    path = fg.trace(5 * 40 + 35, 1e6)
    assert path[-1] % 40 == 0                    # reaches the west edge
    assert fg.acc.max() == pytest.approx((np.isfinite(dem)).sum(), rel=0.2)


# ---------------------------------------------------------------- change / metrics / split
def test_transition_roundtrip_and_gate():
    c1 = np.array([[CLASS_INDEX["background"], CLASS_INDEX["water"]]], np.uint8)
    c2 = np.array([[CLASS_INDEX["water"], CLASS_INDEX["water"]]], np.uint8)
    assert decode_transition(transition_code(c1, c2)[0, 0]) == ("background", "water")
    p = np.full((6, 1, 2), 0.1); p[0] = 0.5
    ch = class_change(c1, c2, p, p, min_confidence=0.6)
    assert not ch["changed"].any()              # low confidence suppresses change


def test_metrics():
    y = np.array([0, 0, 1, 1, IGNORE_LABEL])
    m = segmentation_metrics(y, np.array([0, 1, 1, 1, 0]))
    assert m["per_class"]["water"]["recall"] == 1.0 and m["per_class"]["background"]["recall"] == 0.5
    c = change_metrics(np.array([1, 0, 1, 0], bool), np.array([1, 1, 0, 0], bool))
    assert c["f1"] == pytest.approx(0.5) and c["false_alarm_rate"] == pytest.approx(0.5)


def test_geographic_split_has_gap_and_order():
    s = geographic_split((10, 640), block_px=64, test_fraction=0.2, val_fraction=0.1, gap_blocks=1)
    row = s[0]
    assert row[0] == SPLIT_TRAIN and row[-1] == SPLIT_TEST
    assert SPLIT_GAP in row and SPLIT_VAL in row
    first_test = np.argmax(row == SPLIT_TEST)
    assert row[first_test - 1] == SPLIT_GAP


def test_percentile_score_requires_30():
    assert percentile_score(5.0, np.arange(10))["score"] is None
    assert 0 <= percentile_score(5.0, np.arange(40))["score"] <= 1


# ---------------------------------------------------------------- model
def test_unet_shapes_non_multiple():
    net = UNet(10, 6, base=8, depth=3)
    out = net(torch.zeros(2, 10, 50, 37))
    assert out.shape == (2, 6, 50, 37)


# ---------------------------------------------------------------- end to end (baseline, fast)
def test_pipeline_baseline_on_synthetic(tmp_path):
    from mhs.config import load_config
    from mhs.pipeline import run_pipeline
    from mhs.synthetic import generate

    g = generate(tmp_path / "data", seed=3)
    cfg = load_config(g["config"])
    cfg["output"]["figures"] = False
    res = run_pipeline(cfg, None, tmp_path / "run", log=lambda *a: None)
    assert res["n_zones"] >= 3
    types = {z["source_type"] for z in res["zones"]}
    assert "water_storage" in types and "rock_soil_storage" in types
    for f in ("report.md", "zones.geojson", "zones.csv", "manifest.json", "maps/zones.tif"):
        assert (tmp_path / "run" / f).exists()
    gj = json.loads((tmp_path / "run" / "zones.geojson").read_text())
    assert len(gj["features"]) >= res["n_zones"]
    # every energy value is an interval and never a single number
    for z in res["zones"]:
        av = z["physics"]["available"]
        if "energy_J" in av:
            assert av["energy_J"]["low"] <= av["energy_J"]["median"] <= av["energy_J"]["high"]


def test_illumination_and_cast_shadow():
    import datetime as dt
    from mhs.illumination import cast_shadow, cos_incidence, sun_position

    el, az = sun_position(dt.datetime(2021, 1, 16, 5, 19), 30.44, 79.69)
    assert 30 < el < 40 and 140 < az < 170          # low southern winter sun
    # wall to the south of a flat plain casts a shadow northwards
    dem = np.zeros((200, 50), np.float32)
    dem[150:, :] = 500.0                             # 500 m wall from row 150 southwards
    sh = cast_shadow(dem, 10.0, 30.0, 180.0, max_dist_m=3000)
    assert sh[140, 25] and not sh[10, 25]            # 100 m from the wall shaded, 1.4 km away lit
    # north-facing 50 deg slope is self-shadowed under a 34 deg southern sun
    assert cos_incidence(np.array(50.0), np.array(0.0), 34.0, 160.0) < 0.15


def test_preview_and_rasterize_labels(tmp_path):
    from mhs.config import load_config
    from mhs.labels import label_coverage_by_split, rasterize_labels, write_previews
    from mhs.synthetic import X0, Y0, generate

    g = generate(tmp_path / "d", seed=4)
    cfg = load_config(g["config"])
    files = write_previews(cfg, tmp_path / "prev")
    assert any(f.endswith("t1_rgb.tif") for f in files) and any(f.endswith("split.geojson") for f in files)
    def sq(x, y, s=200):  # grid CRS square
        return {"type": "Polygon", "coordinates": [[[X0 + x, Y0 - y], [X0 + x + s, Y0 - y], [X0 + x + s, Y0 - y - s],
                                                     [X0 + x, Y0 - y - s], [X0 + x, Y0 - y]]]}
    gj = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:32644"}},
          "features": [{"type": "Feature", "geometry": sq(100, 3000), "properties": {"class": "background", "date": "both"}},
                       {"type": "Feature", "geometry": sq(3500, 3000), "properties": {"class": 5, "date": "t2"}}]}
    p = tmp_path / "poly.geojson"
    p.write_text(json.dumps(gj))
    res = rasterize_labels(cfg, p, tmp_path / "labels.tif")
    assert res["pixels_per_class"]["t1"]["background"] == 400 and res["pixels_per_class"]["t2"]["debris"] == 400
    assert res["pixels_per_class"]["t1"]["debris"] == 0
    cov = label_coverage_by_split(cfg, tmp_path / "labels.tif")
    assert cov["test"]["debris"] == 400 and cov["train"]["background"] == 800


def test_label_aid_random_and_steep_points(tmp_path):
    from mhs.config import load_config
    from mhs.labels import label_aid
    from mhs.synthetic import generate

    g = generate(tmp_path / "d", seed=5)
    cfg = load_config(g["config"])
    r1 = label_aid(cfg, tmp_path / "aid", n_points=10)
    assert r1["file"] == "test_points.geojson" and r1["points"] > 0
    r2 = label_aid(cfg, tmp_path / "aid", n_points=10, steep_only=True, prefix="R", clean_px=1,
                   avoid_points=tmp_path / "aid" / "test_points.geojson", seed=23)
    assert r2["file"] == "r_points.geojson"
    pts = json.loads((tmp_path / "aid" / "r_points.geojson").read_text())["features"]
    assert all(f["properties"]["id"].startswith("R") for f in pts)
    assert all(f["properties"]["slope_deg"] > 35 for f in pts)
    assert (tmp_path / "aid" / "test_points.geojson").exists()      # first set not overwritten


def test_event_evaluation_on_synthetic(tmp_path):
    from rasterio.warp import transform as tr

    from mhs.config import load_config
    from mhs.event_eval import evaluate_event, event_figure, rank_key, summary_markdown, VARIANTS
    from mhs.io import read_raster
    from mhs.pipeline import run_pipeline
    from mhs.synthetic import generate

    g = generate(tmp_path / "data", seed=3)
    cfg = load_config(g["config"])
    cfg["output"]["figures"] = False
    res = run_pipeline(cfg, None, tmp_path / "run", log=lambda *a: None)
    top = res["zones"][0]
    zones = read_raster(tmp_path / "run" / "maps" / "zones.tif")
    rr, cc = np.nonzero(zones.data[0] == top["id"])
    x, y = zones.xy(int(np.median(rr)), int(np.median(cc)))
    lon, lat = (v[0] for v in tr(zones.crs, "EPSG:4326", [float(x)], [float(y)]))
    ex, ey = top["route"]["end_xy"]
    elon, elat = (v[0] for v in tr(zones.crs, "EPSG:4326", [ex], [ey]))
    cfg["evaluation"]["event"] = {"name": "synthetic", "source": {"lon": lon, "lat": lat},
                                  "checkpoints": [{"name": "route end", "lon": elon, "lat": elat, "published_km": None}],
                                  "damaged_assets": []}
    ev = evaluate_event(cfg, tmp_path / "run", log=lambda *a: None)
    row = ev["ranking"]["500"]
    assert row["zones_matched"] >= 1
    assert row["variants"]["pipeline_as_run"]["best_rank"] == 1
    assert set(VARIANTS) <= set(row["variants"])
    assert ev["pathway_from_documented_source"]["n_checkpoints"] == 1
    assert "Ranking" in summary_markdown(ev)
    event_figure(cfg, ev, tmp_path / "run", tmp_path / "ev.png")
    assert (tmp_path / "ev.png").exists()
    # removing a component never uses it: energy-only differences vanish without energy
    a = {"energy": 1e15, "persistence": "unconfirmed", "signals": 0, "route_ok": True, "n_exp": 0,
         "confidence": 0.9, "n_alt": 0, "area": 1e4}
    b = dict(a, energy=1.0)
    assert rank_key(a, VARIANTS["full"]) < rank_key(b, VARIANTS["full"])
    assert rank_key(a, VARIANTS["no_energy"]) == rank_key(b, VARIANTS["no_energy"])


def test_followup_dem_change_and_change_accuracy(tmp_path):
    import datetime as dt
    from rasterio.warp import transform as tr

    from mhs.change_accuracy import change_accuracy, change_points
    from mhs.climate import window_indicators
    from mhs.config import load_config
    from mhs.dem_change import dem_change, dem_change_figure, summary_markdown
    from mhs.io import read_raster, write_raster
    from mhs.pipeline import run_pipeline
    from mhs.synthetic import generate

    g = generate(tmp_path / "d", seed=3)
    cfg = load_config(g["config"])
    cfg["output"]["figures"] = False
    a, b = read_raster(cfg["inputs"]["dem_t1"]), read_raster(cfg["inputs"]["dem_t2"])
    d = (b.data[0] - a.data[0]).astype("float32")
    write_raster(tmp_path / "diff.tif", a.like(d[None], ["dz"], np.nan), "float32")
    r0, c0 = np.unravel_index(np.nanargmin(d), d.shape)
    x, y = a.xy(r0, c0)
    lon, lat = (v[0] for v in tr(a.crs, "EPSG:4326", [float(x)], [float(y)]))
    cfg["evaluation"]["event"] = {"name": "syn", "source": {"lon": lon, "lat": lat}, "checkpoints": [],
                                  "damaged_assets": [], "published_volume_m3": {"central": 1e5}}
    r = dem_change(cfg, tmp_path / "dc", str(tmp_path / "diff.tif"), log=lambda *x: None)
    assert r["n_zones"] >= 1 and r["ranking"]["500"]["zones_matched"] == 1
    assert r["source_zone"]["loss_m3"] > 0 and r["source_zone"]["energy_J"]["median"] > 0
    dem_change_figure(cfg, r, tmp_path / "dc")
    assert "Source zone" in summary_markdown(r)

    run_pipeline(cfg, None, tmp_path / "run", log=lambda *x: None)
    cp = change_points(cfg, tmp_path / "run", tmp_path / "cp", (4, 4, 4, 6), min_sep_m=30)
    gj = json.loads((tmp_path / "cp" / "change_points.geojson").read_text())
    assert all(set(f["properties"]) == {"id", "label"} for f in gj["features"])   # stratum hidden
    key = json.loads((tmp_path / "cp" / "change_points_key.json").read_text())
    for f in gj["features"]:   # oracle labels = rule-baseline change -> baseline scores perfectly
        k = key["points"][f["properties"]["id"]]
        f["properties"]["label"] = "change" if k["stratum"] in ("both", "rules_only") else "no_change"
    (tmp_path / "cp" / "lab.geojson").write_text(json.dumps(gj))
    ca = change_accuracy(cfg, tmp_path / "cp" / "lab.geojson", tmp_path / "cp" / "change_points_key.json",
                         [tmp_path / "run"], n_boot=50)
    base = ca["maps"]["rule baseline (classes + CVA)"]["estimate"]
    assert cp["points"] > 0 and abs(base["overall_accuracy"] - 1.0) < 1e-9

    dates = [dt.date(1990, 1, 1) + dt.timedelta(days=i) for i in range(11400)]
    rng = np.random.default_rng(0)
    wi = window_indicators(dates, rng.gamma(0.5, 4, len(dates)), rng.normal(0, 5, len(dates)),
                           dt.date(2021, 2, 7), [1, 7], [1991, 2020])
    assert wi["7d"]["n_reference_years"] == 30 and wi["7d"]["precip"]["method"] == "percentile"
