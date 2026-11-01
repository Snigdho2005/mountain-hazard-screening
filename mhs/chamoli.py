"""Retrospective demonstration: 2021 Chamoli rock-ice avalanche (paper Section 9).

Two modes:
  * `sensitivity()` reproduces the paper's published-value sensitivity calculation
    (V 24-30e6 m3, rho 2200-2600 kg/m3, drop 700-1000 m) as corner bounds and as
    a Monte Carlo interval with input-contribution shares.
  * `from_dems()` recomputes the collapse volume from the Shean et al. (2021)
    pre/post DEM composites (Zenodo 10.5281/zenodo.4558692) inside a source
    polygon, with spatially correlated error propagation, and then the energy
    with a drop measured from the DEM to a declared valley reference point.
"""
from __future__ import annotations

import json

import numpy as np

from . import G
from .physics import Dist, changed_volume, corner_bounds, energy_interval

PAPER = {
    "volume_m3": (24e6, 27e6, 30e6),
    "density_kg_m3": (2200, 2400, 2600),
    "drop_m": (700, 850, 1000),
    "reported_energy_J": (3.6e14, 5.4e14, 7.7e14),
    "reported_mass_kg": (5.28e10, 6.48e10, 7.80e10),
}


def sensitivity(n=200000, seed=2021, interval=0.90) -> dict:
    v, rho, dh = PAPER["volume_m3"], PAPER["density_kg_m3"], PAPER["drop_m"]
    corners = corner_bounds(v, rho, dh)
    mc = energy_interval(Dist("uniform", v[0], v[2]), Dist("uniform", rho[0], rho[2]),
                         Dist("uniform", dh[0], dh[2]), n=n, interval=interval, seed=seed,
                         threshold=5.0e14)
    check = {
        "mass_matches_paper": bool(np.allclose(corners["mass_kg"], PAPER["reported_mass_kg"], rtol=0.01)),
        "energy_matches_paper": bool(np.allclose(corners["energy_J"], PAPER["reported_energy_J"], rtol=0.03)),
    }
    return {"g": G, "corners": corners, "monte_carlo": mc, "check_against_paper": check,
            "note": "Sensitivity ranges, not measured event-energy bounds; not impact or delivered energy."}


def from_dems(pre_dem: str, post_dem: str, polygon_geojson: str, reference_xy: tuple[float, float] | None = None,
              reference_elevation: float | None = None, dz_sigma=2.0, corr_range=150.0,
              density=(2200, 2600), n=200000, seed=2021) -> dict:
    """Recompute volume and energy from the original elevation files.

    The post-event DEM is resampled onto the pre-event grid. The source is the
    part of the polygon with negative dz (lowering). The drop is from the
    mass-weighted mean elevation of the lost column (pre DEM) to the reference.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import rasterize
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_geom

    with rasterio.open(pre_dem) as a:
        z1 = a.read(1, masked=True).filled(np.nan).astype(np.float64)
        tr, crs, shape = a.transform, a.crs, a.shape
        with rasterio.open(post_dem) as b, WarpedVRT(b, crs=crs, transform=tr, width=shape[1], height=shape[0],
                                                      resampling=Resampling.bilinear) as vb:
            z2 = vb.read(1, masked=True).filled(np.nan).astype(np.float64)
    with open(polygon_geojson) as f:
        gj = json.load(f)
    src = (gj.get("crs") or {}).get("properties", {}).get("name", "EPSG:4326")
    geoms = [transform_geom(src, crs, ft["geometry"]) for ft in gj["features"]]
    poly = rasterize([(g, 1) for g in geoms], out_shape=shape, transform=tr, dtype="uint8").astype(bool)
    dz = z2 - z1
    loss = poly & np.isfinite(dz) & (dz < 0)
    px_area = abs(tr.a * tr.e)
    cv = changed_volume(dz, loss, px_area, dz_sigma, corr_range)
    w = -dz[loss]
    zc = float(np.sum((z1[loss] - 0.5 * w) * w) / np.sum(w))  # centre of the lost column
    if reference_elevation is None:
        if reference_xy is None:
            raise ValueError("give reference_elevation or reference_xy (valley-floor point)")
        r, c = rasterio.transform.rowcol(tr, *reference_xy)
        reference_elevation = float(z1[r, c])
    drop = zc - reference_elevation
    vol = abs(cv["loss_m3"])
    e = energy_interval(Dist("normal", vol, cv["sigma_m3"]), Dist("uniform", *density),
                        Dist("normal", drop, dz_sigma * np.sqrt(2)), n=n, seed=seed)
    return {"changed_volume": cv, "source_centre_elevation_m": zc,
            "reference_elevation_m": reference_elevation, "drop_m": drop, "energy": e}
