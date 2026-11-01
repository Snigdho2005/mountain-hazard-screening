"""Physical screening quantities (paper Section 6, Eqs. 1-3, Section 6.2).

Two volumes are always kept separate:
  * changed volume  dV  = integral of dz over the zone (needs two DEMs)
  * available volume V  = material that could take part in a future release
                          (needs independently supported thickness / depth)
Energy is computed only for available volume and is reported as an interval.
Stored energy is NOT a failure probability, runout, impact force or loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import G


# ----------------------------------------------------------------- Eq. (1)
def changed_volume(dz: np.ndarray, mask: np.ndarray, pixel_area: float, dz_sigma: float,
                   corr_range_m: float) -> dict:
    """Net, loss and gain volume from a DEM difference inside a zone, with 1-sigma error.

    Spatially correlated DEM error is handled with the common approximation
    sigma_V = sigma_dz * A / sqrt(n_eff), n_eff = max(1, A / (pi L^2)).
    """
    d = dz[mask]
    d = d[np.isfinite(d)]
    area = d.size * pixel_area
    if d.size == 0:
        return {"available": False}
    n_eff = max(1.0, area / (np.pi * corr_range_m ** 2))
    sigma_v = dz_sigma * area / np.sqrt(n_eff)
    return {
        "available": True,
        "net_m3": float(d.sum() * pixel_area),
        "loss_m3": float(d[d < 0].sum() * pixel_area),
        "gain_m3": float(d[d > 0].sum() * pixel_area),
        "sigma_m3": float(sigma_v),
        "mean_dz_m": float(d.mean()),
        "area_m2": float(area),
        "detectable": bool(abs(d.sum() * pixel_area) > 2 * sigma_v),
    }


# ----------------------------------------------------------------- sampling
@dataclass
class Dist:
    """A declared input distribution: 'uniform' (a=low, b=high), 'normal' (a=mean, b=sd),
    'triangular' (a=low, b=high, c=mode) or 'fixed' (a)."""
    kind: str
    a: float
    b: float | None = None
    c: float | None = None

    def sample(self, n, rng):
        if self.kind == "fixed" or (self.kind == "uniform" and self.a == self.b):
            return np.full(n, float(self.a))
        if self.kind == "uniform":
            return rng.uniform(self.a, self.b, n)
        if self.kind == "normal":
            return rng.normal(self.a, self.b, n)
        if self.kind == "triangular":
            return rng.triangular(self.a, self.c, self.b, n)
        raise ValueError(self.kind)

    def to_dict(self):
        return {k: v for k, v in {"kind": self.kind, "a": self.a, "b": self.b, "c": self.c}.items()
                if v is not None}


def _rank(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    return r


def summarise(samples: np.ndarray, interval: float) -> dict:
    lo, hi = (1 - interval) / 2, 1 - (1 - interval) / 2
    q = np.quantile(samples, [lo, 0.5, hi])
    return {"median": float(q[1]), "low": float(q[0]), "high": float(q[2]), "interval": interval}


def energy_interval(volume: Dist, density: Dist, drop: Dist, n: int = 20000, interval: float = 0.90,
                    threshold: float | None = None, seed: int = 0, max_interval_ratio: float = 10.0) -> dict:
    """Monte Carlo mass and gravitational potential energy E = rho V g dh.

    Returns medians and intervals, exceedance probability of `threshold`, and the
    share of output spread attributable to each input (squared Spearman rank
    correlations, normalised).
    """
    rng = np.random.default_rng(seed)
    if isinstance(volume, np.ndarray):  # pre-sampled (e.g. area x thickness)
        V = np.clip(volume[:n], 0, None)
        n = len(V)
        volume = Dist("samples", float(np.median(V)), float(V.min()), float(V.max()))
    else:
        V = np.clip(volume.sample(n, rng), 0, None)
    rho = np.clip(density.sample(n, rng), 1e-9, None)
    dh = np.clip(drop.sample(n, rng), 0, None)
    m = rho * V
    E = m * G * dh
    rE = _rank(E)
    corr = {}
    for name, x in (("volume", V), ("density", rho), ("drop", dh)):
        corr[name] = 0.0 if np.ptp(x) == 0 else float(np.corrcoef(_rank(x), rE)[0, 1] ** 2)
    tot = sum(corr.values()) or 1.0
    es = summarise(E, interval)
    ratio = es["high"] / es["low"] if es["low"] > 0 else np.inf
    out = {
        "mass_kg": summarise(m, interval),
        "energy_J": es,
        "interval_ratio": float(ratio),
        "inconclusive": bool(ratio > max_interval_ratio),
        "spread_share": {k: v / tot for k, v in corr.items()},
        "dominant_input": max(corr, key=corr.get),
        "inputs": {"volume_m3": volume.to_dict(), "density_kg_m3": density.to_dict(),
                   "drop_m": drop.to_dict()},
        "n_samples": n,
        "seed": seed,
    }
    if threshold is not None:
        out["p_exceed_threshold"] = float((E > threshold).mean())
        out["threshold_J"] = threshold
    return out


def first_order_relative_error(sig_rho_rel, sig_v_rel, sig_dh_rel) -> float:
    """(sigma_E/E)^2 = (sigma_rho/rho)^2 + (sigma_V/V)^2 + (sigma_dh/dh)^2, independent small errors."""
    return float(np.sqrt(sig_rho_rel ** 2 + sig_v_rel ** 2 + sig_dh_rel ** 2))


def corner_bounds(v, rho, dh) -> dict:
    """Deterministic min / mid / max products for (low, mid, high) tuples."""
    return {
        "mass_kg": (v[0] * rho[0], v[1] * rho[1], v[2] * rho[2]),
        "energy_J": (v[0] * rho[0] * G * dh[0], v[1] * rho[1] * G * dh[1], v[2] * rho[2] * G * dh[2]),
    }


# ----------------------------------------------------------------- Eq. (2)
def column_energy(surface: np.ndarray, thickness: np.ndarray, mask: np.ndarray, density: float,
                  pixel_area: float, h_ref: float) -> float:
    """E = sum over columns of rho g A * integral_{zb}^{zt} (z - h_r) dz (distributed, not point mass)."""
    zt = surface[mask]
    zb = zt - thickness[mask]
    per = density * G * pixel_area * 0.5 * ((zt - h_ref) ** 2 - (zb - h_ref) ** 2)
    return float(np.nansum(per))


def mass_hypsometry(surface, thickness, mask, density, pixel_area, bin_m=10.0):
    """mu(z): available mass per unit elevation (kg/m), from stacked columns.

    Returns (bin centres, mu). The integral g * sum (z - h_r) mu(z) dz reproduces column_energy.
    """
    zt = surface[mask].astype(float)
    t = thickness[mask].astype(float)
    zb = zt - t
    lo, hi = np.floor(np.nanmin(zb) / bin_m) * bin_m, np.ceil(np.nanmax(zt) / bin_m) * bin_m
    edges = np.arange(lo, hi + bin_m, bin_m)
    mu = np.zeros(len(edges) - 1)
    for a, b in zip(zb, zt):
        ov = np.clip(np.minimum(edges[1:], b) - np.maximum(edges[:-1], a), 0, None)
        mu += ov * density * pixel_area
    return 0.5 * (edges[1:] + edges[:-1]), mu / bin_m


def energy_from_hypsometry(z, mu, h_ref, bin_m) -> float:
    m = z >= h_ref
    return float(np.sum(G * (z[m] - h_ref) * mu[m] * bin_m))


# ----------------------------------------------------------------- Eq. (3)
def delta_energy(m1, h1, m2, h2, h_ref) -> float:
    """dE = g [m2 (h2 - h_r) - m1 (h1 - h_r)]; only meaningful with comparable methods on both dates."""
    return float(G * (m2 * (h2 - h_ref) - m1 * (h1 - h_ref)))


def energy_rate_split(m, h, dm_dt, dh_dt, h_ref) -> dict:
    """dE/dt = g[(h - h_r) dm/dt + m dh/dt]: mass growth vs elevation change terms."""
    return {"mass_term": G * (h - h_ref) * dm_dt, "elevation_term": G * m * dh_dt}


# ----------------------------------------------------------------- Section 6.2
def available_energy_downstream(z, mu, activation, h_d, bin_m) -> float:
    """E_avail,d = integral_{h_d}^{H} g (z - h_d) a(z) mu(z) dz for an activation profile a(z) in [0, 1]."""
    a = np.clip(np.broadcast_to(activation, z.shape), 0, 1)
    m = z >= h_d
    return float(np.sum(G * (z[m] - h_d) * a[m] * mu[m] * bin_m))


def route_energy_balance(s, z, M, loss_per_m, K0=0.0) -> np.ndarray:
    """Integrate dK/ds = -M g dz/ds - l(s) along a route (scenario tool, not a runout model).

    `M` and `loss_per_m` are arrays or scalars on the same stations as `s`.
    Kinetic energy is floored at zero (material stops).
    """
    s, z = np.asarray(s, float), np.asarray(z, float)
    M = np.broadcast_to(np.asarray(M, float), s.shape)
    L = np.broadcast_to(np.asarray(loss_per_m, float), s.shape)
    K = np.zeros_like(s)
    K[0] = K0
    for i in range(1, len(s)):
        ds = s[i] - s[i - 1]
        Mm = 0.5 * (M[i] + M[i - 1])
        dK = -Mm * G * (z[i] - z[i - 1]) - 0.5 * (L[i] + L[i - 1]) * ds
        K[i] = max(0.0, K[i - 1] + dK)
        if K[i] == 0.0:
            break
    return K


def mass_interval(volume: Dist, density: Dist, n=20000, interval=0.90, seed=0) -> dict:
    """Mass interval for a CHANGED volume (no energy is attached to changed volume)."""
    rng = np.random.default_rng(seed)
    m = np.clip(volume.sample(n, rng), 0, None) * np.clip(density.sample(n, rng), 0, None)
    return {"mass_kg": summarise(m, interval), "inputs": {"volume_m3": volume.to_dict(),
                                                          "density_kg_m3": density.to_dict()}}


def area_thickness_samples(area_m2: float, area_rel_sigma: float, scenario: list[float], n: int,
                           seed: int) -> np.ndarray:
    """Available volume samples V = A x t with A ~ N(A, rel*A) and t ~ triangular(low, mid, high)."""
    rng = np.random.default_rng(seed + 1)
    A = np.clip(rng.normal(area_m2, area_rel_sigma * area_m2, n), 0, None)
    lo, mid, hi = scenario
    t = rng.triangular(lo, mid, hi, n)
    return A * t


# ----------------------------------------------------------------- scenarios
def thickness_scenario_ok(scen: list[float], max_ratio: float) -> bool:
    lo, _, hi = scen
    return lo > 0 and hi / lo <= max_ratio
