"""DEM-derived downslope routes and screening corridors (paper Section 7.1, stage one).

This is a first-pass connection map. It does NOT estimate runout distance,
velocity, flow depth, spreading or inundation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.morphology import reconstruction

OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def fill_depressions(dem: np.ndarray) -> np.ndarray:
    """Fill sinks by morphological reconstruction; NaN cells and the border act as outlets."""
    z = dem.astype(np.float64).copy()
    nan = ~np.isfinite(z)
    zmin = np.nanmin(z) - 1.0
    z[nan] = zmin
    seed = np.full_like(z, np.nanmax(z))
    seed[0, :], seed[-1, :], seed[:, 0], seed[:, -1] = z[0, :], z[-1, :], z[:, 0], z[:, -1]
    seed[nan] = zmin
    filled = reconstruction(seed, z, method="erosion")
    filled[nan] = np.nan
    return filled


def _shifted(a, dr, dc, fill):
    out = np.full_like(a, fill)
    h, w = a.shape
    if abs(dr) >= h or abs(dc) >= w:
        return out
    rs = slice(max(dr, 0), h + min(dr, 0))
    rd = slice(max(-dr, 0), h + min(-dr, 0))
    cs = slice(max(dc, 0), w + min(dc, 0))
    cd = slice(max(-dc, 0), w + min(-dc, 0))
    out[rd, cd] = a[rs, cs]
    return out  # out[r, c] = a[r + dr, c + dc]


@dataclass
class FlowGrid:
    dem: np.ndarray
    res: float
    filled: np.ndarray = field(init=False)
    down: np.ndarray = field(init=False)      # flat index of receiver, -1 = outlet
    acc: np.ndarray = field(init=False)       # contributing cells (including self)
    fill_depth: np.ndarray = field(init=False)

    def __post_init__(self):
        h, w = self.dem.shape
        self.filled = fill_depressions(self.dem)
        self.fill_depth = np.nan_to_num(self.filled - self.dem, nan=0.0)
        z = np.where(np.isfinite(self.filled), self.filled, -np.inf)
        best = np.zeros((h, w))
        dirn = np.full((h, w), -1, np.int8)
        for k, (dr, dc) in enumerate(OFFSETS):
            nb = _shifted(z, dr, dc, -np.inf)
            nb_inside = _shifted(np.ones((h, w), bool), dr, dc, False)
            dist = self.res * (np.sqrt(2) if dr and dc else 1.0)
            drop = (z - nb) / dist
            drop[~nb_inside] = -np.inf
            better = drop > best + 1e-12
            best[better] = drop[better]
            dirn[better] = k
        valid = np.isfinite(self.filled)
        edge = np.zeros((h, w), bool)
        edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
        resolved = valid & ((dirn >= 0) | edge)
        dirn[edge & (dirn < 0)] = -1
        # resolve flats: point to an equal-height neighbour that is already resolved
        flat_dist = np.zeros((h, w), np.int32)
        it = 0
        while True:
            it += 1
            todo = valid & ~resolved
            if not todo.any():
                break
            progressed = False
            for k, (dr, dc) in enumerate(OFFSETS):
                nb_res = _shifted(resolved, dr, dc, False)
                nb_z = _shifted(z, dr, dc, -np.inf)
                cand = todo & nb_res & (np.abs(nb_z - z) < 1e-6) & (dirn < 0)
                if cand.any():
                    dirn[cand] = k
                    flat_dist[cand] = it
                    progressed = True
            resolved |= valid & (dirn >= 0)
            if not progressed:
                break  # isolated cells (should not occur after filling)
        rr, cc = np.indices((h, w))
        down = np.full(h * w, -1, np.int64)
        for k, (dr, dc) in enumerate(OFFSETS):
            m = (dirn == k).ravel()
            down[m] = ((rr + dr) * w + (cc + dc)).ravel()[m]
        self.down = down
        # accumulation: upstream before downstream (higher first, then farther from flat outlet)
        order = np.lexsort((-flat_dist.ravel(), -np.nan_to_num(self.filled, nan=-1e9).ravel()))
        acc = valid.ravel().astype(np.float64)
        dn = down
        for i in order:
            j = dn[i]
            if j >= 0:
                acc[j] += acc[i]
        self.acc = acc.reshape(h, w)

    @property
    def acc_km2(self) -> np.ndarray:
        return self.acc * self.res * self.res / 1e6

    def trace(self, start: int, max_len_m: float) -> np.ndarray:
        path, length, i, seen = [start], 0.0, start, {start}
        w = self.dem.shape[1]
        while True:
            j = int(self.down[i])
            if j < 0 or j in seen:
                break
            dr, dc = abs(j // w - i // w), abs(j % w - i % w)
            length += self.res * (np.sqrt(2) if dr and dc else 1.0)
            if length > max_len_m:
                break
            path.append(j)
            seen.add(j)
            i = j
        return np.array(path, np.int64)


@dataclass
class Route:
    path: np.ndarray            # flat indices from source outlet downstream
    distance_m: np.ndarray      # cumulative distance along path
    elevation_m: np.ndarray     # original DEM elevation along path
    reference_index: int        # index in path of first valley-floor point
    reference_elevation_m: float
    corridor: np.ndarray        # boolean (H, W)
    qc_flags: list[str]

    @property
    def length_m(self) -> float:
        return float(self.distance_m[-1]) if len(self.distance_m) else 0.0


def route_from_zone(fg: FlowGrid, zone_mask: np.ndarray, slope_deg: np.ndarray, rcfg: dict) -> Route:
    h, w = zone_mask.shape
    flat = np.flatnonzero(zone_mask.ravel())
    start = int(flat[np.argmax(fg.acc.ravel()[flat])])  # zone outlet = highest accumulation cell
    path = fg.trace(start, rcfg["max_route_km"] * 1000.0)
    r, c = path // w, path % w
    step = np.hypot(np.diff(r), np.diff(c)) * fg.res
    dist = np.concatenate([[0.0], np.cumsum(step)])
    elev = fg.dem.ravel()[path]
    acc_km2 = fg.acc_km2.ravel()[path]
    valley = np.flatnonzero((acc_km2 >= rcfg["valley_accum_km2"]) &
                            (np.nan_to_num(slope_deg.ravel()[path], nan=90) <= rcfg["valley_slope_deg"]) &
                            ~zone_mask.ravel()[path])
    ref_i = int(valley[0]) if len(valley) else len(path) - 1
    flags = []
    if not len(valley):
        flags.append("no_valley_floor_found_reference_is_route_end")
    fd = float(fg.fill_depth.ravel()[path].max(initial=0))
    if fd > rcfg["fill_depth_flag_m"]:
        flags.append(f"route_crosses_{fd:.0f}m_filled_depression_check_lake_dam_or_dem_artefact")
    if np.isnan(elev).any():
        flags.append("route_crosses_dem_nodata")
    if len(path) < 3:
        flags.append("route_too_short")
    on_path = np.zeros(h * w, bool)
    on_path[path] = True
    on_path = on_path.reshape(h, w)
    corridor = ndimage.distance_transform_edt(~on_path) * fg.res <= rcfg["corridor_halfwidth_m"]
    return Route(path, dist, elev, ref_i, float(elev[ref_i]), corridor, flags)


def linked_sources(routes: dict[int, Route], zone_labels: np.ndarray) -> dict[int, list[int]]:
    """Zones whose downslope route passes through another zone (paper Section 5, 'linked sources')."""
    flat = zone_labels.ravel()
    out = {}
    for zid, rt in routes.items():
        hit = set(int(v) for v in np.unique(flat[rt.path]) if v > 0 and v != zid)
        if hit:
            out[zid] = sorted(hit)
    return out
