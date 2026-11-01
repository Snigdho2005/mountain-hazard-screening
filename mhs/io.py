"""Raster input/output on a shared grid."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine


@dataclass
class Raster:
    """A (bands, rows, cols) array with georeferencing and band names."""

    data: np.ndarray
    transform: Affine
    crs: object
    names: list[str] = field(default_factory=list)
    nodata: float | None = None
    tags: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.data.ndim == 2:
            self.data = self.data[None]
        if not self.names:
            self.names = [f"b{i + 1}" for i in range(self.data.shape[0])]

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[1], self.data.shape[2]

    @property
    def res(self) -> float:
        return float(abs(self.transform.a))

    @property
    def pixel_area(self) -> float:
        return float(abs(self.transform.a * self.transform.e))

    def band(self, name: str) -> np.ndarray:
        return self.data[self.names.index(name)]

    def has(self, name: str) -> bool:
        return name in self.names

    def xy(self, row, col):
        """Map coordinates of pixel centres."""
        row = np.asarray(row) + 0.5
        col = np.asarray(col) + 0.5
        x = self.transform.c + col * self.transform.a + row * self.transform.b
        y = self.transform.f + col * self.transform.d + row * self.transform.e
        return x, y

    def rowcol(self, x, y):
        inv = ~self.transform
        col, row = inv * (np.asarray(x), np.asarray(y))
        return np.floor(row).astype(int), np.floor(col).astype(int)

    def like(self, data: np.ndarray, names: list[str] | None = None, nodata=None) -> "Raster":
        return Raster(data, self.transform, self.crs, names or [], nodata)


def read_raster(path: str | Path) -> Raster:
    with rasterio.open(path) as src:
        data = src.read()
        names = [d if d else f"b{i + 1}" for i, d in enumerate(src.descriptions)]
        tags = src.tags()
        r = Raster(data, src.transform, src.crs, names, src.nodata, tags)
    return r


def write_raster(path: str | Path, raster: Raster, dtype=None, compress="deflate") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = raster.data
    dtype = dtype or data.dtype
    profile = dict(
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=np.dtype(dtype).name,
        crs=raster.crs,
        transform=raster.transform,
        compress=compress,
        tiled=data.shape[1] >= 256 and data.shape[2] >= 256,
    )
    if raster.nodata is not None:
        profile["nodata"] = raster.nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype))
        for i, n in enumerate(raster.names, start=1):
            dst.set_band_description(i, n)
        if raster.tags:
            dst.update_tags(**{k: (json.dumps(v) if not isinstance(v, str) else v) for k, v in raster.tags.items()})
    return path


def same_grid(a: Raster, b: Raster, tol: float = 1e-6) -> bool:
    return (
        a.shape == b.shape
        and str(a.crs) == str(b.crs)
        and all(abs(x - y) <= tol for x, y in zip(tuple(a.transform)[:6], tuple(b.transform)[:6]))
    )


def regrid(src_path: str | Path, like: Raster, resampling: str = "bilinear") -> Raster:
    """Resample any raster onto the grid of `like` (e.g. a 2 m DEM onto the 10 m image grid)."""
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    with rasterio.open(src_path) as src, WarpedVRT(
            src, crs=like.crs, transform=like.transform, width=like.shape[1], height=like.shape[0],
            resampling=Resampling[resampling], nodata=np.nan, dtype="float32") as vrt:
        data = vrt.read()
        names = [d or f"b{i + 1}" for i, d in enumerate(src.descriptions)]
        return Raster(data, like.transform, like.crs, names, np.nan, dict(src.tags()))
