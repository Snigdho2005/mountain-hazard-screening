"""Asset inventory for exposure from OpenStreetMap (Overpass API).

Independent of any event: everything of the listed kinds inside the case bbox is taken, not only
what a past event damaged, so exposure scoring is not circular. Recorded with the OSM data date.
Roads are left out: as lines they cross almost every corridor and would swamp the count.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINTS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
             "https://overpass.private.coffee/api/interpreter"]


def _query(bbox) -> str:
    w, s, e, n = bbox
    b = f"({s},{w},{n},{e})"
    return f"""[out:json][timeout:120];
(
  node["place"~"^(town|village|hamlet|isolated_dwelling)$"]{b};
  nwr["power"~"^(plant|generator)$"]{b};
  nwr["waterway"~"^(dam|weir)$"]{b};
  way["bridge"]["highway"~"^(trunk|primary|secondary|tertiary|unclassified|track)$"]{b};
);
out center tags;"""


def _type(tags: dict) -> str:
    if "place" in tags:
        return "settlement"
    if tags.get("waterway") in ("dam", "weir") or "power" in tags:
        return "hydropower"
    if "bridge" in tags:
        return "bridge"
    return "other"


def fetch_osm_assets(cfg: dict, out_path: str | Path, timeout: int = 180) -> dict:
    bbox = cfg["case"]["bbox"]
    import time

    data, last = None, None
    for url in [u for _ in range(3) for u in ENDPOINTS]:  # up to 3 rounds over the mirrors
        if data is not None:
            break
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": _query(bbox)}).encode(),
                                         headers={"User-Agent": "mhs-research/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
        except Exception as e:  # busy server: wait, then try the next mirror
            last = e
            print(f"Overpass {url} failed ({type(e).__name__}); retrying ...")
            time.sleep(10)
    if data is None:
        raise RuntimeError(f"Overpass request failed on all endpoints: {last}")
    date = (data.get("osm3s") or {}).get("timestamp_osm_base")
    feats = []
    for el in data.get("elements", []):
        tags = el.get("tags") or {}
        lon, lat = (el["lon"], el["lat"]) if "lon" in el else (el["center"]["lon"], el["center"]["lat"])
        pop = tags.get("population")
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                      "properties": {"id": f"osm_{el['type']}_{el['id']}", "type": _type(tags),
                                     "name": tags.get("name:en") or tags.get("name") or f"unnamed {_type(tags)}",
                                     "population": int(pop) if pop and str(pop).isdigit() else None,
                                     "data_date": date, "source": "OpenStreetMap contributors (ODbL), Overpass API"}})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    counts = {}
    for f in feats:
        counts[f["properties"]["type"]] = counts.get(f["properties"]["type"], 0) + 1
    return {"out": str(out_path), "features": len(feats), "by_type": counts, "osm_data_date": date}
