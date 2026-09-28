"""Is this coordinate inside the USA?

The assignment requires both endpoints to be within the USA. A lat/lng bounding
box cannot express that -- any rectangle covering Alaska, Hawaii and Maine also
covers most of Canada and much of Mexico (Toronto sits comfortably inside one).

So containment is decided in three escalating steps, cheapest first:

1. **Bounding box** -- rejects the obvious outliers for free.
2. **Point-in-polygon** against a vendored US land outline
   (``data/us_boundary.json``, Natural Earth 1:110m, 447 vertices, ~8 KB).
3. **Nearest-city rescue** -- accept anything within
   ``CITY_RESCUE_MILES`` of a known US city centroid. The 1:110m outline is
   coarse and drops small islands (the Florida Keys, Nantucket) and clips some
   coastal cities; those places are all within a mile or two of a city in
   ``data/uscities.csv``, so this recovers them without loosening the polygon.

Note that endpoints supplied as ``"City, ST"`` are resolved *from* that same
city dataset, so they sit at distance zero and can never be falsely rejected.
Only raw coordinate input is subject to the polygon test.

Known limit: twin border cities within a few miles of a US counterpart
(Windsor/Detroit, Tijuana/San Diego, Ciudad Juarez/El Paso, Niagara Falls ON,
Sarnia, Nogales MX) still pass via step 3. No coordinate-only test separates
cities two miles apart; see README for the full discussion.
"""

from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from django.conf import settings

from stations.constants import in_us_bbox
from stations.geocoding import get_default_geocoder

# How close to a known US city centroid a coordinate must be to be accepted
# when it falls outside the coarse outline. Measured against the reference set
# in tests/test_territory.py: every US place tested is within 2.5 mi of a city,
# while the nearest rejected foreign city (Vancouver BC) is 20.6 mi out.
CITY_RESCUE_MILES = 15.0


class _Boundary:
    """Ray-casting point-in-polygon over the vendored US outline."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._rings: list[tuple[np.ndarray, list[np.ndarray]]] | None = None
        self._lock = threading.Lock()

    def _load(self) -> list[tuple[np.ndarray, list[np.ndarray]]]:
        if self._rings is None:
            with self._lock:
                if self._rings is None:
                    with open(self.path, encoding="utf-8") as fh:
                        doc = json.load(fh)
                    self._rings = [
                        (
                            np.asarray(poly[0], dtype=float),  # exterior ring
                            [np.asarray(h, dtype=float) for h in poly[1:]],  # holes
                        )
                        for poly in doc["rings"]
                    ]
        return self._rings

    @staticmethod
    def _ring_contains(ring: np.ndarray, lat: float, lng: float) -> bool:
        """Crossing-number test: count ring edges crossed by a ray heading west."""
        lng1, lat1 = ring[:, 0], ring[:, 1]
        lng2, lat2 = np.roll(lng1, -1), np.roll(lat1, -1)
        # Edges that straddle the point's latitude (each contributes one crossing).
        straddles = (lat1 > lat) != (lat2 > lat)
        if not straddles.any():
            return False
        d_lat = np.where(straddles, lat2 - lat1, 1.0)  # guard the unused divisions
        crossing_lng = lng1 + (lat - lat1) * (lng2 - lng1) / d_lat
        return bool(np.count_nonzero(straddles & (lng < crossing_lng)) % 2)

    def contains(self, lat: float, lng: float) -> bool:
        for exterior, holes in self._load():
            if self._ring_contains(exterior, lat, lng) and not any(
                self._ring_contains(h, lat, lng) for h in holes
            ):
                return True
        return False


@lru_cache(maxsize=1)
def _boundary() -> _Boundary:
    return _Boundary(settings.US_BOUNDARY_PATH)


def in_usa(lat: float, lng: float) -> bool:
    """True if (lat, lng) falls within the USA (50 states + DC)."""
    if not in_us_bbox(lat, lng):
        return False
    if _boundary().contains(lat, lng):
        return True
    return get_default_geocoder().miles_to_nearest_city(lat, lng) <= CITY_RESCUE_MILES
