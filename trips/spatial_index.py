"""In-memory store of geocoded fuel stations.

Built ONCE per process from the FuelStation table (the table is the source of
truth; this is a rebuildable cache). Holding the ~6.6k stations as plain objects
plus parallel coordinate arrays lets the corridor query project all of them
against a route in one vectorised pass -- see trips.geo.RouteProjector.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class Station:
    opis_id: int
    name: str
    city: str
    state: str
    price: float
    lat: float
    lng: float


class StationStore:
    def __init__(self, stations: list[Station]):
        self.stations = stations
        # Parallel coordinate arrays, indexed identically to ``stations``.
        self.lats = np.array([s.lat for s in stations], dtype=float)
        self.lngs = np.array([s.lng for s in stations], dtype=float)

    def __len__(self) -> int:
        return len(self.stations)


_index_lock = threading.Lock()
_index: StationStore | None = None


def _load_stations() -> list[Station]:
    from stations.models import FuelStation

    rows = FuelStation.objects.filter(
        is_geocoded=True, latitude__isnull=False, longitude__isnull=False
    ).values_list("opis_id", "name", "city", "state", "retail_price", "latitude", "longitude")
    # retail_price is a DecimalField (the DB is the money source of truth); we
    # cast to float here because the optimizer runs heavy float arithmetic. The
    # reported dollar amounts are quantized back to cents with Decimal in
    # trips.services._serialize, so this cast only affects intermediate
    # precision, never the cents shown to the caller.
    return [
        Station(
            opis_id=r[0],
            name=r[1],
            city=r[2],
            state=r[3],
            price=float(r[4]),
            lat=r[5],
            lng=r[6],
        )
        for r in rows
    ]


def get_index() -> StationStore:
    """Process-wide singleton, built lazily on first use."""
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = StationStore(_load_stations())
    return _index


def reset_index() -> None:
    """Drop the cached store (used by tests after loading fixture data)."""
    global _index
    with _index_lock:
        _index = None
