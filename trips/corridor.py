"""Select fuel stations within a corridor of the route and place each at a
distance-from-start "mile marker".

All ~6.6k stations are projected onto the route in one vectorised pass
(trips.geo.RouteProjector), which both prefilters to the corridor and yields
each station's exact perpendicular distance and mile marker. Mile markers are
scaled to the provider's reported road distance so the downstream 500-mile
range checks use consistent units.
"""

from __future__ import annotations

from dataclasses import dataclass

from django.conf import settings

from trips.geo import RouteProjector, cumulative_miles, haversine_miles
from trips.spatial_index import Station, get_index


@dataclass
class Candidate:
    station: Station
    mile_marker: float  # distance from route start, miles (scaled to road distance)
    detour_miles: float  # perpendicular distance from the route, miles


def _densify(points: list[tuple[float, float]], spacing: float) -> list[tuple[float, float]]:
    """Insert intermediate points so consecutive vertices are <= spacing apart.

    Real provider geometry is already far denser than this (OSRM returns a
    vertex every ~0.1 mi), so this is normally a no-op. It exists to bound the
    longest segment for a provider that returns coarse geometry, which in turn
    bounds RouteProjector's search radius -- see ``project_within``.
    """
    if len(points) < 2:
        return list(points)
    out = [points[0]]
    for (lat1, lng1), (lat2, lng2) in zip(points, points[1:], strict=False):
        dist = haversine_miles(lat1, lng1, lat2, lng2)
        if dist > spacing:
            steps = int(dist // spacing)
            for k in range(1, steps + 1):
                frac = k * spacing / dist
                if frac >= 1.0:
                    break
                out.append((lat1 + (lat2 - lat1) * frac, lng1 + (lng2 - lng1) * frac))
        out.append((lat2, lng2))
    return out


def find_candidates(route, buffer_miles: float | None = None) -> tuple[list[Candidate], float]:
    """Return (candidates ordered by mile marker, total_distance_miles)."""
    if buffer_miles is None:
        buffer_miles = settings.CORRIDOR_BUFFER_MILES

    points = _densify(route.points, spacing=buffer_miles)
    cum = cumulative_miles(points)
    polyline_len = cum[-1] if cum else 0.0
    total_miles = route.total_distance_miles
    # Scale haversine polyline positions to the provider's road distance.
    scale = (total_miles / polyline_len) if polyline_len > 0 else 1.0

    index = get_index()
    projector = RouteProjector(points, cum)
    idxs, markers, perps = projector.project_within(index.lats, index.lngs, buffer_miles)

    candidates = [
        Candidate(
            station=index.stations[int(i)],
            mile_marker=float(marker) * scale,
            detour_miles=float(perp),
        )
        for i, marker, perp in zip(idxs, markers, perps, strict=True)
        if perp <= buffer_miles
    ]

    candidates.sort(key=lambda c: c.mile_marker)
    return candidates, total_miles
