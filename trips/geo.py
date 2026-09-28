"""Pure geographic math on (lat, lng) coordinates, in miles.

We deliberately use haversine arc-length rather than planar projection: 1 degree
of longitude varies from ~69 mi at the equator to ~49 mi at 49 N, so projecting
raw lon/lat onto a polyline with planar math distorts distances by 20-30% on a
transcontinental route -- which would corrupt the hard 500-mile range constraint.

``project_to_route`` is the readable scalar reference implementation;
``RouteProjector`` is the vectorised one used in the request path. The test
suite asserts the two agree, so the reference stays the specification.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

EARTH_RADIUS_MILES = 3958.7613
METERS_PER_MILE = 1609.344

# Miles per degree of latitude. Longitude is scaled by cos(latitude) locally.
MILES_PER_DEGREE_LAT = 69.0


def unit_xyz(lats, lngs) -> np.ndarray:
    """Embed (lat, lng) arrays as points on the unit sphere.

    A Euclidean (chord) distance in this space is a monotone function of the
    great-circle distance, which is what lets a KD-tree answer "everything
    within R miles of this point" -- see ``chord_for_miles``.
    """
    rlat = np.radians(np.asarray(lats, dtype=float))
    rlng = np.radians(np.asarray(lngs, dtype=float))
    cos_lat = np.cos(rlat)
    return np.column_stack([cos_lat * np.cos(rlng), cos_lat * np.sin(rlng), np.sin(rlat)])


def chord_for_miles(miles: float) -> float:
    """Euclidean chord length on the unit sphere for a given surface distance."""
    return 2.0 * math.sin(miles / EARTH_RADIUS_MILES / 2.0)


def haversine_miles(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance between two points in miles."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * math.asin(math.sqrt(a))


def cumulative_miles(points: list[tuple[float, float]]) -> list[float]:
    """Cumulative arc-length (miles) at each vertex of a [(lat, lng), ...] path.

    Returns a list the same length as ``points`` where index 0 is 0.0 and the
    last entry is the total polyline length.
    """
    cum = [0.0]
    for (lat1, lng1), (lat2, lng2) in zip(points, points[1:], strict=False):
        cum.append(cum[-1] + haversine_miles(lat1, lng1, lat2, lng2))
    return cum


def _project_point_on_segment(
    plat: float,
    plng: float,
    alat: float,
    alng: float,
    blat: float,
    blng: float,
) -> tuple[float, float, float]:
    """Project point P onto segment A->B using a local equirectangular plane
    centred at A (accurate for the short segments of a road polyline).

    Returns (frac, perp_miles, along_miles): the clamped fraction along the
    segment, the perpendicular distance from P to the segment in miles, and the
    along-segment distance from A to the projection in miles.
    """
    deg_lat = MILES_PER_DEGREE_LAT
    deg_lng = MILES_PER_DEGREE_LAT * math.cos(math.radians(alat))

    # A is the local origin (0, 0); B and P are expressed relative to it.
    bx, by = (blng - alng) * deg_lng, (blat - alat) * deg_lat
    px, py = (plng - alng) * deg_lng, (plat - alat) * deg_lat

    seg_len_sq = bx * bx + by * by
    if seg_len_sq == 0.0:
        frac = 0.0
    else:
        frac = (px * bx + py * by) / seg_len_sq
        frac = max(0.0, min(1.0, frac))

    proj_x, proj_y = frac * bx, frac * by
    perp = math.hypot(px - proj_x, py - proj_y)
    along = math.hypot(proj_x, proj_y)
    return frac, perp, along


def project_to_route(
    plat: float,
    plng: float,
    route: list[tuple[float, float]],
    cum: list[float],
) -> tuple[float, float]:
    """Project a point onto a route polyline.

    ``route`` is [(lat, lng), ...]; ``cum`` is the matching cumulative_miles list.
    Returns (mile_marker, perp_miles): distance from the route start to the
    nearest point on the route, and the perpendicular detour distance in miles.
    """
    best_perp = float("inf")
    best_marker = 0.0
    for i in range(len(route) - 1):
        alat, alng = route[i]
        blat, blng = route[i + 1]
        frac, perp, along = _project_point_on_segment(plat, plng, alat, alng, blat, blng)
        if perp < best_perp:
            best_perp = perp
            best_marker = cum[i] + along
    return best_marker, best_perp


class RouteProjector:
    """Projects many points onto one route polyline, fast.

    ``project_to_route`` above scans every segment for every point. On a
    transcontinental route that is ~35k segments x ~400 candidate stations =
    ~15M scalar projections, which dominated request latency (5.6 s of a 7 s
    response). This class fixes that with two changes:

    1. **Precompute per-segment geometry once** (origins, offsets, squared
       lengths, cumulative distance) instead of recomputing it -- including a
       ``cos()`` per segment -- for every point.
    2. **Only look at segments that could possibly win.** A KD-tree over the
       route vertices answers "which bits of the route run near this point",
       so each point is projected against a few hundred nearby segments rather
       than all of them.

    Together these take the same work from ~5.6 s to ~30 ms, bit-for-bit
    identical (see ``tests/test_geo_projection.py``).
    """

    def __init__(self, points: list[tuple[float, float]], cum: list[float] | None = None):
        self.points = points
        self.cum = np.asarray(cum if cum is not None else cumulative_miles(points), dtype=float)

        arr = np.asarray(points, dtype=float).reshape(-1, 2)
        # Segment i runs from vertex i to vertex i+1.
        a, b = arr[:-1], arr[1:]
        self._a_lat, self._a_lng = a[:, 0], a[:, 1]
        deg_lng = MILES_PER_DEGREE_LAT * np.cos(np.radians(self._a_lat))
        # Segment vector in the local equirectangular frame centred on A.
        self._bx = (b[:, 1] - a[:, 1]) * deg_lng
        self._by = (b[:, 0] - a[:, 0]) * MILES_PER_DEGREE_LAT
        self._deg_lng = deg_lng
        self._len_sq = self._bx * self._bx + self._by * self._by
        self._cum_start = self.cum[:-1]

        # Longest segment, which sets the safe KD-tree search radius below.
        self.max_segment_miles = float(np.max(np.diff(self.cum))) if len(self.cum) > 1 else 0.0
        self._tree = cKDTree(unit_xyz(arr[:, 0], arr[:, 1])) if len(arr) else None

    @property
    def num_segments(self) -> int:
        return len(self._len_sq)

    def project_within(self, lats, lngs, buffer_miles: float):
        """Project the points that plausibly lie within ``buffer_miles`` of the route.

        Returns ``(indices, mile_markers, perp_miles)`` -- parallel arrays, where
        ``indices`` selects the input points that were actually projected.

        Points are searched against route vertices within
        ``buffer_miles + max_segment_miles``. That radius is provably safe: if a
        point lies within ``buffer_miles`` of some segment of length L, its
        distance to the nearer end of that segment is at most
        ``sqrt(buffer^2 + (L/2)^2) <= buffer + L/2``, so the winning segment is
        always inside the window. Restricting the search can therefore only
        *overestimate* the perpendicular distance of points that were already
        outside the buffer -- which the caller discards anyway.
        """
        empty = (np.empty(0, dtype=int), np.empty(0), np.empty(0))
        lats = np.asarray(lats, dtype=float)
        if self._tree is None or self.num_segments == 0 or lats.size == 0:
            return empty

        radius = chord_for_miles(buffer_miles + self.max_segment_miles)
        windows = self._tree.query_ball_point(unit_xyz(lats, lngs), r=radius)

        idx_out, marker_out, perp_out = [], [], []
        last_seg = self.num_segments - 1
        for i, vertices in enumerate(windows):
            if len(vertices) == 0:
                continue
            v = np.asarray(vertices, dtype=int)
            # A vertex v touches segments v-1 and v.
            segs = np.unique(np.clip(np.concatenate((v - 1, v)), 0, last_seg))
            marker, perp = self._project_one(lats[i], lngs[i], segs)
            idx_out.append(i)
            marker_out.append(marker)
            perp_out.append(perp)

        if not idx_out:
            return empty
        return (
            np.asarray(idx_out, dtype=int),
            np.asarray(marker_out, dtype=float),
            np.asarray(perp_out, dtype=float),
        )

    def _project_one(self, plat: float, plng: float, segs: np.ndarray) -> tuple[float, float]:
        """Vectorised equivalent of ``_project_point_on_segment`` over ``segs``."""
        bx, by = self._bx[segs], self._by[segs]
        len_sq = self._len_sq[segs]
        # P relative to each segment's origin A, in that segment's local frame.
        px = (plng - self._a_lng[segs]) * self._deg_lng[segs]
        py = (plat - self._a_lat[segs]) * MILES_PER_DEGREE_LAT

        # Clamped projection parameter. Zero-length segments (duplicate route
        # vertices) collapse to their origin, matching the scalar version.
        nonzero = len_sq > 0.0
        frac = np.zeros_like(len_sq)
        np.divide(px * bx + py * by, len_sq, out=frac, where=nonzero)
        np.clip(frac, 0.0, 1.0, out=frac)

        proj_x, proj_y = frac * bx, frac * by
        perp = np.hypot(px - proj_x, py - proj_y)

        best = int(np.argmin(perp))
        along = math.hypot(proj_x[best], proj_y[best])
        return float(self._cum_start[segs[best]] + along), float(perp[best])
