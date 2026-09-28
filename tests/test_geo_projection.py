"""RouteProjector must agree exactly with the scalar reference implementation.

``project_to_route`` is the readable specification; ``RouteProjector`` is the
vectorised, KD-tree-windowed version that actually runs in the request path
(~100x faster on a transcontinental route). These tests pin the two together,
including the two properties the windowing optimisation depends on:

  * for any point inside the corridor buffer, the answer is *identical*, and
  * no point inside the buffer is ever dropped by the window.
"""

import math
import random

import numpy as np
import pytest

from trips.geo import RouteProjector, cumulative_miles, project_to_route

BUFFER = 7.0


def _reference(points, lats, lngs):
    cum = cumulative_miles(points)
    return [project_to_route(la, lo, points, cum) for la, lo in zip(lats, lngs, strict=True)]


def _assert_matches_reference(points, lats, lngs, buffer_miles=BUFFER):
    """Every in-buffer point must be projected, and projected identically."""
    proj = RouteProjector(points)
    idxs, markers, perps = proj.project_within(lats, lngs, buffer_miles)
    got = {int(i): (float(m), float(p)) for i, m, p in zip(idxs, markers, perps, strict=True)}

    for i, (ref_marker, ref_perp) in enumerate(_reference(points, lats, lngs)):
        if ref_perp <= buffer_miles:
            assert i in got, f"point {i} is {ref_perp:.3f} mi off-route but was dropped"
            marker, perp = got[i]
            assert perp == pytest.approx(ref_perp, abs=1e-9)
            assert marker == pytest.approx(ref_marker, abs=1e-9)
        elif i in got:
            # Outside the buffer the window may only ever overestimate, never
            # claim a point is closer than it really is.
            assert got[i][1] >= ref_perp - 1e-9


def _random_route(rng, n, lat0=39.0, lng0=-98.0):
    """A wandering polyline, roughly of US size."""
    points = [(lat0, lng0)]
    for _ in range(n - 1):
        lat, lng = points[-1]
        points.append((lat + rng.uniform(-0.3, 0.3), lng + rng.uniform(0.05, 0.4)))
    return points


def test_matches_reference_on_random_routes():
    rng = random.Random(7)
    for _ in range(25):
        points = _random_route(rng, rng.randint(2, 400))
        lats = [p[0] + rng.uniform(-0.6, 0.6) for p in points for _ in (0,)]
        lngs = [p[1] + rng.uniform(-0.6, 0.6) for p in points]
        _assert_matches_reference(points, lats, lngs)


def test_matches_reference_on_sparse_two_point_route():
    # A single 500+ mile segment: the window degenerates to "the whole route".
    points = [(40.0, -90.0), (40.0, -80.0)]
    rng = random.Random(11)
    lats = [40.0 + rng.uniform(-0.2, 0.2) for _ in range(60)]
    lngs = [rng.uniform(-91.0, -79.0) for _ in range(60)]
    _assert_matches_reference(points, lats, lngs)


def test_handles_duplicate_consecutive_vertices():
    # Zero-length segments must collapse to their origin, not divide by zero.
    points = [(40.0, -90.0), (40.0, -90.0), (40.0, -85.0), (40.0, -85.0), (40.0, -80.0)]
    lats = [40.0, 40.05, 39.95]
    lngs = [-88.0, -85.0, -81.0]
    with np.errstate(all="raise"):
        _assert_matches_reference(points, lats, lngs)


def test_degenerate_routes_return_nothing():
    for points in ([], [(40.0, -90.0)]):
        idxs, markers, perps = RouteProjector(points).project_within([40.0], [-90.0], BUFFER)
        assert len(idxs) == len(markers) == len(perps) == 0


def test_no_points_to_project():
    points = [(40.0, -90.0), (40.0, -80.0)]
    idxs, _, _ = RouteProjector(points).project_within([], [], BUFFER)
    assert len(idxs) == 0


def test_mile_marker_tracks_distance_along_route():
    points = [(40.0, -90.0), (40.0, -80.0)]
    proj = RouteProjector(points)
    total = proj.cum[-1]
    # A point level with the midpoint of the route sits at ~half the length.
    idxs, markers, perps = proj.project_within([40.0], [-85.0], BUFFER)
    assert len(idxs) == 1
    assert markers[0] == pytest.approx(total / 2, rel=1e-3)
    assert perps[0] == pytest.approx(0.0, abs=1e-6)


def test_search_radius_is_bounded_by_longest_segment():
    points = [(40.0, -90.0), (40.0, -89.0), (40.0, -80.0)]
    proj = RouteProjector(points)
    longest = max(np.diff(proj.cum))
    assert proj.max_segment_miles == pytest.approx(longest)
    assert proj.num_segments == 2


def test_projection_is_clamped_to_the_route_ends():
    # Points past the finish must all pile up on the final vertex rather than
    # running off the end -- this is what keeps stations past the destination
    # from being handed a mile marker beyond the route length.
    points = [(40.0, -90.0), (40.0, -80.0)]
    proj = RouteProjector(points)
    idxs, markers, _ = proj.project_within([40.0, 40.0], [-79.95, -79.90], buffer_miles=10.0)
    assert len(idxs) == 2
    assert markers[0] == pytest.approx(markers[1], abs=1e-9)  # both clamped to the end
    assert markers[0] == pytest.approx(proj.cum[-1], rel=1e-3)
    # Likewise before the start.
    idxs, markers, _ = proj.project_within([40.0, 40.0], [-90.05, -90.10], buffer_miles=10.0)
    assert len(idxs) == 2
    assert markers[0] == pytest.approx(0.0, abs=1e-9)
    assert markers[1] == pytest.approx(0.0, abs=1e-9)


def test_equirectangular_frame_is_accurate_at_high_latitude():
    # Longitude degrees shrink towards the poles; the projection must use the
    # local scale, not a flat 69 mi/degree.
    points = [(60.0, -150.0), (60.0, -149.0)]
    proj = RouteProjector(points)
    # 1 degree of longitude at 60N is ~34.5 mi, not ~69.
    assert proj.cum[-1] == pytest.approx(69.0 * math.cos(math.radians(60.0)), rel=0.01)
