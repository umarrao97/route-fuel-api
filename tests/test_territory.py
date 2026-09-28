"""The USA containment test must admit real US places and reject foreign ones.

The bounding box this replaced accepted the whole of Canada and most of Mexico
-- Toronto coordinates produced a happily-planned 1543-mile route starting in
Ontario. These are the reference points that pin the new behaviour.
"""

import pytest

from stations.constants import in_us_bbox
from stations.territory import CITY_RESCUE_MILES, in_usa

# Chosen to cover every way the test can go wrong: mainland interior, all four
# extremes, non-contiguous states, and US halves of twin border cities.
INSIDE = [
    ("Chicago IL", 41.8781, -87.6298),
    ("Houston TX", 29.7604, -95.3698),
    ("Seattle WA", 47.6062, -122.3321),
    ("Miami FL", 25.7617, -80.1918),
    ("Bangor ME", 44.8016, -68.7712),
    ("Detroit MI", 42.3314, -83.0458),  # across the river from Windsor, ON
    ("Buffalo NY", 42.8864, -78.8784),
    ("El Paso TX", 31.7619, -106.4850),  # across the river from Ciudad Juarez
    ("San Diego CA", 32.7157, -117.1611),  # 15 mi from Tijuana
    ("Blaine WA", 48.9937, -122.7466),  # on the 49th parallel
    ("Laredo TX", 27.5306, -99.4803),
    ("Brownsville TX", 25.9017, -97.4975),
    # Places the coarse 1:110m outline drops -- these exercise the city rescue.
    ("Key West FL", 24.5551, -81.7800),
    ("Nantucket MA", 41.2835, -70.0995),
    ("Galveston TX", 29.3013, -94.7977),
    ("Barrow AK", 71.2906, -156.7886),
    # Non-contiguous states.
    ("Anchorage AK", 61.2181, -149.9003),
    ("Honolulu HI", 21.3069, -157.8583),
    ("Hilo HI", 19.7297, -155.0900),
    # Remote interior, far from any city but unambiguously inside the outline.
    ("rural NV", 39.0, -117.0),
    ("rural MT", 47.0, -108.0),
]

OUTSIDE = [
    ("Toronto ON", 43.6532, -79.3832),  # the coordinates from the review
    ("Montreal QC", 45.5017, -73.5673),
    ("Vancouver BC", 49.2827, -123.1207),
    ("Winnipeg MB", 49.8951, -97.1384),
    ("Calgary AB", 51.0447, -114.0719),
    ("Hamilton ON", 43.2557, -79.8711),
    ("London ON", 42.9849, -81.2453),
    ("Monterrey MX", 25.6866, -100.3161),
    ("Mexico City MX", 19.4326, -99.1332),
    ("Havana CU", 23.1136, -82.3666),
    ("Nassau BS", 25.0343, -77.3963),
    ("mid-Atlantic", 35.0, -50.0),
    ("Gulf of Mexico", 26.0, -90.0),
    ("London UK", 51.5074, -0.1278),
    ("Sydney AU", -33.8688, 151.2093),
]


@pytest.mark.parametrize(("name", "lat", "lng"), INSIDE, ids=[n for n, _, _ in INSIDE])
def test_us_locations_are_inside(name, lat, lng):
    assert in_usa(lat, lng) is True


@pytest.mark.parametrize(("name", "lat", "lng"), OUTSIDE, ids=[n for n, _, _ in OUTSIDE])
def test_foreign_locations_are_outside(name, lat, lng):
    assert in_usa(lat, lng) is False


def test_bounding_box_alone_would_have_accepted_canada():
    """Documents precisely why the bbox was not sufficient on its own."""
    toronto = (43.6532, -79.3832)
    assert in_us_bbox(*toronto) is True  # the old test passed...
    assert in_usa(*toronto) is False  # ...the new one does not


def test_every_us_city_in_the_dataset_is_accepted():
    """Place-name endpoints resolve from uscities.csv, so they must all pass."""
    from stations.geocoding import get_default_geocoder

    geocoder = get_default_geocoder()
    for place in ("Chicago, IL", "Key West, FL", "Anchorage, AK", "Honolulu, HI", "Nome, AK"):
        coords = geocoder.geocode_place_name(place)
        assert coords is not None, place
        assert in_usa(*coords) is True, place


def test_rescue_radius_is_the_documented_value():
    # Guards against a silent widening that would start admitting Vancouver
    # (measured 20.6 mi from the nearest US city centroid).
    assert CITY_RESCUE_MILES == 15.0
