"""End-to-end API test with the routing call mocked (no network)."""

import json
from html.parser import HTMLParser
from unittest.mock import patch

import pytest
from rest_framework.test import APIClient

from routing.client import RouteResult
from stations.models import FuelStation
from trips.geo import cumulative_miles

ROUTE_POINTS = [(40.0, -90.0), (40.0, -80.0)]
ROUTE_LEN = cumulative_miles(ROUTE_POINTS)[-1]


def _fake_route(*args, **kwargs):
    return RouteResult(
        geometry={"type": "LineString", "coordinates": [[lng, lat] for lat, lng in ROUTE_POINTS]},
        points=ROUTE_POINTS,
        total_distance_miles=ROUTE_LEN,
        duration_minutes=600.0,
        provider="mock",
        start=ROUTE_POINTS[0],
        finish=ROUTE_POINTS[-1],
    )


@pytest.fixture
def stations_on_route(db):
    # Stations spread along the line (detour ~0), prices varying.
    specs = [(-89.0, 3.40), (-87.0, 3.10), (-85.0, 2.95), (-83.0, 3.30), (-81.0, 3.05)]
    for i, (lng, price) in enumerate(specs, start=1):
        FuelStation.objects.create(
            opis_id=i,
            name=f"STOP {i}",
            address="",
            city="X",
            state="IL",
            retail_price=price,
            latitude=40.0,
            longitude=lng,
            is_geocoded=True,
        )


@pytest.mark.django_db
def test_post_route_fuel_plan(stations_on_route):
    client = APIClient()
    with patch("trips.services.get_route", side_effect=_fake_route):
        resp = client.post(
            "/api/v1/route-fuel-plan/",
            {"start": {"lat": 40.0, "lng": -90.0}, "finish": {"lat": 40.0, "lng": -80.0}},
            format="json",
        )
    assert resp.status_code == 200
    data = resp.json()

    # Shape
    assert set(data) == {"route", "fuel", "fuel_stops"}
    assert data["route"]["provider"] == "mock"
    assert "map_url" not in data["route"]
    assert "geometry" not in data["route"]

    # Fuel stops ordered by mile marker.
    markers = [s["route_mile_marker"] for s in data["fuel_stops"]]
    assert markers == sorted(markers)
    assert [s["order"] for s in data["fuel_stops"]] == list(range(1, len(markers) + 1))

    # Totals reconcile: total gallons == distance / mpg, and the per-stop costs
    # sum to the reported total cost.
    total_gallons = data["fuel"]["total_gallons"]
    assert total_gallons == pytest.approx(ROUTE_LEN / 10.0, abs=0.05)
    stop_cost_sum = sum(float(s["cost_usd"]) for s in data["fuel_stops"])
    assert stop_cost_sum == pytest.approx(float(data["fuel"]["total_cost_usd"]), abs=0.02)


@pytest.mark.django_db
def test_get_form_and_missing_params(stations_on_route):
    client = APIClient()
    # Missing params -> 400.
    assert client.get("/api/v1/route-fuel-plan/").status_code == 400

    with patch("trips.services.get_route", side_effect=_fake_route):
        resp = client.get("/api/v1/route-fuel-plan/?start=40.0,-90.0&finish=40.0,-80.0")
    assert resp.status_code == 200
    assert resp.json()["fuel_stops"]
    assert "geometry" not in resp.json()["route"]
    assert "map_url" not in resp.json()["route"]


@pytest.mark.django_db
@pytest.mark.parametrize("method", ["get", "post"])
def test_geometry_is_opt_in_without_changing_the_cached_plan(stations_on_route, method):
    client = APIClient()
    url = "/api/v1/route-fuel-plan/"
    query = "?start=40,-90&finish=40,-80"
    with patch("trips.services.get_route", side_effect=_fake_route) as get_route:
        if method == "post":
            detailed = client.post(
                url,
                {
                    "start": {"lat": 40.0, "lng": -90.0},
                    "finish": {"lat": 40.0, "lng": -80.0},
                    "include_geometry": True,
                },
                format="json",
            )
        else:
            detailed = client.get(url + query + "&include_geometry=true")
        compact = client.get(url + query)
        detailed_again = client.get(url + query + "&include_geometry=true")
    assert detailed.status_code == compact.status_code == detailed_again.status_code == 200
    assert detailed.json()["route"]["geometry"]["type"] == "LineString"
    assert "map_url" not in detailed.json()["route"]
    assert "geometry" not in compact.json()["route"]
    assert detailed_again.json()["route"]["geometry"] == detailed.json()["route"]["geometry"]
    assert compact.json()["fuel"] == detailed.json()["fuel"]
    assert compact.json()["fuel_stops"] == detailed.json()["fuel_stops"]
    get_route.assert_called_once()


@pytest.mark.django_db
def test_map_data_is_ready_for_leaflet(stations_on_route):
    """The browser can parse each map payload once into its expected type."""

    class MapDataParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.current = None
            self.scripts = {}

        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            if tag == "script" and attributes.get("type") == "application/json":
                self.current = attributes["id"]
                self.scripts[self.current] = ""

        def handle_data(self, data):
            if self.current:
                self.scripts[self.current] += data

        def handle_endtag(self, tag):
            if tag == "script":
                self.current = None

    client = APIClient()
    with patch("trips.services.get_route", side_effect=_fake_route) as get_route:
        compact = client.get("/api/v1/route-fuel-plan/?start=40,-90&finish=40,-80")
        response = client.get("/api/v1/route-fuel-plan/map/?start=40,-90&finish=40,-80")
    assert compact.status_code == 200
    assert "geometry" not in compact.json()["route"]
    get_route.assert_called_once()
    assert response.status_code == 200
    parser = MapDataParser()
    parser.feed(response.content.decode())
    geometry = json.loads(parser.scripts["geometry-data"])
    stops = json.loads(parser.scripts["stops-data"])
    summary = json.loads(parser.scripts["summary-data"])
    assert geometry["type"] == "LineString"
    assert geometry["coordinates"] == [[lng, lat] for lat, lng in ROUTE_POINTS]
    assert isinstance(stops, list) and stops
    assert summary["fuel"]["mpg"] == 10
    assert summary["fuel"]["tank_range_miles"] == 500
