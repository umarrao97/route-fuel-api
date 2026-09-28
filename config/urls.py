"""URL configuration for the USA Fuel-Optimal Route API."""

from django.contrib import admin
from django.urls import include, path

from trips.views import index

urlpatterns = [
    path("", index, name="index"),
    path("admin/", admin.site.urls),
    path("api/v1/", include("trips.urls")),
]
