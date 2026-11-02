"""
test_health.py
--------------
Endpoint GET /api/v1/health.

Yang dijaga di sini: health harus cepat (tanpa inference), membaca konfigurasi
dari settings, dan benar-benar berada di bawah prefix /api/v1.
"""

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.settings import get_settings  # noqa: E402

ALLOWED_CV_MODES = {"indicator_only", "model", "off"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_200(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"]


def test_health_reports_app_version(client):
    """Item: health harus mengembalikan status DAN versi app."""
    from app import __version__

    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["version"] == __version__


def test_app_version_is_registered_in_openapi(client):
    from app import __version__

    spec = client.get("/openapi.json").json()
    assert spec["info"]["version"] == __version__


def test_health_reports_cv_mode_from_settings(client):
    """cv_mode harus datang dari settings, bukan string yang diketik di router."""
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["cv_mode"] == get_settings().cv_mode
    assert body["cv_mode"] in ALLOWED_CV_MODES


def test_health_reports_model_availability_as_bool(client):
    r = client.get("/api/v1/health")
    assert isinstance(r.json()["model_available"], bool)


def test_health_is_under_api_v1_prefix(client):
    """Tanpa prefix harus 404, supaya tidak ada /health yang tersembunyi."""
    assert client.get("/health").status_code == 404
    assert client.get("/api/v1/health").status_code == 200


def test_root_endpoint_still_works(client):
    """Root berada di luar /api/v1 dan tidak boleh hilang saat router dipisah."""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["endpoint"] == "/api/v1/screening/analyze"
