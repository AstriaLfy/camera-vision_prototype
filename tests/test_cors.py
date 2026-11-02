"""
test_cors.py
------------
CORS harus ketat secara default: daftar origin kosong, bukan "*".

Tanpa ALLOWED_ORIGINS, tidak ada origin asing yang boleh memanggil API dari
browser. Pembukaan origin lewat env diuji di tests/test_settings.py.
"""

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.settings import get_settings  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_default_allowed_origins_is_empty_not_wildcard():
    """Default ketat: tidak ada origin, dan tidak pernah '*'."""
    origins = get_settings().allowed_origins
    assert origins == []
    assert "*" not in origins


def test_cross_origin_request_gets_no_access_control_header(client):
    r = client.get("/api/v1/health", headers={"Origin": "https://evil.example"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") is None


def test_preflight_from_unknown_origin_is_not_granted(client):
    r = client.options(
        "/api/v1/screening/analyze",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.headers.get("access-control-allow-origin") is None
    assert r.status_code in (200, 400)


def test_same_origin_request_is_unaffected(client):
    """Tanpa header Origin sama sekali, endpoint tetap jalan normal."""
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
