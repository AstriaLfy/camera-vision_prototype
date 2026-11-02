"""
test_deploy.py
--------------
Perilaku yang ditambahkan untuk deploy (Docker/ngrok) TANPA mengubah logika
endpoint:

  * konfigurasi baru di app/settings.py punya default yang aman;
  * /docs + /openapi.json aktif di mode dev dan bisa dimatikan lewat env;
  * /api/v1/debug-landmarks hanya terdaftar bila ENABLE_DEBUG_ENDPOINTS=true;
  * CORS mengizinkan header Authorization dan ngrok-skip-browser-warning;
  * /api/v1/health tetap melaporkan status model dengan pesan yang jelas.

Semua test di sini murni soal registrasi rute & konfigurasi; tidak menyentuh
business logic pipeline.
"""

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import CORS_ALLOW_HEADERS, app, create_app  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402

DEBUG_PATH = "/api/v1/debug-landmarks"
HEALTH_PATH = "/api/v1/health"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Default konfigurasi baru
# ---------------------------------------------------------------------------

def test_new_settings_defaults_are_safe():
    s = Settings()
    assert s.database_url.startswith("postgresql+psycopg://")
    assert s.postgres_user == "app"
    assert s.postgres_db == "app"
    assert s.public_base_url == "https://jaws-hurler-unranked.ngrok-free.dev"
    assert s.ngrok_domain == "jaws-hurler-unranked.ngrok-free.dev"
    # Authtoken tidak boleh punya default nyata.
    assert s.ngrok_authtoken == ""
    assert s.web_workers == 1
    assert s.enable_debug_endpoints is False
    assert s.app_env == "dev"


def test_database_url_uses_psycopg3_driver():
    assert Settings().database_url.startswith("postgresql+psycopg://")


def test_resolved_enable_docs_follows_app_env():
    assert Settings().resolved_enable_docs is True          # dev default
    assert Settings(app_env="prod").resolved_enable_docs is False
    assert Settings(app_env="prod", enable_docs=True).resolved_enable_docs is True
    assert Settings(enable_docs=False).resolved_enable_docs is False


# ---------------------------------------------------------------------------
# Dokumentasi: aktif di dev, bisa dimatikan
# ---------------------------------------------------------------------------

def test_docs_active_by_default_in_dev(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_docs_can_be_disabled():
    with TestClient(create_app(Settings(enable_docs=False))) as c:
        assert c.get("/docs").status_code == 404
        assert c.get("/redoc").status_code == 404
        assert c.get("/openapi.json").status_code == 404
        # Endpoint produksi tetap hidup walau dokumentasi mati.
        assert c.get(HEALTH_PATH).status_code == 200


# ---------------------------------------------------------------------------
# Endpoint debug: hanya terdaftar bila diaktifkan
# ---------------------------------------------------------------------------

def test_debug_endpoint_not_registered_by_default(client):
    spec = client.get("/openapi.json").json()
    assert DEBUG_PATH not in spec["paths"]
    # Rute benar-benar tidak ada, bukan sekadar 401/403.
    assert client.post(DEBUG_PATH).status_code == 404


def test_debug_endpoint_registered_when_enabled():
    with TestClient(create_app(Settings(enable_debug_endpoints=True))) as c:
        spec = c.get("/openapi.json").json()
        assert DEBUG_PATH in spec["paths"]
        # Tanpa file, validasi multipart 422 -- bukan 404 (rute ada).
        assert c.post(DEBUG_PATH).status_code == 422


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

def test_cors_allows_authorization_and_ngrok_header():
    settings = Settings(allowed_origins=["https://app.example"])
    with TestClient(create_app(settings)) as c:
        r = c.options(
            "/api/v1/screening/analyze",
            headers={
                "Origin": "https://app.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": (
                    "authorization, ngrok-skip-browser-warning, content-type"
                ),
            },
        )
    assert r.headers.get("access-control-allow-origin") == "https://app.example"
    allowed = r.headers.get("access-control-allow-headers", "").lower()
    assert "authorization" in allowed
    assert "ngrok-skip-browser-warning" in allowed


def test_sensitive_headers_are_in_configured_allow_list():
    lowered = [h.lower() for h in CORS_ALLOW_HEADERS]
    assert "authorization" in lowered
    assert "ngrok-skip-browser-warning" in lowered


def test_wildcard_is_never_a_default_origin():
    assert "*" not in Settings().allowed_origins


# ---------------------------------------------------------------------------
# Health: status model dengan pesan jelas
# ---------------------------------------------------------------------------

def test_health_reports_model_message(client):
    body = client.get(HEALTH_PATH).json()
    assert isinstance(body["model_available"], bool)
    assert isinstance(body["model_message"], str) and body["model_message"]
    # Pesan tidak boleh membocorkan path filesystem.
    assert ".task" not in body["model_message"]
    assert "/" not in body["model_message"]


def test_health_docs_field_matches_settings(client):
    body = client.get(HEALTH_PATH).json()
    assert body["docs"] == ("/docs" if get_settings().resolved_enable_docs else None)
