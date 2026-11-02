"""
test_settings.py
----------------
app/settings.py: konfigurasi dibaca dari env, default-nya sama dengan
perilaku lama, dan CORS origin menerima dua format penulisan.
"""

import pytest

pydantic_settings = pytest.importorskip("pydantic_settings")  # noqa: F401

from app.config import MODEL_PATH  # noqa: E402
from app.settings import Settings  # noqa: E402


def test_defaults_match_legacy_behaviour():
    s = Settings()
    assert s.cv_mode == "indicator_only"
    assert s.max_upload_mb == 12
    # 12 MB -- angka yang sama dengan QUALITY["max_upload_bytes"] lama
    assert s.max_upload_bytes == 12 * 1024 * 1024
    assert s.max_dimension_px == 2000
    assert s.allowed_origins == []
    # default path model harus persis yang dipakai pipeline
    assert s.model_path == MODEL_PATH


def test_upload_limit_converted_to_bytes():
    assert Settings(max_upload_mb=4).max_upload_bytes == 4 * 1024 * 1024


def test_cv_mode_only_accepts_documented_values(monkeypatch):
    from pydantic import ValidationError

    monkeypatch.setenv("CV_MODE", "model")
    assert Settings().cv_mode == "model"

    monkeypatch.setenv("CV_MODE", "neura")
    with pytest.raises(ValidationError, match="cv_mode"):
        Settings()

    with pytest.raises(ValidationError, match="cv_mode"):
        Settings(cv_mode="neura")


def test_env_overrides_upload_limit(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "40")
    s = Settings()
    assert s.max_upload_mb == 40
    assert s.max_upload_bytes == 40 * 1024 * 1024


def test_allowed_origins_from_json_array(monkeypatch):
    monkeypatch.setenv(
        "ALLOWED_ORIGINS", '["https://a.example", "https://b.example"]'
    )
    assert Settings().allowed_origins == ["https://a.example", "https://b.example"]


def test_allowed_origins_from_comma_separated(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.example, https://b.example")
    assert Settings().allowed_origins == ["https://a.example", "https://b.example"]


def test_allowed_origins_empty_is_strict_not_wildcard(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "")
    s = Settings()
    assert s.allowed_origins == []
    assert "*" not in s.allowed_origins
