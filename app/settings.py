"""
settings.py
-----------
Konfigurasi runtime layanan, dibaca dari environment variable atau file `.env`.

Beda peran dengan app/config.py:

  * app/config.py  -> angka threshold & index landmark yang DIPAKAI pipeline CV.
                      File itu tidak dikonfigurasi lewat env (dan tidak boleh
                      diubah -- pipeline dan dataset harus tetap satu versi).
  * app/settings.py -> pengaturan deploy: mode layanan, batas upload, CORS,
                      path model. Semua bisa diganti lewat env tanpa menyentuh
                      kode.

Semua field punya default yang SAMA PERSIS dengan perilaku lama, jadi env var
boleh ditinggalkan tanpa mengubah hasil.

Pakai:
    cp .env.example .env    # lalu sesuaikan; .env jangan di-commit

Akses dari kode:
    from app.settings import get_settings
    get_settings().max_upload_mb
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from .config import MODEL_PATH


#: Tipe upload default (jpeg/png). Dipakai juga saat ALLOWED_UPLOAD_TYPES
#: kosong, supaya "kosong" tidak berarti "terima semua".
DEFAULT_UPLOAD_TYPES: list[str] = ["image/jpeg", "image/jpg", "image/png"]


class Settings(BaseSettings):
    """Pengaturan layanan. Nama field == nama env var (huruf besar)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `FOO=` (kosong) di .env dianggap "tidak diisi", bukan value kosong.
        env_ignore_empty=True,
    )

    # Mode layanan CV. Saat ini hanya "indicator_only" yang punya implementasi
    # (pipeline rule-based). Nilai lain disiapkan supaya penambahan model
    # terlatih nanti tidak perlu mengubah bentuk konfigurasi.
    cv_mode: Literal["indicator_only", "model", "off"] = "indicator_only"

    # Batas ukuran upload foto, dalam MB. Default 12 MB, sama dengan nilai lama
    # di app/config.py::QUALITY. Lihat properti max_upload_bytes di bawah.
    max_upload_mb: int = 12

    # Sisi terpanjang foto (px). Lebih besar dari ini di-downscale dulu.
    # Default 2000, sama dengan nilai lama di app/config.py::QUALITY.
    max_dimension_px: int = 2000

    # Model FaceLandmarker. Default diambil dari app/config.py::MODEL_PATH
    # supaya tidak mungkin beda.
    #
    # PERINGATAN: app/pipeline.py masih membaca config.MODEL_PATH secara
    # langsung (mengubah pipeline.py di luar cakupan refaktor ini). Menimpa
    # MODEL_PATH lewat env karena itu BARU memengaruhi pelaporan
    # /api/v1/health, bukan model yang benar-benar dipakai pipeline.
    # Karena itu .env.example men-disable baris MODEL_PATH secara default.
    model_path: Path = MODEL_PATH

    # Origin yang diizinkan CORS. Default KOSONG = ketat: tidak ada origin
    # asing yang boleh memanggil API dari browser. Isi sebagai JSON array
    # atau daftar dipisah koma:
    #   ALLOWED_ORIGINS=["https://app.example.com","https://admin.example.com"]
    #   ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com
    # Nilai "*" TIDAK disarankan: itu membuka API ke situs mana pun.
    allowed_origins: Annotated[list[str], NoDecode] = []

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: object) -> object:
        """Env var tunggal menerima JSON array ATAU daftar dipisah koma."""
        if not isinstance(value, str):
            return value
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("["):
            return json.loads(raw)
        return [part.strip() for part in raw.split(",") if part.strip()]

    # Tipe file yang diterima upload foto. Default jpeg/png saja. Format env
    # sama dengan ALLOWED_ORIGINS: JSON array ATAU daftar dipisah koma.
    #   ALLOWED_UPLOAD_TYPES=["image/jpeg","image/png"]
    allowed_upload_types: Annotated[list[str], NoDecode] = [
        "image/jpeg", "image/jpg", "image/png",
    ]

    @field_validator("allowed_upload_types", mode="before")
    @classmethod
    def _parse_upload_types(cls, value: object) -> object:
        """Env var tunggal menerima JSON array ATAU daftar dipisah koma."""
        if not isinstance(value, str):
            return value
        raw = value.strip()
        if not raw:
            return DEFAULT_UPLOAD_TYPES
        if raw.startswith("["):
            return json.loads(raw)
        return [part.strip() for part in raw.split(",") if part.strip()]

    # ------------------------------------------------------------------
    # Deploy (Docker / Postgres / ngrok). Semua OPSIONAL dan punya default
    # yang sama persis dengan perilaku lama, jadi tanpa .env pun layanan
    # CV tetap jalan seperti sebelumnya.
    # ------------------------------------------------------------------

    # Lingkungan deploy: "dev" (default) atau "prod". Dipakai hanya untuk
    # menentukan default dokumentasi interaktif (lihat enable_docs).
    app_env: Literal["dev", "prod"] = "dev"

    # Dokumentasi interaktif (/docs, /redoc, /openapi.json).
    #   None  -> ikut app_env: aktif di "dev", mati di "prod".
    #   true  -> selalu aktif (mis. API internal di balik auth).
    #   false -> selalu mati (aman untuk endpoint publik lewat ngrok).
    # Default None == perilaku lama (dokumen aktif), karena app_env default "dev".
    enable_docs: bool | None = None

    # Endpoint debug (/api/v1/debug-landmarks) hanya didaftarkan bila true.
    # Default false: endpoint itu mengembalikan gambar berisi landmark, tidak
    # untuk produksi. Lihat app/main.py.
    enable_debug_endpoints: bool = False

    # Koneksi database. Format driver: postgresql+psycopg:// (psycopg 3).
    # Default menunjuk ke service `db` di docker-compose.yml; di luar Docker
    # ganti host/port lewat .env.
    database_url: str = "postgresql+psycopg://app:ganti-ini@db:5432/app"

    # Kredensial Postgres untuk service `db`/`db-test` di compose. Nilai nyata
    # hanya boleh ada di .env (jangan di-commit).
    postgres_user: str = "app"
    postgres_password: str = "ganti-ini"
    postgres_db: str = "app"

    # URL publik layanan (lewat ngrok). Dipakai untuk membangun tautan absolut
    # / dokumentasi; tidak mengubah perilaku endpoint yang ada.
    public_base_url: str = "https://jaws-hurler-unranked.ngrok-free.dev"

    # Domain statis ngrok dan authtoken. Authtoken SENGAJA default kosong:
    # disuplai lewat environment saat menjalankan profil "tunnel" dan TIDAK
    # pernah ditulis ke file yang di-commit.
    ngrok_domain: str = "jaws-hurler-unranked.ngrok-free.dev"
    ngrok_authtoken: str = ""

    # Jumlah worker uvicorn. Default 1 (perilaku lama: proses tunggal).
    web_workers: int = 1

    @property
    def resolved_enable_docs(self) -> bool:
        """Dokumentasi aktif? `enable_docs` bila diisi, kalau tidak ikut app_env."""
        if self.enable_docs is not None:
            return self.enable_docs
        return self.app_env == "dev"

    # Ambang referensi untuk indikator visual (app/indicators.py).
    #
    # DEFAULT KOSONG = tidak ada ambang yang dipakai sama sekali. Modul
    # indikator bersifat deskriptif: tidak ada ambang normal/abnormal, tidak
    # ada klasifikasi.
    #
    # Setiap entri WAJIB punya field `sumber` (asal angka: literatur,
    # pedoman, atau dataset mana). Entri tanpa `sumber` DIABAIKAN saat
    # dibaca lewat app.indicators.muat_ambang() -- ambang tanpa asal usul
    # tidak bisa dipertanggungjawabkan.
    #
    # Format env: JSON array of object.
    #   AMBANG_INDIKATOR=[{"kunci":"hair_l","sumber":"WHO 2006","bawah":10}]
    ambang_indikator: Annotated[list[dict], NoDecode] = []

    @field_validator("ambang_indikator", mode="before")
    @classmethod
    def _parse_ambang(cls, value: object) -> object:
        """Env var berupa JSON array; kosong -> daftar kosong."""
        if not isinstance(value, str):
            return value
        raw = value.strip()
        if not raw:
            return []
        return json.loads(raw)

    @property
    def max_upload_bytes(self) -> int:
        """Batas upload dalam byte, untuk dipakai kode decode/upload."""
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton konfigurasi (di-cache supaya `.env` tidak dibaca berulang).

    Untuk menguji perubahan konfigurasi: `get_settings.cache_clear()`.
    """
    return Settings()
