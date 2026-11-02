"""
health.py
----------
Status layanan.

Dua router di file ini:
  * `router`      -> `/api/v1/health`
  * `root_router` -> `/` (beranda singkat; di luar namespace /api/v1)

Root sengaja tidak ditaruh di main.py supaya main.py tetap hanya
mendaftarkan router.
"""

from __future__ import annotations

from fastapi import APIRouter

from .. import __version__, content
from ..settings import get_settings

router = APIRouter(prefix="/api/v1", tags=["health"])
root_router = APIRouter(tags=["root"])

#: Pesan yang menjelaskan status model TANPA membocorkan path filesystem.
MODEL_AVAILABLE_MESSAGE = "Model FaceLandmarker tersedia."
MODEL_MISSING_MESSAGE = (
    "Model FaceLandmarker belum tersedia di server; layanan tetap hidup dan "
    "endpoint screening tetap memproses data antropometri, tetapi data "
    "pendukung visual dilewati. Sediakan model lewat volume ke MODEL_PATH "
    "atau build arg MODEL_URL (lihat README bagian deploy)."
)


@router.get("/health")
def health() -> dict:
    """Liveness/readiness singkat tanpa melakukan inference apa pun.

    Tidak memuat path model -- cukup boolean + pesan singkat, supaya detail
    filesystem server tidak ikut terkirim ke klien. Aplikasi tetap hidup walau
    model tidak ada; `model_available=false` beserta `model_message` menjelaskan
    kenapa data pendukung visual tidak akan muncul.
    """
    settings = get_settings()
    model_available = settings.model_path.exists()
    return {
        "status": "ok",
        "version": __version__,
        "service": "skrining-malnutrisi-cv",
        "cv_mode": settings.cv_mode,
        "model_available": model_available,
        "model_message": (
            MODEL_AVAILABLE_MESSAGE if model_available else MODEL_MISSING_MESSAGE
        ),
        "docs": "/docs" if settings.resolved_enable_docs else None,
    }


@root_router.get("/")
def root() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "docs": "/docs" if settings.resolved_enable_docs else None,
        "endpoint": "/api/v1/screening/analyze",
        "disclaimer": content.get_text("endpoint.disclaimer_akar"),
    }
