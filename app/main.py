"""
main.py
-------
Titik masuk FastAPI.

File ini HANYA membuat app, memasang CORS + handler error seragam, dan
mendaftarkan router. Semua logika endpoint ada di app/routers/:

  * screening.py -> POST /api/v1/screening/analyze, POST /api/v1/debug-landmarks
  * health.py    -> GET  /api/v1/health, GET /

Menambah endpoint baru = bikin app/routers/<domain>.py lalu
`app.include_router(...)` di bawah. Lihat app/routers/__init__.py.

Konfigurasi layanan (mode CV, batas upload, CORS) ada di app/settings.py;
dependencies lintas endpoint (autentikasi) ada di app/deps.py.

Jalankan:
    uvicorn app.main:app --reload --port 8000

Dokumentasi interaktif: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from . import __version__
from .errors import install_error_handlers
from .routers import health, screening
from .routers.screening import _apply_exif_orientation, _decode_upload  # noqa: F401
from .settings import Settings, get_settings

#: Header yang boleh dipakai browser pada request lintas-origin.
#:
#:   * Authorization            -> token auth (klien web & admin).
#:   * ngrok-skip-browser-warning -> melewati halaman peringatan ngrok Free
#:     saat klien memanggil URL publik (lihat README bagian deploy).
#:   * Content-Type/Accept/Origin -> upload multipart + preflight normal.
#:
#: Daftar ini eksplisit (bukan "*") supaya header yang diizinkan terlihat
#: jelas dan tidak ikut terbuka saat origin baru didaftarkan.
CORS_ALLOW_HEADERS = [
    "Authorization",
    "Content-Type",
    "Accept",
    "Origin",
    "ngrok-skip-browser-warning",
]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Bangun aplikasi FastAPI dari `settings`.

    Dipisah dari instance modul supaya test bisa membuat app dengan
    konfigurasi berbeda (mis. dokumentasi nonaktif, endpoint debug aktif)
    tanpa mengubah environment global. `app` di bawah memakai `get_settings()`
    sehingga perilaku uvicorn tidak berubah.
    """
    settings = settings or get_settings()
    docs_enabled = settings.resolved_enable_docs

    app = FastAPI(
        title="Skrining Malnutrisi dari Foto Wajah - CV Pipeline Prototype",
        version=__version__,
        description=(
            "ALAT BANTU SKRINING, BUKAN ALAT DIAGNOSIS. Output rule-based berbasis "
            "landmark, tanpa model yang dilatih dari dataset malnutrisi. Konfirmasi "
            "wajib dengan pemeriksaan MUAC dan/atau tes hemoglobin."
        ),
        # Dokumentasi bisa dimatikan lewat ENABLE_DOCS / APP_ENV=prod supaya
        # tidak terekspos saat API dipublikasikan lewat ngrok. Saat mati,
        # /openapi.json ikut mati (FastAPI menurunkannya dari openapi_url).
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )

    # CORS: default ketat -- daftar origin kosong, jadi tidak ada origin asing
    # yang boleh memanggil API dari browser. Buka seperlunya lewat ALLOWED_ORIGINS
    # di .env (lihat app/settings.py). Origin "*"/tanpa filter tidak dipakai.
    #
    # Catatan: aplikasi mobile native TIDAK terikat CORS; daftar origin hanya
    # relevan untuk klien berbasis browser (lihat README bagian deploy).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=CORS_ALLOW_HEADERS,
    )

    install_error_handlers(app)

    app.include_router(screening.router)
    # Endpoint debug hanya terdaftar bila diaktifkan (default mati), supaya
    # tidak ikut terekspos saat API dipublikasikan.
    if settings.enable_debug_endpoints:
        app.include_router(screening.debug_router)
    app.include_router(health.router)
    app.include_router(health.root_router)

    return app


app = create_app()
