# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Image aplikasi skrining-malnutrisi-cv.
#
# Multi-stage:
#   * builder  -> venv + dependency dari requirements.txt (pin tetap)
#   * runtime  -> hanya venv + kode + library sistem yang benar-benar dipakai
#                 OpenCV/MediaPipe, dijalankan sebagai user non-root.
#
# Versi Python DISAMAKAN dengan venv proyek (target runtime terverifikasi:
# Python 3.14). Jangan naik/turun tanpa menguji ulang wheel numpy/opencv.
# ---------------------------------------------------------------------------

FROM python:3.14-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

# build-essential hanya di stage builder: sebagian dependency bisa datang
# sebagai source distribution (tanpa wheel untuk arsitektur/versi tertentu).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip wheel \
    && /opt/venv/bin/pip install -r requirements.txt


FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

# Library sistem yang diperlukan binary wheel:
#   * libgl1          -> libGL.so.1 (Qt5Gui/Widgets bawaan opencv-contrib)
#   * libglib2.0-0    -> libglib-2.0.so.0 / libgthread-2.0.so.0 (Qt5Core)
#   * libegl1         -> libEGL.so.1 (libmediapipe.so)
#   * libgles2        -> libGLESv2.so.2 (libmediapipe.so)
# Sisanya (libstdc++, libgcc, zlib, pthread) sudah ada di base image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libegl1 \
        libgles2 \
    && rm -rf /var/lib/apt/lists/*

# User non-root. UID/GID tetap supaya izin volume konsisten antar-host.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Kode aplikasi + Alembic (dipakai service `migrate` yang memakai image ini).
# tests/, data/, reports/, .env dsb. TIDAK ikut -- lihat .dockerignore.
COPY requirements.txt ./
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
COPY download_model.py ./

# Direktori model boleh kosong: aplikasi tetap hidup tanpa model
# (/api/v1/health melaporkan model_available=false). Model bisa datang lewat
# (a) volume ke /app/models, atau (b) build arg MODEL_URL di bawah.
RUN mkdir -p /app/models \
    && chown -R app:app /app

# Build arg MODEL_URL: KOSONG secara default. JANGAN mengarang URL -- isi
# hanya bila kamu punya URL model resmi. Volume lebih disarankan untuk dev.
ARG MODEL_URL=""
RUN if [ -n "${MODEL_URL}" ]; then \
        python -c "import urllib.request; urllib.request.urlretrieve('${MODEL_URL}', '/app/models/face_landmarker.task')" \
        && chown app:app /app/models/face_landmarker.task ; \
    fi

USER app

EXPOSE 8000

# Healthcheck memanggil GET /api/v1/health (tanpa dependency eksternal).
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).status == 200 else 1)"]

# --proxy-headers: percaya header X-Forwarded-* (dari ngrok, yang berjalan di
# jaringan docker). --forwarded-allow-ips DIBATASI ke subnet jaringan compose,
# bukan "*", supaya spoofing header dari sumber lain tidak dipercaya.
# WEB_WORKERS diatur lewat environment (default 1).
ENV WEB_WORKERS=1 \
    FORWARDED_ALLOW_IPS=172.28.0.0/16
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips=${FORWARDED_ALLOW_IPS} --workers ${WEB_WORKERS}"]
