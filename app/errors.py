"""
errors.py
---------
Format error seragam untuk seluruh API.

Setiap response error memuat field canonical:

    {"error": {"code": "<slug>", "message": "<pesan yang bisa dibaca>"}}

Field `detail` dari FastAPI TETAP dikirim apa adanya. Ini sengaja: format lama
sudah dipakai klien dan test yang ada, dan perintah refaktor ini melarang
mengubah perilaku endpoint yang sudah ada. Jadi `error` ditambahkan, bukan
menggantikan `detail`. Klien baru cukup membaca `error`; klien lama tetap jalan.

`code` adalah slug stabil turunan dari status code. Endpoint bisa membedakan
kasus dengan status code sama (mis. 422 quality gate vs 422 validasi) dengan
memasukkan kunci "code" di dict `detail` -- handler menghormatinya.

Tiga handler dipasang di install_error_handlers():

  1. HTTPException           -> 4xx/5xx yang disengaja (error_code/error_message)
  2. RequestValidationError -> 422 validasi input FastAPI (VALIDATION_CODE)
  3. Exception               -> error tak terduga (500), balasan generik tanpa
                                stack trace / pesan exception, karena pesan itu
                                bisa berisi path file, isi upload, atau data
                                subjek. Traceback hanya ditulis ke log server.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("app.errors")

#: Slug `error.code` per status code HTTP.
STATUS_CODES: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "unprocessable_entity",
    429: "too_many_requests",
    500: "internal_error",
    501: "not_implemented",
    503: "service_unavailable",
}

#: `error.code` untuk kegagalan validasi input (422 dari RequestValidationError).
VALIDATION_CODE = "validation_error"


def error_code(status_code: int, detail: Any = None) -> str:
    """Slug untuk satu response error."""
    if isinstance(detail, dict):
        override = detail.get("code")
        if isinstance(override, str) and override:
            return override
    return STATUS_CODES.get(status_code, f"http_{status_code}")


def error_message(detail: Any) -> str:
    """Sahkan `detail` FastAPI jadi satu kalimat yang bisa dibaca manusia."""
    if detail is None:
        return ""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        parts = [str(detail[k]) for k in ("reason", "hint", "message")
                 if detail.get(k)]
        if parts:
            return " — ".join(parts)
        # error validasi per-field: {loc, msg, type, ...}
        msg = str(detail.get("msg", "")).strip()
        if msg:
            loc = ".".join(str(seg) for seg in detail.get("loc", ()))
            return f"{loc}: {msg}" if loc else msg
        return str(detail)
    if isinstance(detail, list):
        return "; ".join(
            part for part in (error_message(item) for item in detail) if part
        )
    return str(detail)


def error_body(status_code: int, detail: Any) -> dict:
    """Bentuk response error seragam, ditambah `detail` lama."""
    body: dict[str, Any] = {
        "error": {
            "code": error_code(status_code, detail),
            "message": error_message(detail),
        }
    }
    if detail is not None:
        body["detail"] = detail
    return body


#: Pesan untuk 500. DIPERBAIKI -- jangan pernah diganti dengan str(exc),
#: karena str(exception) bisa membocorkan path, nama field, atau data subjek.
INTERNAL_ERROR_MESSAGE = (
    "Terjadi kesalahan tak terduga di server. Permintaan tidak diproses. "
    "Coba lagi, atau hubungi pengelola kalau berulang."
)


def internal_error_body() -> dict:
    """Bentuk 500 yang aman: tanpa stack trace, tanpa pesan exception."""
    return {
        "error": {
            "code": STATUS_CODES[500],
            "message": INTERNAL_ERROR_MESSAGE,
        }
    }


def install_error_handlers(app: FastAPI) -> None:
    """Pasang handler error seragam. Dipanggil sekali dari main.py.

    Menimpa handler bawaan FastAPI untuk HTTPException dan
    RequestValidationError; status code dan `detail` tidak berubah.
    Menambah handler untuk error tak terduga (belum pernah ada).
    """

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.status_code, exc.detail),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        # jsonable_encoder: field `input`/`ctx` bisa berisi bytes atau objek
        # yang tidak serializable -- persis seperti handler bawaan FastAPI.
        detail = jsonable_encoder(exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": VALIDATION_CODE,
                    "message": error_message(exc.errors()),
                },
                "detail": detail,
            },
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        # Traceback hanya untuk log server (di sini), TIDAK pernah ke klien.
        logger.exception(
            "500 tak terduga: %s %s",
            request.method,
            request.url.path,
        )
        return JSONResponse(status_code=500, content=internal_error_body())
