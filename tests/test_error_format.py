"""
test_error_format.py
--------------------
Format error seragam: {"error": {"code": str, "message": str}}.

Handler di app/errors.py menangani dua kasus:
  1. HTTPException apa pun (400/413/415/422/501/503/...)
  2. validasi input FastAPI (RequestValidationError -> 422)

Field `detail` lama sengaja TETAP dikirim supaya struktur error seragam tidak
mempreteli detail FastAPI; test di file ini juga menguncinya.

CATATAN PERUBAHAN KONTRAK (Task F): endpoint `/api/v1/analyze-face` diganti
`/api/v1/screening/analyze`. Dua konsekuensi untuk file ini:
    * field upload `image` -> `foto`, dan permintaan butuh data antropometri
    (kategori/lila_cm/berat_kg) supaya validasi Form berjalan;
  * penolakan quality gate BUKAN lagi 422 -- foto tidak layak kini 200, jadi
    test 422 quality gate diganti test "tidak menambah field error".
"""

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

import numpy as np  # noqa: E402

from app import content, pipeline  # noqa: E402
from app.errors import (  # noqa: E402
    VALIDATION_CODE, error_body, error_code, error_message,
)
from app.main import app  # noqa: E402
from tests.conftest import render_frontal_face  # noqa: E402

ENDPOINT = "/api/v1/screening/analyze"
FORM_IBU = {"kategori": "ibu_hamil", "lila_cm": "23.0", "berat_kg": "55.0"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _png(img_bgr) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", np.asarray(img_bgr))
    assert ok
    return buf.tobytes()


def _post(client, nama="face.png", payload=None, tipe="image/png", data=None):
    return client.post(
        ENDPOINT,
        data=FORM_IBU if data is None else data,
        files={"foto": (nama, payload if payload is not None
                        else _png(np.zeros((64, 64, 3), np.uint8)), tipe)},
    )


def _assert_uniform(body: dict) -> dict:
    """Bentuk wajib: error berisi tepat code + message, keduanya string."""
    assert "error" in body, f"response error tanpa field error: {body}"
    err = body["error"]
    assert set(err) == {"code", "message"}, f"field error tidak seragam: {set(err)}"
    assert isinstance(err["code"], str) and err["code"], "code harus string terisi"
    assert isinstance(err["message"], str) and err["message"], \
        "message harus string terisi"
    return err


# ---------------------------------------------------------------------------
# HTTPException
# ---------------------------------------------------------------------------

def test_http_exception_has_uniform_error_shape(client):
    r = _post(client, nama="a.txt", payload=b"hello", tipe="text/plain")
    assert r.status_code == 415
    err = _assert_uniform(r.json())
    assert err["code"] == "unsupported_media_type"
    assert "tidak didukung" in err["message"]


def test_legacy_detail_is_kept_beside_error(client):
    """`detail` tetap ada, dan tipenya tetap seperti semula (string)."""
    r = _post(client, nama="a.txt", payload=b"hello", tipe="text/plain")
    body = r.json()
    assert "detail" in body, "detail lama hilang -> klien lama pecah"
    assert body["detail"] == (
        "Content-Type 'text/plain' tidak didukung. "
        "Yang diterima: image/jpeg, image/jpg, image/png."
    )
    assert body["error"]["message"] == body["detail"]


def test_invalid_image_bytes_use_400_bad_request(client):
    r = _post(client, nama="fake.png",
              payload=b"\x89PNG\r\n\x1a\n broken", tipe="image/png")
    assert r.status_code == 400
    err = _assert_uniform(r.json())
    assert err["code"] == "bad_request"
    assert "gambar" in err["message"].lower()


def test_oversized_upload_uses_413_payload_too_large(client):
    import numpy as np

    from app.settings import get_settings

    payload = np.zeros((1, int(get_settings().max_upload_bytes * 1.1)),
                       dtype=np.uint8).tobytes()
    r = _post(client, nama="big.png", payload=payload, tipe="image/png")
    assert r.status_code == 413
    err = _assert_uniform(r.json())
    assert err["code"] == "payload_too_large"
    assert "terlalu besar" in err["message"].lower()


def test_quality_rejection_is_200_without_error_field(client, monkeypatch):
    """Foto tidak layak bukan error HTTP lagi (dulu 422 + reason/hint).

    Sekarang: 200, pesan ada di `kualitas_foto.pesan`, dan TIDAK ada field
    `error` maupun `detail` (karena ini bukan kegagalan permintaan).
    """
    def _reject(*a, **k):
        raise pipeline.ImageRejectedError(
            "Kepala tidak menghadap kamera lurus.",
            "yaw_proxy=0.41 (maks 0.3). Minta subjek menghadap lurus ke kamera.",
        )

    monkeypatch.setattr(pipeline, "detect_landmarks",
                        lambda img: (np.zeros((478, 3)), 1))
    monkeypatch.setattr(pipeline, "sample_skin_lab",
                        lambda img, lm: np.array([60.0, 12.0, 18.0]))
    monkeypatch.setattr(pipeline, "assess_quality", _reject)

    r = _post(client, payload=_png(render_frontal_face()))
    assert r.status_code == 200, r.text
    body = r.json()
    assert "error" not in body and "detail" not in body
    assert body["kualitas_foto"]["pesan"] == [content.get_text("foto.pose_miring")]


def test_model_missing_uses_503_service_unavailable(client, monkeypatch):
    def _missing(_img):
        raise pipeline.ModelMissingError("model di /home/user/secret/x.task")

    monkeypatch.setattr(pipeline, "detect_landmarks", _missing)
    r = _post(client, payload=_png(render_frontal_face()))
    assert r.status_code == 503, r.text
    err = _assert_uniform(r.json())
    assert err["code"] == "service_unavailable"
    assert "/home/user" not in err["message"]
    assert "download_model.py" in err["message"]


# ---------------------------------------------------------------------------
# Validasi input (RequestValidationError)
# ---------------------------------------------------------------------------

def test_validation_error_has_uniform_error_shape(client):
    r = client.post(ENDPOINT)                        # tanpa field sama sekali
    assert r.status_code == 422
    err = _assert_uniform(r.json())
    assert err["code"] == VALIDATION_CODE
    assert "foto" in err["message"]


def test_validation_error_keeps_fastapi_detail_list(client):
    """Detail validasi FastAPI (list of dict) tidak boleh dipreteli."""
    detail = client.post(ENDPOINT).json()["detail"]
    assert isinstance(detail, list) and detail
    assert {"type", "loc", "msg", "input"} <= set(detail[0])


def test_unknown_route_uses_uniform_error_shape(client):
    r = client.get("/api/v1/tidak-ada")
    assert r.status_code == 404
    err = _assert_uniform(r.json())
    assert err["code"] == "not_found"


# ---------------------------------------------------------------------------
# Error tak terduga (500)
# ---------------------------------------------------------------------------

def _boom(_img_bgr):
    raise RuntimeError("RAHASIA-subjek-42 path=/home/astria/data/foto.jpg")


@pytest.fixture()
def client_no_raise():
    """TestClient yang mengembalikan respons 500 alih-alih melempar ulang.

    Starlette sengaja melempar lagi exception yang sudah ditangani handler
    (supaya server tetap bisa mencatatnya di log). Karena itu test 500 harus
    mematikan perilaku itu.
    """
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_unexpected_error_returns_uniform_500(client_no_raise, monkeypatch):
    monkeypatch.setattr(pipeline, "detect_landmarks", _boom)
    r = _post(client_no_raise, payload=_png(render_frontal_face()))
    assert r.status_code == 500, r.text
    err = _assert_uniform(r.json())
    assert err["code"] == "internal_error"
    assert err["message"]


def test_unexpected_error_never_leaks_internals(client_no_raise, monkeypatch):
    """Stack trace, pesan exception, path file, dan data subjek TIDAK boleh
    sampai ke klien."""
    monkeypatch.setattr(pipeline, "detect_landmarks", _boom)
    r = _post(client_no_raise, payload=_png(render_frontal_face()))
    text = r.text
    for forbidden in (
        "RAHASIA-subjek-42",      # pesan exception
        "Traceback",              # isi traceback
        'File "',                 # path file source
        "/home/astria",           # path absolut server
        "RuntimeError",           # tipe exception
        "detect_landmarks",       # nama internal fungsi
    ):
        assert forbidden not in text, f"membocorkan {forbidden!r}: {text[:400]}"


def test_unexpected_error_is_logged_server_side(caplog, monkeypatch):
    """Traceback-nya HARUS ada di log server -- hanya yang ke klien yang
    dilarang."""
    monkeypatch.setattr(pipeline, "detect_landmarks", _boom)
    with TestClient(app, raise_server_exceptions=False) as c:
        with caplog.at_level("ERROR", logger="app.errors"):
            _post(c, payload=_png(render_frontal_face()))
    # caplog.text sudah termasuk traceback yang diformat (exc_info),
    # sedangkan getMessage() tidak.
    logged = caplog.text
    assert "RAHASIA-subjek-42" in logged, logged[:600]
    assert "Traceback" in logged, logged[:600]


# ---------------------------------------------------------------------------
# Unit: penyusun bentuk error
# ---------------------------------------------------------------------------

def test_error_code_honours_override_in_detail():
    """Endpoint boleh menentukan code sendiri lewat dict detail."""
    assert error_code(422, {"code": "quality_rejected"}) == "quality_rejected"
    assert error_code(422, {"reason": "x"}) == "unprocessable_entity"
    assert error_code(499, None) == "http_499"


def test_error_code_has_not_implemented_slug():
    assert error_code(501, "teks apa saja") == "not_implemented"


def test_error_message_flattens_reason_and_hint():
    assert error_message({"reason": "alasan", "hint": "saran"}) == "alasan — saran"
    assert error_message("pesan polos") == "pesan polos"
    assert error_message(None) == ""


def test_error_body_omits_detail_when_absent():
    body = error_body(404, None)
    assert body == {"error": {"code": "not_found", "message": ""}}
    assert "detail" not in body
