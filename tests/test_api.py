"""
test_api.py
-----------
Test endpoint HTTP dasar: root/OpenAPI, validasi upload, pemetaan error ke
status code, dan helper dekode (EXIF + downscale).

Kontrak endpoint BERUBAH di Task F: `/api/v1/analyze-face` diganti
`/api/v1/screening/analyze` (multipart `foto` + data antropometri). Perubahan
yang memengaruhi file ini:

  * foto tidak layak / tanpa wajah  -> HTTP 200 (dulu 422), dengan
    `kualitas_foto.ok=false` dan `data_pendukung=[]`;
  * penolakan quality gate tidak lagi mengembalikan `reason`+`hint`, tapi
    pesan di `kualitas_foto.pesan` (teks dari app/content.py);
  * respons tidak lagi punya `status_cv`/`score`/`indicators`;
  * nama field upload `image` -> `foto`.

Test kontrak respons yang panjang ada di tests/test_screening.py; di sini fokus
ke jalur HTTP dan helper.
"""

import io

import cv2
import numpy as np
import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from app import content, pipeline  # noqa: E402
from app.main import app, create_app  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402
from tests.conftest import requires_model, render_frontal_face  # noqa: E402

ENDPOINT = "/api/v1/screening/analyze"

#: Data antropometri minimal supaya validasi Form lolos.
FORM_IBU = {"kategori": "ibu_hamil", "lila_cm": "23.0", "berat_kg": "55.0"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _png_bytes(img_bgr) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    assert ok
    return buf.tobytes()


def _wall_image(h=480, w=640) -> np.ndarray:
    """Gambar tanpa wajah sama sekali."""
    rng = np.random.default_rng(3)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _post(client, data=None, nama="f.png", payload=None, tipe="image/png"):
    return client.post(
        ENDPOINT,
        data=data if data is not None else FORM_IBU,
        files={"foto": (nama, payload if payload is not None
                        else _png_bytes(np.zeros((64, 64, 3), np.uint8)),
                        tipe)},
    )


# ---------------------------------------------------------------------------
# Endpoint dasar
# ---------------------------------------------------------------------------

def test_root_exposes_endpoint_and_disclaimer(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["endpoint"] == ENDPOINT
    assert "bukan diagnosis" in body["disclaimer"].lower()


def test_openapi_documents_multipart_upload(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    op = spec["paths"][ENDPOINT]["post"]
    content_ = op["requestBody"]["content"]
    assert "multipart/form-data" in content_

    schema = content_["multipart/form-data"]["schema"]
    if "$ref" in schema:
        schema = spec["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]]
    assert "foto" in schema["properties"]
    assert "lila_cm" in schema["properties"]


# ---------------------------------------------------------------------------
# Validasi upload
# ---------------------------------------------------------------------------

def test_missing_file_returns_422(client):
    r = client.post("/api/v1/screening/analyze")
    assert r.status_code == 422


def test_unsupported_content_type_returns_415(client):
    r = _post(client, nama="notimage.txt", payload=b"hello world",
              tipe="text/plain")
    assert r.status_code == 415
    assert "tidak didukung" in r.json()["detail"]


def test_non_image_bytes_return_400(client):
    r = _post(client, nama="fake.png",
              payload=b"\x89PNG\r\n\x1a\n broken bytes", tipe="image/png")
    assert r.status_code == 400
    assert "gambar" in r.json()["detail"].lower()


def test_oversized_upload_returns_413(client):
    max_bytes = get_settings().max_upload_bytes
    # Bytes acak besar; tidak harus gambar valid karena ukuran dicek lebih dulu.
    payload = np.zeros((1, int(max_bytes * 1.2)), dtype=np.uint8).tobytes()
    r = _post(client, nama="big.png", payload=payload, tipe="image/png")
    assert r.status_code == 413
    assert "terlalu besar" in r.json()["detail"].lower()


def test_image_without_face_returns_200_not_suitable(client):
    """Tanpa wajah bukan lagi 422: status LiLA tetap dikembalikan (200)."""
    r = _post(client, payload=_png_bytes(_wall_image()))
    assert r.status_code == 200
    body = r.json()
    assert body["kualitas_foto"]["ok"] is False
    assert body["data_pendukung"] == []


# ---------------------------------------------------------------------------
# Bentuk response / error
# ---------------------------------------------------------------------------

def test_quality_rejection_returns_200_with_message(client, monkeypatch):
    """Penolakan quality gate tidak lagi menggagalkan permintaan.

    Dulu endpoint mengembalikan 422 + reason/hint. Sekarang foto tidak layak
    menghasilkan 200, `kualitas_foto.ok=false`, dan pesan yang bisa dibaca
    pengguna (teks dari app/content.py). Exception pipeline di-inject lewat
    monkeypatch supaya kontrak HTTP bisa diuji tanpa foto 3/4 view.
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

    r = _post(client, payload=_png_bytes(render_frontal_face()))
    assert r.status_code == 200
    body = r.json()
    assert body["kualitas_foto"]["ok"] is False
    assert body["kualitas_foto"]["pesan"] == [
        content.get_text("foto.pose_miring")
    ]
    assert body["data_pendukung"] == []


def test_face_not_found_records_message(client, monkeypatch):
    def _no_face(_img):
        raise pipeline.FaceNotFoundError("Tidak ada wajah yang terdeteksi.")

    monkeypatch.setattr(pipeline, "detect_landmarks", _no_face)
    r = _post(client, payload=_png_bytes(render_frontal_face()))
    assert r.status_code == 200
    body = r.json()
    assert body["kualitas_foto"]["ok"] is False
    assert body["kualitas_foto"]["pesan"] == [
        content.get_text("foto.wajah_tidak_terdeteksi")
    ]


def test_missing_model_returns_503_without_leaking_paths(client, monkeypatch):
    """Model hilang = masalah server (503), dan respons tidak boleh membocorkan
    path filesystem."""
    def _missing(_img):
        raise pipeline.ModelMissingError("model di /home/user/secret/path/model.task")

    monkeypatch.setattr(pipeline, "detect_landmarks", _missing)
    r = _post(client, payload=_png_bytes(render_frontal_face()))
    assert r.status_code == 503
    detail = str(r.json()["detail"])
    assert "/home/user" not in detail and ".task" not in detail
    assert "download_model.py" in detail


@requires_model
def test_successful_analysis_response_shape(client):
    """Uji end-to-end 200 OK memakai wajah frontal hasil render."""
    img = render_frontal_face()
    try:
        pipeline.detect_landmarks(img)
    except pipeline.FaceNotFoundError:
        pytest.skip("wajah hasil render tidak terdeteksi")

    r = _post(client, payload=_png_bytes(img))
    assert r.status_code == 200, r.text
    body = r.json()

    # Field sesuai kontrak baru; tidak ada status_cv/skor.
    assert set(body) == {
        "ringkasan", "subjek", "pengukuran", "data_pendukung",
        "kualitas_foto", "rekomendasi", "alasan", "disclaimer", "meta",
    }
    assert "status_cv" not in body and "score" not in body
    from app import risk_status

    risk = risk_status.evaluate_risk(kategori="ibu_hamil", lila_cm=23.0,
                                     berat_kg=55.0)
    assert body["ringkasan"]["status"] == risk.status_akhir
    assert isinstance(body["pengukuran"], list) and body["pengukuran"]
    assert all(p["sumber"] == "terukur" for p in body["pengukuran"])


@requires_model
def test_debug_landmarks_returns_jpeg():
    # Endpoint debug hanya terdaftar bila diaktifkan (default mati), jadi app
    # dibuat khusus dengan ENABLE_DEBUG_ENDPOINTS=true.
    img = render_frontal_face()
    try:
        pipeline.detect_landmarks(img)
    except pipeline.FaceNotFoundError:
        pytest.skip("wajah hasil render tidak terdeteksi")

    with TestClient(create_app(Settings(enable_debug_endpoints=True))) as dbg:
        r = dbg.post(
            "/api/v1/debug-landmarks",
            files={"image": ("face.png", _png_bytes(img), "image/png")},
        )
    if r.status_code == 422:
        pytest.skip(f"debug endpoint menolak fixture: {r.json()['detail']}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/jpeg")
    decoded = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None, "response harus gambar yang bisa didekode"
    assert decoded.shape[:2] == img.shape[:2]


# ---------------------------------------------------------------------------
# EXIF orientation
# ---------------------------------------------------------------------------

def _jpeg_with_orientation(img_bgr, orientation: int) -> bytes:
    """Buat JPEG dengan tag EXIF Orientation tertentu."""
    from PIL import Image

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    exif = im.getexif()
    exif[274] = orientation
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


@pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
def test_exif_orientation_1_to_8_produces_expected_geometry(orientation):
    """Semua 8 nilai EXIF harus ditangani.

    Versi lama memetakan 2/3/4/5/6/8 secara manual dan TIDAK menangani 7, jadi
    foto berorientasi transverse diproses dalam orientasi yang salah tanpa error.
    """
    from PIL import Image, ImageOps
    from app.main import _apply_exif_orientation

    # gambar asimetris supaya setiap orientasi bisa dibedakan
    img = np.zeros((60, 100, 3), np.uint8)
    img[:10, :20] = (255, 0, 0)      # kiri-atas
    img[-10:, -20:] = (0, 255, 0)    # kanan-bawah

    contents = _jpeg_with_orientation(img, orientation)
    raw = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    fixed = _apply_exif_orientation(contents, raw)

    # acuan: Pillow yang melakukan transpose
    with Image.open(io.BytesIO(contents)) as im:
        expected = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    expected_bgr = cv2.cvtColor(expected, cv2.COLOR_RGB2BGR)

    assert fixed.shape == expected_bgr.shape, (
        f"orientation {orientation}: bentuk {fixed.shape} != {expected_bgr.shape}"
    )
    assert np.array_equal(fixed, expected_bgr), f"orientation {orientation} tidak cocok"


def test_exif_orientation_none_is_passthrough():
    from PIL import Image
    from app.main import _apply_exif_orientation

    img = np.zeros((40, 50, 3), np.uint8)
    img[5:10, 5:10] = (10, 20, 30)
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")

    raw = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    fixed = _apply_exif_orientation(buf.getvalue(), raw)
    assert np.array_equal(fixed, raw), "tanpa tag EXIF, gambar harus tidak berubah"


# ---------------------------------------------------------------------------
# Downscale
# ---------------------------------------------------------------------------

def test_large_image_is_downscaled(client):
    """Foto besar harus di-downscale, bukan diproses ukuran penuh."""
    big = np.random.default_rng(1).integers(
        0, 255, (2600, 3400, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", big)
    assert ok
    from app.main import _decode_upload

    out = _decode_upload(buf.tobytes(), "image/jpeg")
    assert max(out.shape[:2]) <= get_settings().max_dimension_px
