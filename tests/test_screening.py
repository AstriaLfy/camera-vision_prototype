"""
test_screening.py
-----------------
POST /api/v1/screening/analyze: struktur respons, aturan LiLA vs foto,
mode CV, privasi (stateless + tanpa log data subjek), dan validasi input.

Kebanyakan test mem-monkeypatch fungsi pipeline supaya tidak bergantung pada
model MediaPipe; test end-to-end yang butuh model ditandai @requires_model.

Separation of concerns yang dijaga test ini:
  * status MURNI dari LiLA -- foto tidak memengaruhi ringkasan;
  * status_cv / skor pipeline TIDAK pernah dipakai atau muncul di respons;
  * item berbadge HANYA boleh berasal dari pengukuran ("terukur");
  * foto tidak layak -> HTTP 200, kualitas_foto.ok=false, data_pendukung=[].
"""

from __future__ import annotations

import builtins
import io
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from app import content, indicators, pipeline, risk_status  # noqa: E402
from app.main import app  # noqa: E402
from app.routers import screening  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402
from tests.conftest import requires_model, render_frontal_face  # noqa: E402

ENDPOINT = "/api/v1/screening/analyze"

#: Urutan field respons yang dikunci kontrak.
URUTAN_FIELD = [
    "ringkasan", "subjek", "pengukuran", "data_pendukung",
    "kualitas_foto", "rekomendasi", "alasan", "disclaimer", "meta",
]

#: Token internal pipeline yang tidak boleh bocor ke respons.
TERLARANG_BOCOR = (
    "status_cv", "normalized", "flagged", "is_flagged",
    "combine_status", "process_face_photo", "score", "skor", "tidak_valid",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _png(img_bgr) -> bytes:
    ok, buf = cv2.imencode(".png", np.asarray(img_bgr))
    assert ok
    return buf.tobytes()


def _wall_image(h=160, w=160) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _data_ibu(**ubah) -> dict:
    data = {"kategori": "ibu_hamil", "lila_cm": "22.0", "berat_kg": "55.0"}
    data.update(ubah)
    return data


def _data_balita(**ubah) -> dict:
    data = {"kategori": "balita", "lila_cm": "13.0", "berat_kg": "9.0",
            "usia_bulan": "24"}
    data.update(ubah)
    return data


def _post(client, data, img=None, nama="f.png", tipe="image/png"):
    return client.post(
        ENDPOINT, data=data,
        files={"foto": (nama, _png(img if img is not None
                                    else np.zeros((64, 64, 3), np.uint8)), tipe)},
    )


def _stub_pipeline_ok(monkeypatch):
    """Pipeline palsu yang selalu lolos gate dan menghasilkan fitur."""
    monkeypatch.setattr(pipeline, "detect_landmarks",
                        lambda img: (np.zeros((478, 3)), 1))
    monkeypatch.setattr(pipeline, "sample_skin_lab",
                        lambda img, lm: np.array([60.0, 12.0, 18.0]))
    monkeypatch.setattr(pipeline, "assess_quality",
                        lambda *a, **k: pipeline.QualityReport(ok=True))
    monkeypatch.setattr(pipeline, "extract_geometry_features", lambda lm: {
        "available": True, "cheek_width_px": 100.0, "jaw_width_px": 80.0,
        "cheek_to_jaw_ratio": 1.25, "lower_face_height_ratio": 0.40,
    })
    monkeypatch.setattr(pipeline, "extract_hair_features", lambda *a, **k: {
        "available": True, "hair_l": 18.0, "hair_chroma": 7.0,
        "l_delta": 42.0, "chroma_ratio": 0.50,
    })
    monkeypatch.setattr(pipeline, "extract_sclera_features", lambda *a, **k: {
        "available": True, "sclera_a_star": -1.0, "a_star_ratio": -0.08,
    })


def _stub_tolak(monkeypatch, reason: str):
    monkeypatch.setattr(pipeline, "detect_landmarks",
                        lambda img: (np.zeros((478, 3)), 1))
    monkeypatch.setattr(pipeline, "sample_skin_lab",
                        lambda img, lm: np.array([60.0, 12.0, 18.0]))

    def _raises(*a, **k):
        raise pipeline.ImageRejectedError(reason, "detail teknis")

    monkeypatch.setattr(pipeline, "assess_quality", _raises)


def _set_mode(monkeypatch, mode: str):
    monkeypatch.setattr(screening, "get_settings",
                        lambda: Settings(cv_mode=mode))


# ---------------------------------------------------------------------------
# Struktur respons & urutan field (Pydantic model)
# ---------------------------------------------------------------------------

def test_success_response_has_exact_field_order(client, monkeypatch):
    _stub_pipeline_ok(monkeypatch)
    r = _post(client, _data_ibu())
    assert r.status_code == 200, r.text
    assert list(r.json()) == URUTAN_FIELD


def test_sub_structure_matches_models(client, monkeypatch):
    _stub_pipeline_ok(monkeypatch)
    body = _post(client, _data_ibu()).json()

    assert list(body["ringkasan"]) == ["status", "label", "deskripsi"]
    assert list(body["subjek"]) == ["kategori", "kelompok"]
    assert list(body["kualitas_foto"]) == ["ok", "pesan"]
    assert list(body["meta"]) == ["cv_mode", "versi"]
    assert list(body["pengukuran"][0]) == [
        "kunci", "label", "nilai", "satuan", "badge", "keterangan", "sumber",
    ]
    assert list(body["data_pendukung"][0]) == [
        "kunci", "kelompok", "label", "nilai", "satuan", "keterangan",
        "sumber", "badge", "badge_label",
    ]


def test_openapi_documents_new_endpoint(client):
    spec = client.get("/openapi.json").json()
    assert ENDPOINT in spec["paths"]
    op = spec["paths"][ENDPOINT]["post"]
    assert "multipart/form-data" in op["requestBody"]["content"]
    assert "/api/v1/analyze-face" not in spec["paths"]


def test_root_points_to_new_endpoint(client):
    assert client.get("/").json()["endpoint"] == ENDPOINT


def test_meta_berisi_mode_dan_versi(client, monkeypatch):
    from app import __version__

    _stub_pipeline_ok(monkeypatch)
    meta = _post(client, _data_ibu()).json()["meta"]
    assert meta == {"cv_mode": get_settings().cv_mode, "versi": __version__}


# ---------------------------------------------------------------------------
# Status murni LiLA (foto tidak memengaruhi)
# ---------------------------------------------------------------------------

def test_status_murni_lila_dan_label_dari_content(client, monkeypatch):
    _stub_pipeline_ok(monkeypatch)
    body = _post(client, _data_ibu()).json()
    risk = risk_status.evaluate_risk(
        kategori="ibu_hamil", lila_cm=22.0, berat_kg=55.0
    )
    assert body["ringkasan"]["status"] == risk.status_akhir == "merah"
    assert body["ringkasan"]["label"] == risk.label_status
    assert body["ringkasan"]["deskripsi"] == content.get_text(
        "ringkasan.deskripsi.merah"
    )
    assert body["subjek"] == {"kategori": "ibu_hamil", "kelompok": "ibu"}


def test_status_sama_baik_foto_layak_maupun_tidak(client, monkeypatch):
    _stub_pipeline_ok(monkeypatch)
    layak = _post(client, _data_ibu()).json()
    _stub_tolak(monkeypatch, "Foto terlalu buram.")
    buram = _post(client, _data_ibu()).json()
    assert layak["ringkasan"] == buram["ringkasan"]
    assert layak["pengukuran"] == buram["pengukuran"]


def test_status_tidak_dapat_ditentukan_http_200(client, monkeypatch):
    _stub_pipeline_ok(monkeypatch)
    r = _post(client, _data_ibu(lila_cm="3.0"))       # di luar rentang wajar
    assert r.status_code == 200
    body = r.json()
    assert body["ringkasan"]["status"] == "tidak_dapat_ditentukan"
    assert body["ringkasan"]["deskripsi"] == content.get_text(
        "ringkasan.deskripsi.tidak_dapat_ditentukan"
    )


def test_kategori_balita_pakai_istilah_balita(client, monkeypatch):
    _stub_pipeline_ok(monkeypatch)
    body = _post(client, _data_balita()).json()
    assert body["subjek"] == {"kategori": "balita", "kelompok": "balita"}
    assert body["ringkasan"]["status"] == "hijau"


# ---------------------------------------------------------------------------
# Pengukuran & data pendukung
# ---------------------------------------------------------------------------

def test_pengukuran_dari_risk_status_sumber_terukur(client, monkeypatch):
    _stub_pipeline_ok(monkeypatch)
    body = _post(client, _data_ibu()).json()
    risk = risk_status.evaluate_risk(
        kategori="ibu_hamil", lila_cm=22.0, berat_kg=55.0
    )
    assert [p["kunci"] for p in body["pengukuran"]] == ["lila", "berat_badan"]
    for item, asli in zip(body["pengukuran"], risk.indikator_terukur):
        assert item["sumber"] == "terukur"
        assert item["label"] == asli.label
        assert item["nilai"] == asli.nilai
        assert item["badge"] == asli.badge


def test_item_berbadge_hanya_sumber_terukur(client, monkeypatch):
    _stub_pipeline_ok(monkeypatch)
    body = _post(client, _data_ibu()).json()
    semua = [*body["pengukuran"], *body["data_pendukung"]]
    berbadge = [i for i in semua if i.get("badge") is not None]
    assert berbadge, "harus ada minimal satu item berbadge (lila)"
    assert all(i["sumber"] == "terukur" for i in berbadge)
    assert all(i["badge"] is None for i in body["data_pendukung"])


def test_data_pendukung_semua_eksperimental(client, monkeypatch):
    _stub_pipeline_ok(monkeypatch)
    body = _post(client, _data_ibu()).json()
    assert len(body["data_pendukung"]) == len(indicators.PEMETAAN)
    for item in body["data_pendukung"]:
        assert item["sumber"] == "visual_eksperimental"
        assert item["badge"] is None
        assert item["badge_label"] == "Eksperimental"


# ---------------------------------------------------------------------------
# status_cv / skor tidak dipakai dan tidak bocor
# ---------------------------------------------------------------------------

def test_status_cv_tidak_bocor(client, monkeypatch):
    _stub_pipeline_ok(monkeypatch)
    body = _post(client, _data_ibu()).json()
    teks = json.dumps(body, ensure_ascii=False).lower()
    for token in TERLARANG_BOCOR:
        assert token not in teks, f"respons membocorkan {token!r}"


def test_scoring_pipeline_tidak_pernah_dipanggil(client, monkeypatch):
    """Kalau combine_status/score_* dipanggil, test ini meledak."""
    _stub_pipeline_ok(monkeypatch)

    def _boom(*a, **k):  # pragma: no cover - hanya jalan kalau kontrak rusak
        raise AssertionError("scoring pipeline dipanggil, seharusnya tidak")

    monkeypatch.setattr(pipeline, "combine_status", _boom)
    monkeypatch.setattr(pipeline, "process_face_photo", _boom)
    for nama in ("score_hair", "score_sclera", "score_geometry"):
        monkeypatch.setattr(pipeline, nama, _boom)

    r = _post(client, _data_ibu())
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Foto tidak layak -> 200
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("reason", "kunci"), [
    ("Foto terlalu buram.", "foto.buram"),
    ("Wajah terlalu kecil di frame.", "foto.jarak_terlalu_jauh"),
    ("Kepala tidak menghadap kamera lurus.", "foto.pose_miring"),
    ("Kepala miring.", "foto.pose_miring"),
    ("Kepala menengadah atau mendongak.", "foto.pose_miring"),
    ("Pencahayaan tidak sesuai protokol.", "foto.pencahayaan_kurang"),
    ("2 wajah terdeteksi dalam satu foto.", "foto.beberapa_wajah"),
])
def test_penolakan_quality_jadi_200_dengan_pesan_content(
    client, monkeypatch, reason, kunci
):
    _stub_tolak(monkeypatch, reason)
    r = _post(client, _data_ibu())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kualitas_foto"]["ok"] is False
    assert body["kualitas_foto"]["pesan"] == [content.get_text(kunci)]
    assert body["data_pendukung"] == []
    assert body["ringkasan"]["status"] == "merah"        # status tetap ada
    # disclaimer visual tidak ikut kalau tidak ada data pendukung
    assert body["disclaimer"] == [content.get_text("disclaimer.umum")]


def test_foto_buram_khusus(client, monkeypatch):
    """Kasus yang disebut eksplisit: foto buram."""
    _stub_tolak(monkeypatch, "Foto terlalu buram.")
    body = _post(client, _data_ibu()).json()
    assert body["kualitas_foto"] == {
        "ok": False, "pesan": [content.get_text("foto.buram")]
    }


def test_wajah_tidak_terdeteksi(client, monkeypatch):
    def _no_face(_img):
        raise pipeline.FaceNotFoundError("Tidak ada wajah.")

    monkeypatch.setattr(pipeline, "detect_landmarks", _no_face)
    body = _post(client, _data_ibu()).json()
    assert body["kualitas_foto"]["ok"] is False
    assert body["kualitas_foto"]["pesan"] == [
        content.get_text("foto.wajah_tidak_terdeteksi")
    ]
    assert body["data_pendukung"] == []


def test_pesan_penolakan_tidak_membocorkan_detail_teknis(client, monkeypatch):
    """Detail teknis pipeline (angka, path) tidak boleh jadi pesan pengguna."""
    _stub_tolak(monkeypatch, "Foto terlalu buram.")
    teks = json.dumps(_post(client, _data_ibu()).json())
    assert "detail teknis" not in teks
    assert "sharpness" not in teks


# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

def test_disclaimer_umum_selalu_visual_hanya_jika_ada_data(client, monkeypatch):
    _stub_pipeline_ok(monkeypatch)
    umum = content.get_text("disclaimer.umum")
    visual = content.get_text("disclaimer.visual")

    ada_data = _post(client, _data_ibu()).json()
    assert ada_data["disclaimer"] == [umum, visual]

    _stub_tolak(monkeypatch, "Foto terlalu buram.")
    tanpa_data = _post(client, _data_ibu()).json()
    assert tanpa_data["disclaimer"] == [umum]

    _set_mode(monkeypatch, "off")
    mode_off = _post(client, _data_ibu()).json()
    assert mode_off["disclaimer"] == [umum]


# ---------------------------------------------------------------------------
# Mode CV
# ---------------------------------------------------------------------------

def test_mode_off_tidak_menjalankan_pipeline(client, monkeypatch):
    _set_mode(monkeypatch, "off")

    def _boom(*a, **k):  # pragma: no cover
        raise AssertionError("pipeline jalan padahal CV_MODE=off")

    monkeypatch.setattr(pipeline, "detect_landmarks", _boom)
    r = _post(client, _data_ibu())
    assert r.status_code == 200
    body = r.json()
    assert body["data_pendukung"] == []
    assert body["kualitas_foto"] == {"ok": True, "pesan": []}
    assert body["meta"]["cv_mode"] == "off"


def test_mode_indicator_only_mengisi_data_pendukung(client, monkeypatch):
    _set_mode(monkeypatch, "indicator_only")
    _stub_pipeline_ok(monkeypatch)
    body = _post(client, _data_ibu()).json()
    assert len(body["data_pendukung"]) == len(indicators.PEMETAAN)
    assert body["meta"]["cv_mode"] == "indicator_only"


def test_mode_model_mengembalikan_501(client, monkeypatch):
    _set_mode(monkeypatch, "model")
    r = _post(client, _data_ibu())
    assert r.status_code == 501
    body = r.json()
    assert body["error"]["code"] == "not_implemented"
    assert body["error"]["message"] == content.get_text("endpoint.cv_mode_model")


# ---------------------------------------------------------------------------
# Validasi input (Pydantic) -> 422 seragam
# ---------------------------------------------------------------------------

def _assert_422_seragam(r):
    assert r.status_code == 422, r.text
    err = r.json()["error"]
    assert err["code"] == "validation_error"
    assert isinstance(err["message"], str) and err["message"]
    assert isinstance(r.json()["detail"], list) and r.json()["detail"]


@pytest.mark.parametrize("ubah", [
    {"lila_cm": "abc"},
    {"lila_cm": "-1"},
    {"berat_kg": "abc"},
    {"berat_kg": "0"},
    {"usia_bulan": "-3"},
    {"kategori": "lansia"},
    {"panjang_tinggi_cm": "xyz"},
])
def test_input_tidak_valid_422(client, monkeypatch, ubah):
    r = _post(client, _data_ibu(**ubah))
    _assert_422_seragam(r)


def test_field_wajib_hilang_422(client):
    _assert_422_seragam(client.post(
        ENDPOINT, files={"foto": ("f.png", _png(np.zeros((8, 8, 3), np.uint8)),
                                  "image/png")},
    ))


def test_kategori_hilang_422(client):
    data = _data_ibu()
    data.pop("kategori")
    _assert_422_seragam(_post(client, data))


def test_usia_bulan_dan_usia_opsional(client, monkeypatch):
    _stub_pipeline_ok(monkeypatch)
    # balita tanpa usia_bulan tetap 200 (status tidak dapat ditentukan)
    r = _post(client, {"kategori": "balita", "lila_cm": "13.0",
                       "berat_kg": "9.0"})
    assert r.status_code == 200
    assert r.json()["ringkasan"]["status"] == "tidak_dapat_ditentukan"
    # ibu dengan usia (tahun) diterima
    assert _post(client, _data_ibu(usia="28")).status_code == 200


# ---------------------------------------------------------------------------
# Batas ukuran & tipe file (settings)
# ---------------------------------------------------------------------------

def test_tipe_file_tidak_didukung_415(client):
    r = _post(client, _data_ibu(), nama="a.webp", tipe="image/webp")
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_media_type"


def test_tipe_diizinkan_sesuai_settings(client, monkeypatch):
    # Setting hanya menerima webp -> jpeg ditolak di jalur decode.
    monkeypatch.setattr(
        screening, "get_settings",
        lambda: Settings(cv_mode="indicator_only",
                         allowed_upload_types=["image/webp"]),
    )
    r = _post(client, _data_ibu(), tipe="image/jpeg")
    assert r.status_code == 415


def test_upload_terlalu_besar_413(client, monkeypatch):
    batas = get_settings().max_upload_bytes
    payload = np.zeros(int(batas * 1.1), dtype=np.uint8).tobytes()
    r = client.post(
        ENDPOINT, data=_data_ibu(),
        files={"foto": ("big.png", payload, "image/png")},
    )
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "payload_too_large"


def test_bukan_gambar_400(client):
    r = client.post(
        ENDPOINT, data=_data_ibu(),
        files={"foto": ("x.png", b"\x89PNG\r\n\x1a\nrusak", "image/png")},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_request"


# ---------------------------------------------------------------------------
# Model tidak ada -> 503
# ---------------------------------------------------------------------------

def test_model_hilang_503_tanpa_bocor_path(client, monkeypatch):
    def _missing(_img):
        raise pipeline.ModelMissingError("model di /home/user/secret/model.task")

    monkeypatch.setattr(pipeline, "detect_landmarks", _missing)
    r = _post(client, _data_ibu())
    assert r.status_code == 503
    msg = r.json()["error"]["message"]
    assert "/home/user" not in msg and ".task" not in msg
    assert "download_model.py" in msg


# ---------------------------------------------------------------------------
# Privasi: stateless & tanpa log data subjek
# ---------------------------------------------------------------------------

def test_foto_tidak_ditulis_ke_disk(client, monkeypatch):
    """Selama request, tidak boleh ada file dibuka dengan mode tulis."""
    _stub_pipeline_ok(monkeypatch)
    dicatat: list[tuple[str, str]] = []
    asli_open = builtins.open

    def _open(file, mode="r", *args, **kwargs):
        if any(c in mode for c in "wax+"):
            dicatat.append((str(file), mode))
        return asli_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _open)
    monkeypatch.setattr(io, "open", _open)
    r = _post(client, _data_ibu(), nama="rahasia_subjek_42.png")
    assert r.status_code == 200
    assert dicatat == [], f"ada penulisan disk: {dicatat}"


def test_tidak_ada_data_subjek_di_log(client, monkeypatch, caplog):
    _stub_pipeline_ok(monkeypatch)
    with caplog.at_level(logging.DEBUG):
        r = _post(client, {"kategori": "ibu_hamil", "lila_cm": "22.555",
                           "berat_kg": "55.777"},
                  nama="RAHASIA_NAMA_SUBJEK.png")
    assert r.status_code == 200
    teks = caplog.text
    assert "22.555" not in teks
    assert "55.777" not in teks
    assert "RAHASIA_NAMA_SUBJEK" not in teks


# ---------------------------------------------------------------------------
# End-to-end dengan model asli
# ---------------------------------------------------------------------------

@requires_model
def test_end_to_end_dengan_model(client):
    img = render_frontal_face()
    r = _post(client, _data_balita(), img=img)
    assert r.status_code == 200, r.text
    body = r.json()
    assert list(body) == URUTAN_FIELD
    assert body["ringkasan"]["status"] in risk_status.STATUS_AKHIR
    if body["kualitas_foto"]["ok"]:
        assert body["data_pendukung"]
    else:
        assert body["data_pendukung"] == []
        assert body["kualitas_foto"]["pesan"]


@requires_model
def test_wall_image_tidak_layak_tapi_200(client):
    body = _post(client, _data_ibu(), img=_wall_image()).json()
    assert body["kualitas_foto"]["ok"] is False
    assert body["data_pendukung"] == []
    assert body["kualitas_foto"]["pesan"] == [
        content.get_text("foto.wajah_tidak_terdeteksi")
    ]
