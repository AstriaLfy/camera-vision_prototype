"""
test_content.py
---------------
Kontrak app/content.py: satu-satunya sumber teks yang tampil ke pengguna.

Yang dijaga:
  * setiap status punya label dan punya rekomendasi untuk KEDUA kelompok;
  * kunci hilang = error yang jelas, bukan string kosong;
  * teks konten tidak ditulis literal lagi di endpoint;
  * jumlah rekomendasi yang masih placeholder DIKABARKAN (bukan gagal).
"""

import warnings
from pathlib import Path

import pytest

from app import content
from app.content import (
    DISCLAIMER_PRIVASI,
    DISCLAIMER_UMUM,
    DISCLAIMER_VISUAL,
    ContentKeyError,
    FOTO_TIDAK_LAYAK,
    KELOMPOK,
    LABEL_KARTU_VISUAL,
    STATUS,
    STATUS_LABELS,
    get_recommendation,
    get_text,
)

APP_DIR = Path(__file__).resolve().parents[1] / "app"

#: Kondisi foto tidak layak yang wajib ada (item 7).
#: `beberapa_wajah` & `tidak_layak` ditambahkan di Task F: pipeline bisa
#: menolak karena >1 wajah, dan ada alasan penolakan yang tidak punya kategori
#: spesifik (fallback generik).
KONDISI_FOTO = (
    "buram",
    "wajah_tidak_terdeteksi",
    "pencahayaan_kurang",
    "jarak_terlalu_jauh",
    "jarak_terlalu_dekat",
    "pose_miring",
    "beberapa_wajah",
    "tidak_layak",
)


# ---------------------------------------------------------------------------
# Item 1-4: disclaimer & label kartu
# ---------------------------------------------------------------------------

def test_disclaimer_wajib_ada():
    assert get_text("disclaimer.umum") == DISCLAIMER_UMUM
    assert get_text("disclaimer.visual") == DISCLAIMER_VISUAL
    assert get_text("disclaimer.privasi") == DISCLAIMER_PRIVASI
    for key in ("disclaimer.umum", "disclaimer.visual", "disclaimer.privasi"):
        value = get_text(key)
        assert isinstance(value, str) and value.strip(), key


def test_label_kartu_visual_ada():
    assert get_text("label.kartu_visual") == LABEL_KARTU_VISUAL
    assert LABEL_KARTU_VISUAL.strip()


# ---------------------------------------------------------------------------
# Item 5: label status
# ---------------------------------------------------------------------------

def test_daftar_status_sesuai_permintaan():
    assert STATUS == (
        "hijau", "kuning", "merah", "tidak_dapat_ditentukan",
    )


@pytest.mark.parametrize("status", STATUS)
def test_setiap_status_punya_label(status):
    label = get_text(f"status.{status}")
    assert isinstance(label, str)
    assert label.strip(), f"label status {status!r} kosong"
    assert label == STATUS_LABELS[status]


# ---------------------------------------------------------------------------
# Item 6: rekomendasi per status x kelompok
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", STATUS)
@pytest.mark.parametrize("kelompok", KELOMPOK)
def test_setiap_status_punya_rekomendasi_untuk_dua_kelompok(status, kelompok):
    teks = get_recommendation(status, kelompok)
    assert isinstance(teks, str)
    assert teks.strip(), f"rekomendasi {kelompok}/{status} kosong"


def test_kelompok_hanya_balita_dan_ibu():
    assert KELOMPOK == ("balita", "ibu")


# ---------------------------------------------------------------------------
# Item 7: pesan foto tidak layak
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kondisi", KONDISI_FOTO)
def test_pesan_foto_tidak_layak_ada(kondisi):
    teks = get_text(f"foto.{kondisi}")
    assert teks.strip(), f"pesan foto.{kondisi} kosong"
    assert teks == FOTO_TIDAK_LAYAK[kondisi]


def test_kondisi_foto_tidak_layak_lengkap():
    assert set(FOTO_TIDAK_LAYAK) == set(KONDISI_FOTO)


# ---------------------------------------------------------------------------
# Task F: deskripsi ringkasan per status & pesan mode model
# ---------------------------------------------------------------------------

def test_setiap_status_punya_deskripsi_ringkasan():
    from app.content import RINGKASAN_DESKRIPSI

    assert set(RINGKASAN_DESKRIPSI) == set(STATUS)
    for status in STATUS:
        assert get_text(f"ringkasan.deskripsi.{status}") == RINGKASAN_DESKRIPSI[status]
        assert RINGKASAN_DESKRIPSI[status].strip(), status


def test_pesan_mode_model_ada():
    teks = get_text("endpoint.cv_mode_model")
    assert teks.strip()
    assert "indikator" in teks.lower() or "model" in teks.lower()


# ---------------------------------------------------------------------------
# Kunci hilang harus error yang jelas
# ---------------------------------------------------------------------------

def test_get_text_kunci_hilang_melempar_error():
    with pytest.raises(ContentKeyError) as excinfo:
        get_text("status.oranye")
    pesan = str(excinfo.value)
    assert "status.oranye" in pesan
    assert "tersedia" in pesan, "pesan harus menyebut kunci yang tersedia"


def test_get_text_kunci_kosong_melempar_error():
    with pytest.raises(ContentKeyError):
        get_text("")


def test_get_text_tipe_salah_melempar_error():
    with pytest.raises(ContentKeyError):
        get_text(None)          # type: ignore[arg-type]


def test_error_kunci_hilang_bukan_string_kosong():
    """Syarat: melempar, bukan mengembalikan ''."""
    hasil = []
    try:
        hasil.append(get_text("tidak_ada"))
    except ContentKeyError:
        pass
    assert hasil == [], "get_text harus melempar, jangan mengembalikan nilai"


def test_content_key_error_adalah_keyerror():
    """Klien lama yang menangkap KeyError tetap kena."""
    assert issubclass(ContentKeyError, KeyError)
    with pytest.raises(KeyError):
        get_text("tidak_ada")


def test_get_recommendation_status_tidak_dikenal():
    with pytest.raises(ContentKeyError) as excinfo:
        get_recommendation("biru", "balita")
    pesan = str(excinfo.value)
    assert "biru" in pesan
    assert "Status yang tersedia" in pesan


def test_get_recommendation_kelompok_tidak_dikenal():
    with pytest.raises(ContentKeyError) as excinfo:
        get_recommendation("merah", "lansia")
    pesan = str(excinfo.value)
    assert "lansia" in pesan
    assert "Kelompok yang tersedia" in pesan


def test_get_recommendation_keduanya_salah_melempar():
    with pytest.raises(ContentKeyError):
        get_recommendation("biru", "lansia")


# ---------------------------------------------------------------------------
# Satu-satunya tempat teks: tidak ada literal konten di endpoint
# ---------------------------------------------------------------------------

def test_teks_konten_tidak_dihardcode_di_endpoint():
    """Baris literal item 1-4 tidak boleh ada di app/routers/*.py."""
    teks_wajib_di_content = (
        DISCLAIMER_UMUM,
        DISCLAIMER_VISUAL,
        DISCLAIMER_PRIVASI,
        LABEL_KARTU_VISUAL,
    )
    for path in (APP_DIR / "routers").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for teks in teks_wajib_di_content:
            assert teks not in source, (
                f"teks {teks[:40]!r} hardcode di {path.name}; pindah ke app/content.py"
            )


def test_endpoint_membaca_disclaimer_dari_content():
    """Nilai respons harus persis apa yang ada di content (tanpa duplikasi)."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        root = client.get("/").json()
        assert root["disclaimer"] == get_text("endpoint.disclaimer_akar")


# ---------------------------------------------------------------------------
# Pelapor placeholder (LAPORAN, bukan kegagalan)
# ---------------------------------------------------------------------------

def test_laporkan_rekomendasi_masih_placeholder():
    """Menghitung rekomendasi yang masih placeholder dan MELAPORKANNYA.

    Sengaja TIDAK assert terhadap jumlahnya: saat ahli gizi sudah mengisi
    teks asli, test ini tetap lulus -- yang berubah hanya isi laporannya.
    """
    total = 0
    placeholder = 0
    for kelompok in KELOMPOK:
        for status in STATUS:
            total += 1
            teks = get_recommendation(status, kelompok)
            assert teks.strip(), f"{kelompok}/{status} kosong"   # struktural
            if content.PENANDA_PLACEHOLDER in teks:
                placeholder += 1

    assert total == len(STATUS) * len(KELOMPOK)

    pesan = (
        f"[content] {placeholder}/{total} rekomendasi masih placeholder "
        f"{content.PENANDA_PLACEHOLDER} -- menunggu isi dari pedoman "
        "Kemenkes/WHO."
    )
    warnings.warn(pesan, stacklevel=1)
    print(f"\n{pesan}")
