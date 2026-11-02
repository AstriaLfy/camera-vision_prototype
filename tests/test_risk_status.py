"""Tests untuk app/risk_status.py (status murni LiLA + indikator terukur)."""

from __future__ import annotations

import io
import math
import tokenize
from pathlib import Path

import pytest

from app import content
from app import thresholds
from app.risk_status import (
    IndikatorTerukur,
    RiskStatus,
    evaluate_risk,
    LILA_BALITA_AMAN_CM,
    LILA_BALITA_MERAH_CM,
    LILA_IBU_AMAN_CM,
)
from scripts import extract_dataset as ed


def _token_kode(path: str) -> str:
    """Gabungan token yang BUKAN string dan BUKAN komentar."""
    src = Path(path).read_text(encoding="utf-8")
    return " ".join(
        tok.string
        for tok in tokenize.generate_tokens(io.StringIO(src).readline)
        if tok.type not in (tokenize.STRING, tokenize.COMMENT)
    )


def _indikator(r: RiskStatus, kunci: str) -> IndikatorTerukur:
    item = [i for i in r.indikator_terukur if i.kunci == kunci]
    assert len(item) == 1, f"indikator {kunci!r} muncul {len(item)}x"
    return item[0]


# ---------------------------------------------------------------------------
# Bentuk dasar output
# ---------------------------------------------------------------------------


def test_output_adalah_model_pydantic():
    r = evaluate_risk(kategori="balita", usia_bulan=36, lila_cm=13.0,
                      berat_kg=14.0)
    assert isinstance(r, RiskStatus)
    assert isinstance(r.indikator_terukur[0], IndikatorTerukur)


def test_field_model():
    assert list(RiskStatus.model_fields) == [
        "status_antropometri", "status_akhir", "kelompok", "rekomendasi",
        "label_status", "alasan", "disclaimer", "indikator_terukur",
    ]
    assert list(IndikatorTerukur.model_fields) == [
        "kunci", "label", "nilai", "satuan", "badge", "keterangan",
    ]


def test_adalah_keyword_only():
    with pytest.raises(TypeError):
        evaluate_risk("balita", 13.0, 14.0)  # type: ignore[misc]


def test_alasan_dan_disclaimer_bukan_kosong():
    for kat, extra in (("balita", {"usia_bulan": 36}), ("ibu_menyusui", {})):
        r = evaluate_risk(kategori=kat, lila_cm=13.0, berat_kg=14.0, **extra)
        assert r.alasan, f"{kat}: alasan kosong"
        assert r.disclaimer


def test_disclaimer_dari_content():
    r = evaluate_risk(kategori="ibu_hamil", lila_cm=25.0, berat_kg=55.0)
    assert r.disclaimer == content.get_text("disclaimer.umum")


# ---------------------------------------------------------------------------
# 1. Tidak ada cv_flag / peringatan_visual / hijau_pantau
# ---------------------------------------------------------------------------


def test_cv_flag_ditolak_sebagai_argumen():
    with pytest.raises(TypeError):
        evaluate_risk(kategori="balita", usia_bulan=36, lila_cm=13.0,
                      berat_kg=14.0, cv_flag=True)  # type: ignore[call-arg]


def test_model_tidak_punya_peringatan_visual():
    assert "peringatan_visual" not in RiskStatus.model_fields
    assert "cv_flag" not in evaluate_risk.__code__.co_varnames


def test_status_akhir_sama_dengan_status_antropometri():
    for kwargs in _SEMUA_KASUS():
        r = evaluate_risk(**kwargs)
        assert r.status_akhir == r.status_antropometri


def test_tidak_ada_status_hijau_pantau():
    for kwargs in _SEMUA_KASUS():
        r = evaluate_risk(**kwargs)
        assert r.status_akhir != "hijau_pantau"
        assert r.status_antropometri != "hijau_pantau"


def _SEMUA_KASUS() -> list[dict]:
    return [
        dict(kategori="balita", usia_bulan=36, lila_cm=13.0, berat_kg=14.0),
        dict(kategori="balita", usia_bulan=36, lila_cm=12.0, berat_kg=14.0),
        dict(kategori="balita", usia_bulan=36, lila_cm=11.0, berat_kg=14.0),
        dict(kategori="balita", lila_cm=13.0, berat_kg=14.0),
        dict(kategori="ibu_hamil", lila_cm=30.0, berat_kg=55.0),
        dict(kategori="ibu_hamil", lila_cm=20.0, berat_kg=55.0),
        dict(kategori="ibu_menyusui", lila_cm=30.0, berat_kg=55.0),
        dict(kategori="xxx", lila_cm=13.0, berat_kg=14.0),
        dict(kategori="balita", usia_bulan=24, lila_cm=60.0, berat_kg=14.0),
        dict(kategori="balita", usia_bulan=36, lila_cm=13.0,
             berat_kg=float("nan")),
    ]


def test_kode_modul_bebas_dari_cv():
    """cv_flag / peringatan_visual / hijau_pantau hanya di dokumentasi."""
    src = Path("app/risk_status.py").read_text(encoding="utf-8")
    assert "status_cv" in src          # wajib didokumentasikan sebagai larangan
    assert "cv_flag" in src
    kode = _token_kode("app/risk_status.py")
    for kata in ("cv_flag", "peringatan_visual", "hijau_pantau", "status_cv"):
        assert kata not in kode, f"{kata} masih dipakai di kode"


def test_content_tidak_punya_kunci_peran_cv():
    src = Path("app/content.py").read_text(encoding="utf-8")
    assert "hijau_pantau" not in src
    assert "cv_flag" not in src
    assert not [k for k in content.available_keys() if k.startswith("risiko.cv")]
    assert "hijau_pantau" not in content.STATUS
    assert "hijau_pantau" not in content.STATUS_LABELS
    for kelompok in content.KELOMPOK:
        assert "hijau_pantau" not in content.REKOMENDASI[kelompok]


# ---------------------------------------------------------------------------
# 2. Batas balita 11.5 / 12.5
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lila,harap",
    [
        (11.499, "merah"),
        (11.5, "kuning"),
        (12.499, "kuning"),
        (12.5, "hijau"),
        (5.0, "merah"),
        (50.0, "hijau"),
    ],
)
def test_batas_balita(lila, harap):
    r = evaluate_risk(kategori="balita", usia_bulan=24, lila_cm=lila,
                      berat_kg=12.0)
    assert r.status_antropometri == harap
    assert r.status_akhir == harap


def test_batas_konstanta_dipakai():
    hasil = [
        evaluate_risk(kategori="balita", usia_bulan=24, lila_cm=x,
                      berat_kg=12.0).status_akhir
        for x in (
            LILA_BALITA_MERAH_CM - 0.001,
            LILA_BALITA_MERAH_CM,
            LILA_BALITA_AMAN_CM - 0.001,
            LILA_BALITA_AMAN_CM,
        )
    ]
    assert hasil == ["merah", "kuning", "kuning", "hijau"]


@pytest.mark.parametrize("lila", [0.0, 4.9, 50.1, 100.0])
def test_lila_di_luar_rentang(lila):
    r = evaluate_risk(kategori="balita", usia_bulan=24, lila_cm=lila,
                      berat_kg=12.0)
    assert r.status_antropometri == "tidak_dapat_ditentukan"
    assert r.alasan


# ---------------------------------------------------------------------------
# Batas ibu 23.5 (tanpa kuning)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kat", ["ibu_hamil", "ibu_menyusui"])
def test_batas_ibu(kat):
    r_k = evaluate_risk(kategori=kat, lila_cm=23.499, berat_kg=55.0)
    r_a = evaluate_risk(kategori=kat, lila_cm=23.5, berat_kg=55.0)
    assert r_k.status_akhir == "merah"
    assert r_a.status_akhir == "hijau"


@pytest.mark.parametrize("lila", [5.0, 11.499, 11.5, 12.499, 12.5, 50.0])
def test_persetujuan_dengan_skrip_ekstraksi(lila):
    """Kedua modul membaca ambang yang sama -> keputusannya identik."""
    r = evaluate_risk(kategori="balita", usia_bulan=24, lila_cm=lila,
                      berat_kg=12.0)
    assert r.status_akhir == ed.risiko_balita(lila)


@pytest.mark.parametrize("lila", [5.0, 23.499, 23.5, 50.0])
def test_persetujuan_dengan_skrip_ekstraksi_ibu(lila):
    r = evaluate_risk(kategori="ibu_hamil", lila_cm=lila, berat_kg=55.0)
    assert r.status_akhir == ed.risiko_ibu(lila)


# ---------------------------------------------------------------------------
# Usia balita 6-59
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("usia,ok", [(5, False), (6, True), (59, True),
                                     (60, False), (0, False)])
def test_usia_balita(usia, ok):
    r = evaluate_risk(kategori="balita", usia_bulan=usia, lila_cm=13.0,
                      berat_kg=12.0)
    if ok:
        assert r.status_akhir == "hijau"
        assert any("berada di rentang" in a for a in r.alasan)
    else:
        assert r.status_akhir == "tidak_dapat_ditentukan"
        assert r.alasan


def test_usia_balita_wajib():
    r = evaluate_risk(kategori="balita", lila_cm=13.0, berat_kg=12.0)
    assert r.status_akhir == "tidak_dapat_ditentukan"


def test_usia_ibu_opsional_dan_rentangnya_bebas():
    r1 = evaluate_risk(kategori="ibu_hamil", lila_cm=25.0, berat_kg=55.0)
    r2 = evaluate_risk(kategori="ibu_hamil", usia_bulan=300, lila_cm=25.0,
                       berat_kg=55.0)
    assert r1.status_akhir == r2.status_akhir == "hijau"


# ---------------------------------------------------------------------------
# NaN / inf
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lila", [float("nan"), float("inf"), float("-inf")])
def test_lila_nan_inf(lila):
    r = evaluate_risk(kategori="balita", usia_bulan=24, lila_cm=lila,
                      berat_kg=12.0)
    assert r.status_akhir == "tidak_dapat_ditentukan"
    assert any("NaN" in a for a in r.alasan)


@pytest.mark.parametrize("field", ["berat_kg"])
def test_angka_wajib_finite(field):
    r = evaluate_risk(kategori="balita", usia_bulan=24, lila_cm=13.0,
                      **{field: float("nan")})
    assert r.status_akhir == "tidak_dapat_ditentukan"


def test_usia_nan():
    r = evaluate_risk(kategori="balita", usia_bulan=float("nan"),
                      lila_cm=13.0, berat_kg=12.0)
    assert r.status_akhir == "tidak_dapat_ditentukan"


def test_panjang_nan():
    r = evaluate_risk(kategori="balita", usia_bulan=24, lila_cm=13.0,
                      berat_kg=12.0, panjang_tinggi_cm=float("nan"))
    assert r.status_akhir == "tidak_dapat_ditentukan"


# ---------------------------------------------------------------------------
# Kategori salah -> TIDAK raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kat", ["ibu", "balita_lain", "", "NaN", "pregnant"])
def test_kategori_salah_tidak_menebak(kat):
    r = evaluate_risk(kategori=kat, lila_cm=13.0, berat_kg=12.0)
    assert r.status_akhir == "tidak_dapat_ditentukan"
    assert r.kelompok is None
    assert r.rekomendasi is None
    assert r.alasan


@pytest.mark.parametrize("kat,hp", [
    ("balita", "balita"), ("ibu_hamil", "ibu"), ("ibu_menyusui", "ibu"),
])
def test_kelompok_diturunkan_dari_kategori(kat, hp):
    r = evaluate_risk(kategori=kat, usia_bulan=36, lila_cm=13.0,
                      berat_kg=14.0)
    assert r.kelompok == hp
    assert r.rekomendasi is not None


# ---------------------------------------------------------------------------
# label_status & rekomendasi dari content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [
    "hijau", "kuning", "merah", "tidak_dapat_ditentukan",
])
def test_label_status_dari_content(status):
    r = evaluate_risk(**_input_untuk_status(status))
    assert r.label_status == content.get_text(f"status.{status}")


@pytest.mark.parametrize("status", [
    "hijau", "kuning", "merah", "tidak_dapat_ditentukan",
])
def test_rekomendasi_balita_dari_content(status):
    r = evaluate_risk(**_input_untuk_status(status))
    assert r.status_akhir == status
    assert r.rekomendasi == content.get_recommendation(status, "balita")


@pytest.mark.parametrize("status", ["hijau", "merah", "tidak_dapat_ditentukan"])
def test_rekomendasi_ibu_dari_content(status):
    r = evaluate_risk(**_input_untuk_status_ibu(status))
    assert r.status_akhir == status
    assert r.kelompok == "ibu"
    assert r.rekomendasi == content.get_recommendation(status, "ibu")


def _input_untuk_status(status: str) -> dict:
    if status == "hijau":
        return dict(kategori="balita", usia_bulan=36, lila_cm=13.0,
                    berat_kg=14.0)
    if status == "kuning":
        return dict(kategori="balita", usia_bulan=36, lila_cm=12.0,
                    berat_kg=14.0)
    if status == "merah":
        return dict(kategori="balita", usia_bulan=36, lila_cm=11.0,
                    berat_kg=14.0)
    return dict(kategori="balita", lila_cm=13.0, berat_kg=14.0)


def _input_untuk_status_ibu(status: str) -> dict:
    if status == "hijau":
        return dict(kategori="ibu_hamil", lila_cm=30.0, berat_kg=55.0)
    if status == "merah":
        return dict(kategori="ibu_hamil", lila_cm=20.0, berat_kg=55.0)
    return dict(kategori="ibu_hamil", lila_cm=30.0, berat_kg=float("nan"))


# ---------------------------------------------------------------------------
# 2. indikator_terukur
# ---------------------------------------------------------------------------


def test_indikator_urut_dan_kuncinya():
    r = evaluate_risk(kategori="balita", usia_bulan=36, lila_cm=13.0,
                      berat_kg=14.0)
    assert [i.kunci for i in r.indikator_terukur] == ["lila", "berat_badan"]


@pytest.mark.parametrize("status", [
    "hijau", "kuning", "merah", "tidak_dapat_ditentukan",
])
def test_badge_lila_sama_dengan_status_akhir(status):
    r = evaluate_risk(**_input_untuk_status(status))
    lila = _indikator(r, "lila")
    assert lila.badge == r.status_akhir == status


@pytest.mark.parametrize("status", ["hijau", "merah", "tidak_dapat_ditentukan"])
def test_badge_lila_ibu_sama_dengan_status_akhir(status):
    r = evaluate_risk(**_input_untuk_status_ibu(status))
    assert _indikator(r, "lila").badge == r.status_akhir == status


def test_lila_satuan_dan_nilai():
    r = evaluate_risk(kategori="balita", usia_bulan=36, lila_cm=13.0,
                      berat_kg=14.0)
    lila = _indikator(r, "lila")
    assert lila.satuan == "cm"
    assert lila.nilai == 13.0
    assert lila.keterangan


def test_keterangan_lila_ibu_menyebut_ambang():
    r = evaluate_risk(kategori="ibu_hamil", lila_cm=22.0, berat_kg=55.0)
    assert _indikator(r, "lila").keterangan == "22.0 cm, di bawah 23.5 cm"


def test_keterangan_lila_balita_menyebut_ambang():
    bawah = evaluate_risk(kategori="balita", usia_bulan=36, lila_cm=11.0,
                          berat_kg=14.0)
    tengah = evaluate_risk(kategori="balita", usia_bulan=36, lila_cm=12.0,
                           berat_kg=14.0)
    atas = evaluate_risk(kategori="balita", usia_bulan=36, lila_cm=13.0,
                         berat_kg=14.0)
    assert _indikator(bawah, "lila").keterangan == "11.0 cm, di bawah 11.5 cm"
    assert _indikator(tengah, "lila").keterangan == (
        "12.0 cm, antara 11.5 cm dan 12.5 cm"
    )
    assert _indikator(atas, "lila").keterangan == (
        "13.0 cm, setara atau di atas 12.5 cm"
    )


def test_keterangan_lila_atau_dari_content():
    r = evaluate_risk(kategori="ibu_hamil", lila_cm=22.0, berat_kg=55.0)
    teks = _indikator(r, "lila").keterangan
    assert teks == content.get_text("risiko.keterangan_bawah").format(
        nilai=22.0, ambang=LILA_IBU_AMAN_CM
    )


@pytest.mark.parametrize("kwargs,kunci", [
    (dict(kategori="balita", usia_bulan=60, lila_cm=13.0, berat_kg=14.0),
     "risiko.keterangan_usia_rentang"),
    (dict(kategori="balita", lila_cm=13.0, berat_kg=14.0),
     "risiko.keterangan_usia_kosong"),
    (dict(kategori="xxx", lila_cm=13.0, berat_kg=14.0),
     "risiko.keterangan_tanpa_kategori"),
    (dict(kategori="balita", usia_bulan=36, lila_cm=60.0, berat_kg=14.0),
     "risiko.keterangan_di_luar_rentang"),
    (dict(kategori="balita", usia_bulan=36, lila_cm=float("nan"),
          berat_kg=14.0),
     "risiko.keterangan_nilai_tidak_valid"),
    (dict(kategori="balita", usia_bulan=36, lila_cm=13.0,
          berat_kg=float("nan")),
     "risiko.keterangan_data_tidak_lengkap"),
])
def test_keterangan_lila_ketika_status_tidak_dapat_ditentukan(kwargs, kunci):
    r = evaluate_risk(**kwargs)
    assert r.status_akhir == "tidak_dapat_ditentukan"
    ket = _indikator(r, "lila").keterangan
    assert ket == content.get_text(kunci).format(**_args_dari(kunci, kwargs))
    assert "tidak dibandingkan" in ket


def _args_dari(kunci: str, kwargs: dict) -> dict:
    """Argumen format untuk kunci keterangan tertentu."""
    lila = kwargs["lila_cm"]
    if kunci == "risiko.keterangan_usia_rentang":
        return dict(nilai=lila, usia=kwargs["usia_bulan"],
                    min=thresholds.BALITA_MIN_MONTH,
                    max=thresholds.BALITA_MAX_MONTH)
    if kunci == "risiko.keterangan_tanpa_kategori":
        return dict(nilai=lila, kategori=kwargs["kategori"])
    if kunci == "risiko.keterangan_di_luar_rentang":
        return dict(nilai=lila, min=thresholds.LILA_MIN_CM,
                    max=thresholds.LILA_MAX_CM)
    if kunci == "risiko.keterangan_usia_kosong":
        return dict(nilai=lila)
    if kunci == "risiko.keterangan_data_tidak_lengkap":
        return dict(nilai=lila)
    return {}


def test_berat_badan_tanpa_penilaian():
    """badge selalu null; tidak ada penilaian naik/turun tanpa riwayat."""
    for kwargs in _SEMUA_KASUS():
        r = evaluate_risk(**kwargs)
        berat = _indikator(r, "berat_badan")
        assert berat.kunci == "berat_badan"
        assert berat.badge is None
        assert berat.keterangan is None
        expected = kwargs["berat_kg"]
        if math.isnan(expected):
            assert math.isnan(berat.nilai)
        else:
            assert berat.nilai == expected
        assert berat.satuan == "kg"


def test_berat_badan_label_dari_content():
    r = evaluate_risk(kategori="ibu_hamil", lila_cm=25.0, berat_kg=55.0)
    berat = _indikator(r, "berat_badan")
    assert berat.label == content.get_text("risiko.label_berat")
    assert "kenaikan" not in berat.label.lower()


def test_label_lila_per_kelompok_dari_content():
    balita = evaluate_risk(kategori="balita", usia_bulan=36, lila_cm=13.0,
                           berat_kg=14.0)
    ibu = evaluate_risk(kategori="ibu_hamil", lila_cm=25.0, berat_kg=55.0)
    assert _indikator(balita, "lila").label == content.get_text(
        "risiko.label_lila_balita"
    )
    assert _indikator(ibu, "lila").label == content.get_text(
        "risiko.label_lila_ibu"
    )


def test_label_lila_tanpa_kategori_pakai_label_netral():
    r = evaluate_risk(kategori="xxx", lila_cm=13.0, berat_kg=14.0)
    assert _indikator(r, "lila").label == content.get_text("risiko.label_lila")


def test_indikator_selalu_dua_item():
    for kwargs in _SEMUA_KASUS():
        r = evaluate_risk(**kwargs)
        assert len(r.indikator_terukur) == 2


# ---------------------------------------------------------------------------
# 3. Istilah per kelompok
# ---------------------------------------------------------------------------


def test_istilah_ibu_adalah_risiko_kek():
    nama = content.get_text("risiko.nama_ibu")
    assert "risiko KEK" in nama
    assert "gizi buruk akut" not in nama.lower()


def test_istilah_balita_mengikuti_ambang_balita():
    nama = content.get_text("risiko.nama_balita")
    assert "gizi buruk akut" not in nama.lower()
    # istilah menyebut LiLA, yaitu sumber ambang 11.5/12.5 cm
    assert "LiLA" in nama


def test_alasan_ibu_memakai_istilah_kek():
    r = evaluate_risk(kategori="ibu_hamil", lila_cm=22.0, berat_kg=55.0)
    gabung = " ".join(r.alasan)
    assert "risiko KEK" in gabung
    assert "gizi buruk" not in gabung.lower()


def test_alasan_balita_memakai_istilah_balita():
    r = evaluate_risk(kategori="balita", usia_bulan=36, lila_cm=13.0,
                      berat_kg=14.0)
    gabung = " ".join(r.alasan)
    assert "status gizi balita" in gabung
    assert "gizi buruk" not in gabung.lower()


def test_tidak_ada_istilah_gizi_buruk_akut_untuk_ibu():
    teks_ibu = [content.get_text(k) for k in content.available_keys()
                if k.startswith("risiko.")]
    for teks in teks_ibu:
        if "KEK" in teks or "ibu" in teks:
            assert "gizi buruk akut" not in teks.lower()


# ---------------------------------------------------------------------------
# 4. Ambang tidak diduplikasi
# ---------------------------------------------------------------------------

_AMBANG = (
    "LILA_MIN_CM", "LILA_MAX_CM", "LILA_BALITA_MERAH_CM",
    "LILA_BALITA_AMAN_CM", "LILA_IBU_AMAN_CM",
    "BALITA_MIN_MONTH", "BALITA_MAX_MONTH",
)


def test_kedua_modul_membaca_konstanta_yang_sama():
    """risk_status.py dan scripts/extract_dataset.py harus IDENTIK objectnya."""
    import app.risk_status as rs

    for nama in _AMBANG:
        nilai = getattr(thresholds, nama)
        assert getattr(ed, nama) == nilai, nama
        assert getattr(ed, nama) is nilai, f"{nama} di skrip berupa salinan"
        assert getattr(rs, nama) is nilai, f"{nama} di risk_status berupa salinan"


def test_tidak_ada_definisi_ambang_kedua():
    sumber_skrip = Path("scripts/extract_dataset.py").read_text(encoding="utf-8")
    sumber_modul = Path("app/risk_status.py").read_text(encoding="utf-8")
    for sumber, penanda in ((sumber_skrip, "from app.thresholds import"),
                            (sumber_modul, "from .thresholds import")):
        assert penanda in sumber
        for definisi in ("LILA_MIN_CM = 5.0", "LILA_BALITA_AMAN_CM = 12.5",
                         "LILA_IBU_AMAN_CM = 23.5", "BALITA_MIN_MONTH = 6",
                         "BALITA_MAX_MONTH = 59"):
            assert definisi not in sumber, definisi


def test_nilai_ambang_tidak_berubah():
    assert thresholds.LILA_MIN_CM == 5.0
    assert thresholds.LILA_MAX_CM == 50.0
    assert thresholds.LILA_BALITA_MERAH_CM == 11.5
    assert thresholds.LILA_BALITA_AMAN_CM == 12.5
    assert thresholds.LILA_IBU_AMAN_CM == 23.5
    assert thresholds.BALITA_MIN_MONTH == 6
    assert thresholds.BALITA_MAX_MONTH == 59


# ---------------------------------------------------------------------------
# Teks: semuanya dari content.py
# ---------------------------------------------------------------------------


def test_modul_tidak_menulis_literal_teks_content():
    kode = _token_kode("app/risk_status.py")
    for kunci in ("disclaimer.umum", "risiko.hasil", "risiko.data_tidak_layak",
                  "risiko.nama_ibu", "risiko.keterangan_bawah"):
        teks = content.get_text(kunci)
        assert teks not in kode, kunci
    assert content.PENANDA_PLACEHOLDER not in kode


def test_angka_batas_tidak_dihardcode_di_content():
    r = evaluate_risk(kategori="balita", usia_bulan=36, lila_cm=12.0,
                      berat_kg=14.0)
    assert any("11.5" in a and "12.5" in a for a in r.alasan)


def test_teks_alasan_dan_keterangan_dari_content():
    r = evaluate_risk(kategori="balita", usia_bulan=36, lila_cm=13.0,
                      berat_kg=14.0)
    tersedia = set(content.available_keys())
    for kunci in ("risiko.kategori_dikenal", "risiko.usia_ok",
                  "risiko.lila_ok", "risiko.batas_balita", "risiko.hasil",
                  "risiko.berat_dicatat", "risiko.nama_balita",
                  "risiko.label_lila_balita", "risiko.label_berat",
                  "risiko.keterangan_atas"):
        assert kunci in tersedia
    for baris in r.alasan:
        assert "status_cv" not in baris


# ---------------------------------------------------------------------------
# Data opsional dicatat
# ---------------------------------------------------------------------------


def test_panjang_dan_jk_dicatat():
    r = evaluate_risk(kategori="ibu_hamil", lila_cm=25.0, berat_kg=55.0,
                      panjang_tinggi_cm=160.0, jenis_kelamin="perempuan")
    gabung = " ".join(r.alasan)
    assert "160" in gabung
    assert "perempuan" in gabung
    assert r.status_akhir == "hijau"
