"""
test_indicators.py
------------------
app/indicators.py: fitur mentah pipeline -> daftar baris deskriptif.

Yang diuji:
  * pemetaan fitur -> kelompok memakai nama yang ADA di app/pipeline.py,
    dan fitur yang tidak terpetakan dilaporkan (bukan ditampilkan);
  * nilai NaN / hilang tetap jadi baris dengan nilai None;
  * pipeline gagal (None / bukan dict / kosong) -> items kosong, tanpa raise;
  * badge selalu None;
  * tidak ada kata terlarang di teks content maupun di hasil;
  * status_cv dan skor tidak dipakai serta tidak bocor ke output;
  * mekanisme ambang: default kosong, entri tanpa `sumber` diabaikan.
"""

from __future__ import annotations

import io
import math
import re
import tokenize
import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

from app import content
from app.indicators import (
    AmbangIndikator,
    Indicator,
    IndicatorResult,
    KELOMPOK,
    PEMETAAN,
    SUMBER_EKSPERIMENTAL,
    build_indicators,
    muat_ambang,
)
from app.settings import Settings

SUMBER_PIPELINE = Path("app/pipeline.py")
SUMBER_MODUL = Path("app/indicators.py")

#: Kata yang dilarang muncul di teks indikator (aturan 1).
TERLARANG = (
    "malnutrisi", "anemia", "berisiko", "terdeteksi", "pucat", "cekung",
)

#: Fitur pipeline yang memang BUKAN pengukuran (metadata / kualitas),
#: jadi sengaja tidak ditampilkan sebagai baris indikator.
META_YANG_TIDAK_DITAMPILKAN = (
    "available", "reason", "foreground_ratio", "n_pixels", "n_eyes",
)


# ---------------------------------------------------------------------------
# Membaca nama fitur dari sumber pipeline (bukan dari ingatan)
# ---------------------------------------------------------------------------


def _kunci_blok(teks: str) -> set[str]:
    """Nama kunci dict dari teks fungsi: `"kunci": ...` dan `out["kunci"] = ...`.

    Diparse dengan tokenizer supaya komentar dan f-string di pipeline tidak
    bikin salah baca.
    """
    if not teks.endswith("\n"):
        teks += "\n"
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(teks).readline))
    except (tokenize.TokenError, IndentationError) as exc:  # pragma: no cover
        raise AssertionError(f"pipeline gagal di-tokenize: {exc}") from exc

    urut = [
        (t.type, t.string)
        for t in toks
        if t.type not in (
            tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
            tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER,
        )
    ]
    hasil: set[str] = set()
    for i, (tipe, kata) in enumerate(urut[:-1]):
        if tipe != tokenize.STRING:
            continue
        if not (len(kata) > 1 and kata[0] in "\"'"):
            continue                      # f-string/bytes bukan kunci polos
        nama = kata[1:-1]
        if not re.fullmatch(r"[a-z_0-9]+", nama):
            continue
        berikut = urut[i + 1]
        if berikut == (tokenize.OP, ":"):
            hasil.add(nama)               # entri dict literal:  "kunci": nilai
        elif (
            berikut == (tokenize.OP, "]")
            and i + 2 < len(urut)
            and urut[i + 2] == (tokenize.OP, "=")
        ):
            hasil.add(nama)               # out["kunci"] = nilai
    return hasil


def _kunci_pipeline() -> dict[str, set[str]]:
    """Kunci yang dikeluarkan tiap extractor fitur di app/pipeline.py."""
    src = SUMBER_PIPELINE.read_text(encoding="utf-8")
    hasil: dict[str, set[str]] = {}
    for nama in (
        "extract_hair_features",
        "extract_sclera_features",
        "extract_geometry_features",
    ):
        blok = re.search(
            rf"^def {re.escape(nama)}\(.*?(?=^def |\Z)", src, re.S | re.M
        )
        assert blok, f"fungsi {nama} tidak ditemukan di app/pipeline.py"
        kunci = _kunci_blok(blok.group(0))
        assert kunci, f"{nama}: tidak ada kunci yang berhasil dibaca"
        hasil[nama] = kunci
    return hasil


def _semua_kunci_pipeline() -> set[str]:
    return set().union(*_kunci_pipeline().values())


# ---------------------------------------------------------------------------
# 6. Pemetaan terhadap pipeline
# ---------------------------------------------------------------------------


def test_petaan_hanya_memakai_nama_fitur_pipeline():
    """Tidak boleh ada nama fitur karangan di tabel pemetaan."""
    ada = _semua_kunci_pipeline()
    karangan = sorted(set(PEMETAAN) - ada)
    assert not karangan, (
        f"nama fitur berikut TIDAK ada di app/pipeline.py: {karangan}"
    )


def test_petaan_meliputi_semua_kelompok():
    assert {p.kelompok for p in PEMETAAN.values()} == set(KELOMPOK)


def test_fitur_pipeline_yang_tidak_terpetakan_dilaporkan():
    """Yang tidak terpetakan DILAPORKAN di test, bukan ditampilkan."""
    tak_terpetakan = sorted(_semua_kunci_pipeline() - set(PEMETAAN))
    pesan = (
        "[indicators] fitur pipeline tanpa pemetaan (tidak ditampilkan "
        f"sebagai baris): {tak_terpetakan}"
    )
    warnings.warn(pesan, stacklevel=2)
    print(f"\n{pesan}")
    assert tak_terpetakan == sorted(META_YANG_TIDAK_DITAMPILKAN), (
        "ada fitur baru di app/pipeline.py yang belum diputuskan: masukkan "
        "ke PEMETAAN, atau dokumentasikan di META_YANG_TIDAK_DITAMPILKAN"
    )


def test_petaan_mengarah_ke_kelompok_yang_valid():
    for kunci, p in PEMETAAN.items():
        assert p.kelompok in KELOMPOK, kunci
        assert content.get_text(f"satuan.{p.satuan}"), kunci


# ---------------------------------------------------------------------------
# 7. Bentuk output
# ---------------------------------------------------------------------------


def test_field_model():
    assert list(IndicatorResult.model_fields) == ["items", "tersedia"]
    assert list(Indicator.model_fields) == [
        "kunci", "kelompok", "label", "nilai", "satuan", "keterangan",
        "sumber", "badge", "badge_label",
    ]


def _fitur_lengkap(**tambahan) -> dict:
    fitur = {kunci: 1.0 for kunci in PEMETAAN}
    fitur.update(tambahan)
    return fitur


def test_items_lengkap_dan_urut_sesuai_petaan():
    r = build_indicators(_fitur_lengkap())
    assert r.tersedia is True
    assert [i.kunci for i in r.items] == list(PEMETAAN)


def test_satu_kelompok_per_item():
    r = build_indicators(_fitur_lengkap())
    assert [i.kelompok for i in r.items] == [
        PEMETAAN[k].kelompok for k in PEMETAAN
    ]


def test_konstanta_item():
    r = build_indicators(_fitur_lengkap())
    for item in r.items:
        assert item.sumber == SUMBER_EKSPERIMENTAL == "visual_eksperimental"
        assert item.badge is None
        assert item.badge_label == content.get_text("indikator.badge_label")
        assert item.badge_label == "Eksperimental"


def test_nilai_diteruskan_sebagai_float():
    r = build_indicators(
        _fitur_lengkap(**{"cheek_to_jaw_ratio": 0.82, "jaw_width_px": 300})
    )
    nilai = {i.kunci: i.nilai for i in r.items}
    assert nilai["cheek_to_jaw_ratio"] == 0.82
    assert nilai["jaw_width_px"] == 300.0
    assert isinstance(nilai["jaw_width_px"], float)
    assert nilai["hair_l"] == 1.0


# ---------------------------------------------------------------------------
# 4. NaN / hilang tetap jadi baris
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kasus", ["nan", "inf", "hilang", "string", "bool"])
def test_fitur_tidak_ada_nilai_tetap_jadi_baris(kasus):
    if kasus == "nan":
        tambahan = {"hair_l": float("nan")}
    elif kasus == "inf":
        tambahan = {"hair_l": float("inf")}
    elif kasus == "hilang":
        tambahan = {}
    elif kasus == "string":
        tambahan = {"hair_l": "abc"}
    else:
        tambahan = {"hair_l": True}

    fitur = {kunci: 1.0 for kunci in PEMETAAN}
    fitur.pop("hair_l", None)
    fitur.update(tambahan)

    r = build_indicators(fitur)
    item = next(i for i in r.items if i.kunci == "hair_l")
    assert len(r.items) == len(PEMETAAN)          # tidak dihilangkan diam-diam
    assert item.nilai is None
    assert item.keterangan == content.get_text("indikator.tidak_tersedia")
    assert item.keterangan == "tidak tersedia"
    assert item.badge is None


def test_fitur_tidak_pernah_dihilangkan():
    r = build_indicators({})
    r2 = build_indicators({"hair_l": 40.0})
    assert [i.kunci for i in r2.items] == list(PEMETAAN)
    assert sum(1 for i in r2.items if i.nilai is None) == len(PEMETAAN) - 1
    assert r.items == []


# ---------------------------------------------------------------------------
# 7b. Pipeline gagal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("input_gagal", [None, [], (), "fitur", {}])
def test_pipeline_gagal_tanpa_exception(input_gagal):
    r = build_indicators(input_gagal)  # type: ignore[arg-type]
    assert r.items == []
    assert r.tersedia is False


def test_pipeline_gagal_tidak_mengangkat_exception():
    class BerteleTele:
        def __iter__(self):
            raise RuntimeError("pipeline rusak")

    r = build_indicators(BerteleTele())  # type: ignore[arg-type]
    assert (r.items, r.tersedia) == ([], False)


# ---------------------------------------------------------------------------
# 1 & 3. Teks netral, dari content, tanpa kata terlarang
# ---------------------------------------------------------------------------


def _semua_teks_content_indikator() -> dict[str, str]:
    return {
        kunci: content.get_text(kunci)
        for kunci in content.available_keys()
        if kunci.startswith(("indikator.", "satuan."))
    }


def _semua_string(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _semua_string(v)]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in _semua_string(v)]
    return []


def test_teks_item_diambil_dari_content():
    r = build_indicators(_fitur_lengkap(hair_l=40.5))
    for item in r.items:
        assert item.label == content.get_text(f"indikator.label.{item.kunci}")
        assert item.satuan
        assert item.badge_label == content.get_text("indikator.badge_label")
    normal = next(i for i in r.items if i.nilai is not None)
    assert normal.keterangan == content.get_text("indikator.keterangan").format(
        label=normal.label, nilai=normal.nilai
    )
    assert normal.keterangan == f"{normal.label}: {normal.nilai}"


def test_semua_kunci_label_content_punya_pemetaan():
    for kunci in PEMETAAN:
        assert f"indikator.label.{kunci}" in content.available_keys()


def test_tidak_ada_kata_terlarang_di_content():
    for kunci, teks in _semua_teks_content_indikator().items():
        for kata in TERLARANG:
            assert kata not in teks.lower(), f"{kunci} memuat {kata!r}"


@pytest.mark.parametrize("fitur", [
    _fitur_lengkap(),
    _fitur_lengkap(hair_l=40.0, sclera_a_star=-2.0, a_star_ratio=0.5),
    {"hair_l": float("nan")},
])
def test_tidak_ada_kata_terlarang_di_hasil(fitur):
    r = build_indicators(fitur)
    for teks in _semua_string(r.model_dump()):
        for kata in TERLARANG:
            assert kata not in teks.lower(), f"{kata!r} ada di {teks!r}"


def test_tidak_ada_klasifikasi_di_hasil():
    r = build_indicators(_fitur_lengkap())
    teks = " ".join(_semua_string(r.model_dump())).lower()
    for kata in ("normal", "abnormal", "skor", "positif", "negatif"):
        assert kata not in teks, kata
    assert {i.badge for i in r.items} == {None}
    assert {i.badge_label for i in r.items} == {"Eksperimental"}


# ---------------------------------------------------------------------------
# 2. status_cv / skor tidak dipakai dan tidak bocor
# ---------------------------------------------------------------------------


def test_status_cv_dan_skor_tidak_bocor_ke_output():
    fitur = _fitur_lengkap(
        status_cv="MARKER_STATUS_CV",
        score=0.8734561,
        normalized=0.4123559,
        skor_hair=0.9234561,
        skor_agregat=0.5123559,
        indicators=[],
    )
    r = build_indicators(fitur)
    kunci_item = [i.kunci for i in r.items]
    for kunci in ("status_cv", "score", "normalized", "skor_hair",
                  "skor_agregat"):
        assert kunci not in kunci_item
        assert kunci not in PEMETAAN
    teks = " ".join(_semua_string(r.model_dump()))
    for kata in ("MARKER_STATUS_CV", "0.8734561", "0.4123559",
                 "0.9234561", "0.5123559", "status_cv", "skor"):
        assert kata not in teks, kata


def test_modul_tidak_merujuk_status_cv_atau_skor():
    for kata in ("status_cv", "score", "normalized", "skor"):
        assert kata not in _kode(SUMBER_MODUL), kata


def test_modul_tidak_menulis_literal_teks():
    kode = _kode(SUMBER_MODUL)
    for teks in _semua_teks_content_indikator().values():
        assert teks not in kode, teks


def _kode(path: Path) -> str:
    """Isi file tanpa string dan tanpa komentar."""
    src = path.read_text(encoding="utf-8")
    return " ".join(
        tok.string
        for tok in tokenize.generate_tokens(io.StringIO(src).readline)
        if tok.type not in (tokenize.STRING, tokenize.COMMENT)
    )


# ---------------------------------------------------------------------------
# 5. Mekanisme ambang
# ---------------------------------------------------------------------------


def test_default_ambang_kosong():
    assert Settings().ambang_indikator == []
    assert muat_ambang(None) == []
    assert muat_ambang([]) == []


def test_ambang_wajib_punya_sumber():
    with pytest.raises(ValidationError):
        AmbangIndikator(kunci="hair_l", sumber="")
    with pytest.raises(ValidationError):
        AmbangIndikator(kunci="hair_l")


def test_entri_tanpa_sumber_diabaikan():
    daftar = muat_ambang([
        {"kunci": "hair_l"},                       # tanpa sumber -> buang
        {"kunci": "hair_l", "sumber": ""},         # kosong -> buang
        {"kunci": "hair_l", "sumber": "   "},      # spasi -> buang
        {"kunci": "hair_l", "sumber": "pedoman X", "bawah": 10.0},
        "bukan-dict",                              # rusak -> buang
        {"kunci": "hair_l", "sumber": 123},        # sumber bukan teks -> buang
        None,
        {"kunci": "hair_l", "sumber": "y"},        # valid -> ikut
        {"kunci": "hair_l", "sumber": "y", "tanpa": "apa"},   # extra diabaikan
        {"kunci": "hair_l", "sumber": "y", "bawah": "bukan"},  # nilai rusak -> buang
    ])
    assert [a.sumber for a in daftar] == ["pedoman X", "y", "y"]
    assert daftar[1].bawah is None
    assert daftar[0].bawah == 10.0


def test_mekanisme_ambang_tidak_pernah_mengangkat_exception():
    for rusak in ("{json rusak", 123, object(), {"a": 1}):
        assert muat_ambang(rusak) == []


def test_ambang_dibaca_dari_config():
    s = Settings(ambang_indikator='[{"kunci":"hair_l","sumber":"pedoman X"}]')
    daftar = muat_ambang(s.ambang_indikator)
    assert len(daftar) == 1
    assert daftar[0].kunci == "hair_l"
    assert daftar[0].sumber == "pedoman X"
    assert daftar[0].bawah is None and daftar[0].atas is None


def test_ambang_tidak_menghasilkan_klasifikasi():
    """Output tidak punya jalur ambang: badge tetap None, tanpa penilaian."""
    assert "ambang" not in IndicatorResult.model_fields
    assert "ambang" not in Indicator.model_fields
    r = build_indicators(_fitur_lengkap(hair_l=0.0, sclera_a_star=100.0))
    assert all(i.badge is None for i in r.items)
    assert not math.isnan(sum(i.nilai for i in r.items))
