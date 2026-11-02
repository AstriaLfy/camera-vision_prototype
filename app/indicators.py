"""
indicators.py
--------------
Mengubah fitur MENTAH pipeline menjadi daftar data pendukung visual yang
deskriptif, ditampilkan sebagai baris terpisah dengan badge netral
"Eksperimental".

Tugasnya HANYA melaporkan nilai dan deskripsi netral, misalnya:

    Rasio lebar pipi terhadap lebar rahang: 0.82

Yang TIDAK ada di modul ini (sengaja):

  * klasifikasi / penilaian "atas-bawah normal" / ambang normal-abnormal;
  * skor risiko apa pun;
  * pemakaian `status_cv` maupun `score`/`normalized` dari app/pipeline.py --
    yang diteruskan cuma fitur mentah (rasio, nilai warna, lebar, dsb.);
  * penyaringan diam-diam: fitur NaN atau tidak tersedia tetap jadi baris
    dengan nilai None dan keterangan "tidak tersedia".

Struktur output:

    build_indicators(features) -> IndicatorResult { items, tersedia }
    tiap item -> Indicator { kunci, kelompok, label, nilai, satuan,
                             keterangan, sumber, badge, badge_label }

`features` adalah dict datar berisi fitur mentah hasil
`extract_hair_features()` / `extract_sclera_features()` /
`extract_geometry_features()` di app/pipeline.py (kuncinya persis nama fitur
di sana -- lihat PEMETAAN). Kunci yang tidak ada di PEMETAAN tidak
ditampilkan; test memantau hal itu.

Kalau `features` None / bukan dict / kosong (pipeline gagal atau tidak
menghasilkan apa pun): `items` kosong dan `tersedia` False, tanpa exception.

Mekanisme ambang ada di `app/settings.py::ambang_indikator` dan dibaca
lewat `muat_ambang()`. Setiap entri WAJIB punya `sumber`; entri tanpa
sumber diabaikan; default kosong. Ambang itu sengaja TIDAK dipakai untuk
menilai nilai di sini -- modul ini deskriptif, jadi ambang tidak boleh
menghasilkan normal/abnormal.

Semua teks (label, satuan, keterangan, badge_label) diambil dari
app/content.py; modul ini tidak menulis literal teks yang tampil.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Final, Literal

from pydantic import BaseModel, Field, ValidationError

from . import content

__all__ = [
    "Indicator",
    "IndicatorResult",
    "AmbangIndikator",
    "build_indicators",
    "muat_ambang",
    "PEMETAAN",
    "SUMBER_EKSPERIMENTAL",
    "KELOMPOK",
]

#: Konstan untuk SEMUA baris: sumber nilai, bukan hasil analisis.
SUMBER_EKSPERIMENTAL: Final[str] = "visual_eksperimental"

KELOMPOK: Final[tuple[str, ...]] = ("konjungtiva", "rambut", "pipi", "kulit")

Kelompok = Literal["konjungtiva", "rambut", "pipi", "kulit"]


# ---------------------------------------------------------------------------
# Tabel pemetaan fitur -> kelompok
#
# Kuncinya ADA PERSIS di keluaran app/pipeline.py (dibaca dari kodenya);
# nama yang tidak ada di sana dilarang masuk ke tabel ini -- diuji di
# tests/test_indicators.py. Fitur pipeline yang tidak terpetakan tidak
# ditampilkan (dilaporkan test, bukan ditebak).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pemetaan:
    kelompok: Kelompok
    satuan: str          # kunci satuan di app/content.py ("satuan.<kunci>")


PEMETAAN: Final[dict[str, Pemetaan]] = {
    # kelompok konjungtiva -- extract_sclera_features()
    "sclera_a_star": Pemetaan("konjungtiva", "a_star"),
    "a_star_ratio": Pemetaan("konjungtiva", "rasio"),
    # kelompok rambut -- extract_hair_features()
    "hair_l": Pemetaan("rambut", "l_star"),
    "hair_chroma": Pemetaan("rambut", "chroma"),
    "l_delta": Pemetaan("rambut", "l_star"),
    "chroma_ratio": Pemetaan("rambut", "rasio"),
    # kelompok pipi -- extract_geometry_features()
    "cheek_width_px": Pemetaan("pipi", "piksel"),
    "jaw_width_px": Pemetaan("pipi", "piksel"),
    "cheek_to_jaw_ratio": Pemetaan("pipi", "rasio"),
    "lower_face_height_ratio": Pemetaan("pipi", "rasio"),
    # kelompok kulit -- nilai acuan dari ketiga extractor di atas
    "skin_l": Pemetaan("kulit", "l_star"),
    "skin_chroma": Pemetaan("kulit", "chroma"),
    "skin_a_star": Pemetaan("kulit", "a_star"),
    "skin_a_star_raw": Pemetaan("kulit", "a_star"),
}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class Indicator(BaseModel):
    """Satu baris data pendukung visual -- deskriptif, tanpa penilaian."""

    kunci: str
    kelompok: Kelompok
    label: str
    nilai: float | None
    satuan: str
    keterangan: str
    sumber: str
    badge: None = None          # selalu None: netral, tidak ada status
    badge_label: str


class IndicatorResult(BaseModel):
    """Hasil build_indicators()."""

    items: list[Indicator]
    tersedia: bool


# ---------------------------------------------------------------------------
# Mekanisme ambang (konfigurasi) -- TIDAK dipakai untuk menilai nilai
# ---------------------------------------------------------------------------


class AmbangIndikator(BaseModel):
    """Satu ambang referensi dari konfigurasi.

    `sumber` WAJIB: asal angka (literatur/pedoman/dataset). Ambang tanpa
    asal usul tidak bisa dipertanggungjawabkan, jadi tidak pernah sampai
    ke model ini.
    """

    kunci: str
    sumber: str = Field(min_length=1)
    bawah: float | None = None
    atas: float | None = None


def muat_ambang(raw: object) -> list[AmbangIndikator]:
    """Baca daftar ambang dari konfigurasi, buang yang tidak bersumber.

    Aturan:
      * default / nilai kosong -> daftar kosong;
      * entri bukan dict, atau `sumber` kosong/hilang -> DIABAIKAN (tidak
        mengangkat error);
      * entri rusak lain juga diabaikan -- mekanisme ini tidak boleh membuat
        layanan gagal start.

    Ambang yang lolos di sini pun TIDAK dipakai app/indicators.py untuk
    menilai nilai: tidak ada klasifikasi dan tidak ada ambang
    normal/abnormal di output.
    """
    hasil: list[AmbangIndikator] = []
    if raw is None or isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return hasil
    for entri in raw:
        if not isinstance(entri, Mapping):
            continue
        sumber = entri.get("sumber")
        if not isinstance(sumber, str) or not sumber.strip():
            continue
        try:
            hasil.append(AmbangIndikator.model_validate(dict(entri)))
        except (ValidationError, ValueError, TypeError):
            continue
    return hasil


# ---------------------------------------------------------------------------
# Fungsi utama
# ---------------------------------------------------------------------------


def build_indicators(features: dict | None) -> IndicatorResult:
    """Susun daftar indikator visual deskriptif dari fitur mentah.

    Args:
        features: dict datar fitur mentah pipeline, atau None bila pipeline
            gagal / tidak dijalankan.

    Returns:
        IndicatorResult -- `items` selalu berisi SATU baris per kunci di
        PEMETAAN (urut seperti tabel) selama `tersedia` True, meski nilainya
        None. `tersedia` False (items kosong) bila features None, bukan dict,
        atau kosong. Tidak pernah melempar exception.
    """
    if not isinstance(features, Mapping) or not features:
        return IndicatorResult(items=[], tersedia=False)

    return IndicatorResult(
        items=[_baris(kunci, pemetaan, features)
               for kunci, pemetaan in PEMETAAN.items()],
        tersedia=True,
    )


def _baris(kunci: str, pemetaan: Pemetaan, features: Mapping) -> Indicator:
    """Satu baris indikator. Nilai NaN/hilang -> None + "tidak tersedia"."""
    label = content.get_text(f"indikator.label.{kunci}")
    nilai = _angka(features.get(kunci))
    if nilai is None:
        keterangan = content.get_text("indikator.tidak_tersedia")
    else:
        keterangan = content.get_text("indikator.keterangan").format(
            label=label, nilai=nilai
        )
    return Indicator(
        kunci=kunci,
        kelompok=pemetaan.kelompok,
        label=label,
        nilai=nilai,
        satuan=content.get_text(f"satuan.{pemetaan.satuan}"),
        keterangan=keterangan,
        sumber=SUMBER_EKSPERIMENTAL,
        badge=None,
        badge_label=content.get_text("indikator.badge_label"),
    )


def _angka(nilai: object) -> float | None:
    """True number yang finite -> float. Selain itu -> None (tanpa error)."""
    if isinstance(nilai, bool) or not isinstance(nilai, Real):
        return None
    hasil = float(nilai)
    return hasil if math.isfinite(hasil) else None
