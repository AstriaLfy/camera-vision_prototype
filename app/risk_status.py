"""
risk_status.py
--------------
Status risiko antropometri (LiLA) + daftar indikator terukur.

Model input -> output (Pydantic `RiskStatus`):

    status_antropometri  hasil LiLA saja: hijau / kuning / merah /
                         tidak_dapat_ditentukan
    status_akhir         SELALU sama dengan status_antropometri
    kelompok             balita / ibu, diturunkan dari kategori
    rekomendasi          get_recommendation(status_akhir, kelompok)
    label_status         get_text("status.<status_akhir>")
    alasan               daftar kalimat yang menjelaskan cara status dihitung
    disclaimer           get_text("disclaimer.umum")
    indikator_terukur    daftar {kunci, label, nilai, satuan, badge, keterangan}

ATURAN STATUS -- BACA DULU:
    `status_akhir` murni dari LiLA. Modul ini TIDAK memakai `cv_flag`,
    TIDAK punya `peringatan_visual`, dan TIDAK punya status
    "hijau_pantau". Indikator wajah dari app/pipeline.py (termasuk
    `status_cv` bernilai normal / sedang / tinggi / tidak_valid) tidak
    dipetakan ke status risiko apa pun di sini.

INDIKATOR TERUKUR:
    * "lila"         -> `badge` sama dengan `status_akhir`; `keterangan`
                        membandingkan nilai dengan ambang yang sesuai
                        kelompok (semua teks dari app/content.py).
    * "berat_badan"  -> nilai saja: `badge` selalu None dan `keterangan`
                        None, karena TIDAK ada data riwayat sehingga
                        kenaikan/penurunan berat tidak bisa dinilai.

Ambang LiLA & usia ada di app/thresholds.py -- dimodul yang sama juga
dipakai scripts/extract_dataset.py, jadi label dataset dan status di layar
tidak bisa berbeda. Semua kalimat diambil dari app/content.py (kunci
"risiko.*"); modul ini tidak menulis literal teks yang tampil ke pengguna.

Contoh:
    from app.risk_status import evaluate_risk
    hasil = evaluate_risk(kategori="ibu_hamil", lila_cm=22.0, berat_kg=55.0)
    hasil.status_akhir                                  # 'merah'
    hasil.indikator_terukur[0].keterangan                # '22.0 cm, di bawah 23.5 cm'
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel

from . import content
from .thresholds import (
    BALITA_MAX_MONTH,
    BALITA_MIN_MONTH,
    LILA_BALITA_AMAN_CM,
    LILA_BALITA_MERAH_CM,
    LILA_IBU_AMAN_CM,
    LILA_MAX_CM,
    LILA_MIN_CM,
)

__all__ = [
    "RiskStatus",
    "IndikatorTerukur",
    "evaluate_risk",
    "KATEGORI",
    "KELOMPOK",
    "STATUS_ANTROPOMETRI",
    "STATUS_AKHIR",
    "LILA_MIN_CM",
    "LILA_MAX_CM",
    "LILA_BALITA_MERAH_CM",
    "LILA_BALITA_AMAN_CM",
    "LILA_IBU_AMAN_CM",
    "BALITA_MIN_MONTH",
    "BALITA_MAX_MONTH",
]

KATEGORI = ("balita", "ibu_hamil", "ibu_menyusui")
KELOMPOK = ("balita", "ibu")

#: kategori -> kelompok pemilihan rekomendasi
_KELOMPOK_DARI_KATEGORI = {
    "balita": "balita",
    "ibu_hamil": "ibu",
    "ibu_menyusui": "ibu",
}

STATUS_ANTROPOMETRI = ("hijau", "kuning", "merah", "tidak_dapat_ditentukan")
#: status_akhir murni dari LiLA, jadi daftarnya sama.
STATUS_AKHIR = STATUS_ANTROPOMETRI

BadgeStatus = Literal["hijau", "kuning", "merah", "tidak_dapat_ditentukan"]

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class IndikatorTerukur(BaseModel):
    """Satu pengukuran yang dilaporkan beserta penilaiannya."""

    kunci: Literal["lila", "berat_badan"]
    label: str
    nilai: float
    satuan: str
    badge: BadgeStatus | None
    keterangan: str | None


class RiskStatus(BaseModel):
    """Hasil perhitungan status risiko untuk satu subjek."""

    status_antropometri: BadgeStatus
    status_akhir: BadgeStatus
    kelompok: Literal["balita", "ibu"] | None
    rekomendasi: str | None
    label_status: str
    alasan: list[str]
    disclaimer: str
    indikator_terukur: list[IndikatorTerukur]


# ---------------------------------------------------------------------------
# Perhitungan
# ---------------------------------------------------------------------------


def evaluate_risk(
    *,
    kategori: str,
    lila_cm: float,
    berat_kg: float,
    usia_bulan: float | None = None,
    panjang_tinggi_cm: float | None = None,
    jenis_kelamin: str | None = None,
) -> RiskStatus:
    """Hitung status antropometri dari LiLA beserta indikator terukurnya.

    Args:
        kategori: "balita", "ibu_hamil", atau "ibu_menyusui". Nilai lain ->
            status_antropometri "tidak_dapat_ditentukan" (tidak ditebak).
        lila_cm: lingkar lengan atas. Wajib 5-50 cm dan finite.
        berat_kg: wajib finite. Tidak memengaruhi status, hanya dilaporkan.
        usia_bulan: wajib untuk balita (6-59 bulan). Opsional untuk ibu.
        panjang_tinggi_cm: opsional, dicatat saja.
        jenis_kelamin: opsional, dicatat saja.

    Returns:
        RiskStatus. `status_akhir` selalu sama dengan
        `status_antropometri`. Field `kelompok` dan `rekomendasi` None hanya
        bila kategori tidak dikenal (tidak ada kelompok untuk memilih
        rekomendasi).

    Catatan:
        Tidak ada parameter hasil-pemeriksaan-wajah. `status_cv` dari
        app/pipeline.py sengaja tidak dipakai -- lihat docstring modul.
    """
    alasan: list[str] = []
    masalah: list[str] = []          # penyebab status_antropometri jadi None

    # -- kategori -> kelompok ------------------------------------------------
    nama_kelompok = ""
    kelompok = _KELOMPOK_DARI_KATEGORI.get(kategori)
    if kelompok is None:
        alasan.append(
            content.get_text("risiko.kategori_salah").format(kategori=kategori)
        )
        masalah.append("kategori")
    else:
        nama_kelompok = content.get_text(f"risiko.nama_{kelompok}")
        alasan.append(
            content.get_text("risiko.kategori_dikenal").format(
                kategori=kategori, kelompok=kelompok, nama=nama_kelompok
            )
        )

    # -- semua angka yang diberikan harus finite (NaN/inf -> tidak layak) ----
    for nama, nilai in (
        ("lila_cm", lila_cm),
        ("berat_kg", berat_kg),
        ("usia_bulan", usia_bulan),
        ("panjang_tinggi_cm", panjang_tinggi_cm),
    ):
        if nilai is not None and not math.isfinite(nilai):
            alasan.append(
                content.get_text("risiko.angka_tidak_valid").format(nama=nama)
            )
            masalah.append(nama)

    # -- usia: hanya balita yang punya rentang --------------------------------
    if kelompok == "balita" and "usia_bulan" not in masalah:
        if usia_bulan is None:
            alasan.append(content.get_text("risiko.usia_kosong"))
            masalah.append("usia_bulan")
        elif usia_bulan < BALITA_MIN_MONTH:
            alasan.append(
                content.get_text("risiko.usia_terlalu_muda").format(
                    usia=usia_bulan, min=BALITA_MIN_MONTH
                )
            )
            masalah.append("usia_bulan")
        elif usia_bulan > BALITA_MAX_MONTH:
            alasan.append(
                content.get_text("risiko.usia_terlalu_tua").format(
                    usia=usia_bulan, max=BALITA_MAX_MONTH
                )
            )
            masalah.append("usia_bulan")
        else:
            alasan.append(
                content.get_text("risiko.usia_ok").format(
                    usia=usia_bulan, min=BALITA_MIN_MONTH, max=BALITA_MAX_MONTH
                )
            )

    # -- rentang LiLA ---------------------------------------------------------
    if "lila_cm" not in masalah:
        if lila_cm < LILA_MIN_CM:
            alasan.append(
                content.get_text("risiko.lila_terlalu_kecil").format(
                    lila=lila_cm, min=LILA_MIN_CM
                )
            )
            masalah.append("lila_cm")
        elif lila_cm > LILA_MAX_CM:
            alasan.append(
                content.get_text("risiko.lila_terlalu_besar").format(
                    lila=lila_cm, max=LILA_MAX_CM
                )
            )
            masalah.append("lila_cm")
        else:
            alasan.append(
                content.get_text("risiko.lila_ok").format(
                    lila=lila_cm, min=LILA_MIN_CM, max=LILA_MAX_CM
                )
            )

    # -- data yang dicatat tapi tidak mengubah status -------------------------
    if "berat_kg" not in masalah:
        alasan.append(
            content.get_text("risiko.berat_dicatat").format(berat=berat_kg)
        )
    if panjang_tinggi_cm is not None and "panjang_tinggi_cm" not in masalah:
        alasan.append(
            content.get_text("risiko.panjang_dicatat").format(
                panjang=panjang_tinggi_cm
            )
        )
    if jenis_kelamin is not None:
        alasan.append(
            content.get_text("risiko.jenis_kelamin_dicatat").format(
                jk=jenis_kelamin
            )
        )

    # -- status antropometri (murni LiLA) --------------------------------------
    if masalah:
        status_antropometri: str = "tidak_dapat_ditentukan"
        alasan.append(content.get_text("risiko.data_tidak_layak"))
    elif kelompok == "balita":
        alasan.append(
            content.get_text("risiko.batas_balita").format(
                nama=nama_kelompok,
                min=BALITA_MIN_MONTH,
                max=BALITA_MAX_MONTH,
                merah=LILA_BALITA_MERAH_CM,
                aman=LILA_BALITA_AMAN_CM,
            )
        )
        if lila_cm < LILA_BALITA_MERAH_CM:
            status_antropometri = "merah"
        elif lila_cm < LILA_BALITA_AMAN_CM:
            status_antropometri = "kuning"
        else:
            status_antropometri = "hijau"
        alasan.append(
            content.get_text("risiko.hasil").format(
                lila=lila_cm, nama=nama_kelompok, status=status_antropometri
            )
        )
    else:  # ibu hamil / ibu menyusui
        alasan.append(
            content.get_text("risiko.batas_ibu").format(
                nama=nama_kelompok, aman=LILA_IBU_AMAN_CM
            )
        )
        status_antropometri = "merah" if lila_cm < LILA_IBU_AMAN_CM else "hijau"
        alasan.append(
            content.get_text("risiko.hasil").format(
                lila=lila_cm, nama=nama_kelompok, status=status_antropometri
            )
        )

    # status_akhir tidak pernah berbeda dari status_antropometri
    status_akhir = status_antropometri

    # -- kelompok & rekomendasi ------------------------------------------------
    if kelompok is None:
        kelompok_hasil: str | None = None
        rekomendasi: str | None = None
        label_lila = content.get_text("risiko.label_lila")
    else:
        kelompok_hasil = kelompok
        rekomendasi = content.get_recommendation(status_akhir, kelompok)
        label_lila = content.get_text(f"risiko.label_lila_{kelompok}")

    # -- indikator terukur ------------------------------------------------------
    indikator_lila = IndikatorTerukur(
        kunci="lila",
        label=label_lila,
        nilai=lila_cm,
        satuan="cm",
        badge=status_akhir,
        keterangan=_keterangan_lila(
            lila_cm=lila_cm,
            kelompok=kelompok,
            kategori=kategori,
            usia_bulan=usia_bulan,
            masalah=masalah,
        ),
    )
    indikator_berat = IndikatorTerukur(
        kunci="berat_badan",
        label=content.get_text("risiko.label_berat"),
        nilai=berat_kg,
        satuan="kg",
        badge=None,          # tanpa riwayat -> tidak boleh dinilai
        keterangan=None,     # sama sekali tidak ada penilaian naik/turun
    )

    return RiskStatus(
        status_antropometri=status_antropometri,
        status_akhir=status_akhir,
        kelompok=kelompok_hasil,
        rekomendasi=rekomendasi,
        label_status=content.get_text(f"status.{status_akhir}"),
        alasan=alasan,
        disclaimer=content.get_text("disclaimer.umum"),
        indikator_terukur=[indikator_lila, indikator_berat],
    )


def _keterangan_lila(
    *,
    lila_cm: float,
    kelompok: str | None,
    kategori: str,
    usia_bulan: float | None,
    masalah: list[str],
) -> str:
    """Keterangan indikator "lila": nilai dibandingkan dengan ambangnya.

    Mengikuti status yang dihasilkan -- kalau status tidak dapat ditentukan,
    keterangannya menjelaskan mengapa LiLA tidak dibandingkan, bukan
    menghitung bandingan yang tidak berlaku.
    """
    if not math.isfinite(lila_cm):
        return content.get_text("risiko.keterangan_nilai_tidak_valid")

    if lila_cm < LILA_MIN_CM or lila_cm > LILA_MAX_CM:
        return content.get_text("risiko.keterangan_di_luar_rentang").format(
            nilai=lila_cm, min=LILA_MIN_CM, max=LILA_MAX_CM
        )

    if "kategori" in masalah:
        return content.get_text("risiko.keterangan_tanpa_kategori").format(
            nilai=lila_cm, kategori=kategori
        )

    if "usia_bulan" in masalah:
        if usia_bulan is None:
            return content.get_text("risiko.keterangan_usia_kosong").format(
                nilai=lila_cm
            )
        if not math.isfinite(usia_bulan):
            return content.get_text(
                "risiko.keterangan_data_tidak_lengkap"
            ).format(nilai=lila_cm)
        return content.get_text("risiko.keterangan_usia_rentang").format(
            nilai=lila_cm,
            usia=usia_bulan,
            min=BALITA_MIN_MONTH,
            max=BALITA_MAX_MONTH,
        )

    if masalah:
        # data lain (berat / panjang / jenis kelamin) yang tidak valid
        return content.get_text("risiko.keterangan_data_tidak_lengkap").format(
            nilai=lila_cm
        )

    if kelompok == "balita":
        if lila_cm < LILA_BALITA_MERAH_CM:
            return content.get_text("risiko.keterangan_bawah").format(
                nilai=lila_cm, ambang=LILA_BALITA_MERAH_CM
            )
        if lila_cm < LILA_BALITA_AMAN_CM:
            return content.get_text("risiko.keterangan_rentang").format(
                nilai=lila_cm,
                bawah=LILA_BALITA_MERAH_CM,
                atas=LILA_BALITA_AMAN_CM,
            )
        return content.get_text("risiko.keterangan_atas").format(
            nilai=lila_cm, ambang=LILA_BALITA_AMAN_CM
        )

    if lila_cm < LILA_IBU_AMAN_CM:
        return content.get_text("risiko.keterangan_bawah").format(
            nilai=lila_cm, ambang=LILA_IBU_AMAN_CM
        )
    return content.get_text("risiko.keterangan_atas").format(
        nilai=lila_cm, ambang=LILA_IBU_AMAN_CM
    )
