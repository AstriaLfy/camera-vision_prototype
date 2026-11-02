"""
content.py
----------
SATU-SATUNYA sumber teks yang tampil ke pengguna (konten produk: disclaimer,
label status, rekomendasi, panduan foto).

Prinsip:
  * Teks di sini ditulis sekali, dipakai oleh endpoint/UI lewat get_text()
    atau get_recommendation(). Endpoint TIDAK boleh menulis literal teks
    yang tampil ke pengguna.
  * Kunci hilang = error yang jelas (ContentKeyError), BUKAN string kosong.
    Kegagalan harus ketahuan saat itu juga, bukan diam-diam menampilkan
    teks kosong di depan petugas.
  * Rekomendasi per status x kelompok masih PLACEHOLDER bertanda
    "[PERLU DIVERIFIKASI AHLI GIZI]". TIDAK mengandung isi protokol medis
    atau dosis apa pun -- akan diisi dari pedoman Kemenkes/WHO.

Batas file ini: hanya konten yang tampil di layar/kartu/respons.
Pesan protokol HTTP (detail error, pesan validasi, hint model hilang) tetap
berada di app/routers/ dan app/errors.py -- itu komunikasi teknis antar program,
bukan konten produk.

CATATAN: status_cv yang dihasilkan app/pipeline.py sekarang masih
"normal | sedang | tinggi | tidak_valid", berbeda dari status tampilan di
bawah (hijau/kuning/merah/tidak_dapat_ditentukan). Pemetaannya
belum ada dan sengaja TIDAK ditebak di file ini.

Format: murni Python (tanpa loader YAML) supaya tidak menambah dependency.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "ContentKeyError",
    "DISCLAIMER_UMUM",
    "DISCLAIMER_VISUAL",
    "DISCLAIMER_PRIVASI",
    "LABEL_KARTU_VISUAL",
    "STATUS_LABELS",
    "REKOMENDASI",
    "FOTO_TIDAK_LAYAK",
    "RISK_TEXTS",
    "INDIKATOR_BADGE_LABEL",
    "INDIKATOR_TIDAK_TERSEDIA",
    "INDIKATOR_KETERANGAN",
    "INDIKATOR_LABEL",
    "INDIKATOR_SATUAN",
    "RINGKASAN_DESKRIPSI",
    "ENDPOINT_CV_MODE_MODEL",
    "STATUS",
    "KELOMPOK",
    "PENANDA_PLACEHOLDER",
    "get_text",
    "get_recommendation",
    "available_keys",
]


class ContentKeyError(KeyError):
    """Kunci konten tidak ditemukan / argumen tidak dikenal.

    Subclass KeyError supaya `except KeyError` tetap menangkapnya, tapi
    str(e) berisi penjelasan yang jelas, bukan sekadar nama kunci.
    """


# ---------------------------------------------------------------------------
# 1-3. Disclaimer
# ---------------------------------------------------------------------------

DISCLAIMER_UMUM: Final[str] = (
    "Hasil ini adalah skrining awal dan bukan diagnosis medis. "
    "Keputusan tindakan tetap berada pada tenaga kesehatan."
)

DISCLAIMER_VISUAL: Final[str] = (
    "Analisis visual bersifat eksperimental, belum tervalidasi secara klinis, "
    "dan tidak mengubah status risiko utama."
)

DISCLAIMER_PRIVASI: Final[str] = (
    "Foto wajah diambil dengan persetujuan subjek atau wali dan hanya "
    "digunakan untuk keperluan skrining."
)


# ---------------------------------------------------------------------------
# 4. Label kartu visual
# ---------------------------------------------------------------------------

LABEL_KARTU_VISUAL: Final[str] = "Indikasi visual (eksperimental)"


# ---------------------------------------------------------------------------
# 5. Label tampilan per status
# ---------------------------------------------------------------------------

STATUS_LABELS: Final[dict[str, str]] = {
    "hijau": "Hijau - tidak ada indikasi pada skrining ini",
    "kuning": "Kuning - perlu perhatian",
    "merah": "Merah - perlu ditindaklanjuti",
    "tidak_dapat_ditentukan": "Tidak dapat ditentukan - data tidak mencukupi",
}

STATUS: Final[tuple[str, ...]] = tuple(STATUS_LABELS)
KELOMPOK: Final[tuple[str, ...]] = ("balita", "ibu")


# ---------------------------------------------------------------------------
# 6. Rekomendasi per status x kelompok -- PLACEHOLDER
# ---------------------------------------------------------------------------

#: Penanda yang dicari test pelapor. Kalau sudah lenyap dari semua entri,
#: berarti isi sudah diganti dengan teks asli.
PENANDA_PLACEHOLDER: Final[str] = "[PERLU DIVERIFIKASI AHLI GIZI]"

# Isi setiap entri sengaja ditulis literal (bukan digenerate) supaya mudah
# diganti satu per satu saat pedoman Kemenkes/WHO sudah siap.
# Semuanya HARUS tetap memuat PENANDA_PLACEHOLDER sampai diverifikasi ahli gizi.
REKOMENDASI: Final[dict[str, dict[str, str]]] = {
    "balita": {
        "hijau": (
            "[PERLU DIVERIFIKASI AHLI GIZI] Rekomendasi untuk balita berstatus "
            "hijau belum diisi -- akan diambil dari pedoman Kemenkes/WHO."
        ),
        "kuning": (
            "[PERLU DIVERIFIKASI AHLI GIZI] Rekomendasi untuk balita berstatus "
            "kuning belum diisi -- akan diambil dari pedoman Kemenkes/WHO."
        ),
        "merah": (
            "[PERLU DIVERIFIKASI AHLI GIZI] Rekomendasi untuk balita berstatus "
            "merah belum diisi -- akan diambil dari pedoman Kemenkes/WHO."
        ),
        "tidak_dapat_ditentukan": (
            "[PERLU DIVERIFIKASI AHLI GIZI] Rekomendasi untuk balita berstatus "
            "tidak_dapat_ditentukan belum diisi -- akan diambil dari pedoman "
            "Kemenkes/WHO."
        ),
    },
    "ibu": {
        "hijau": (
            "[PERLU DIVERIFIKASI AHLI GIZI] Rekomendasi untuk ibu berstatus "
            "hijau belum diisi -- akan diambil dari pedoman Kemenkes/WHO."
        ),
        "kuning": (
            "[PERLU DIVERIFIKASI AHLI GIZI] Rekomendasi untuk ibu berstatus "
            "kuning belum diisi -- akan diambil dari pedoman Kemenkes/WHO."
        ),
        "merah": (
            "[PERLU DIVERIFIKASI AHLI GIZI] Rekomendasi untuk ibu berstatus "
            "merah belum diisi -- akan diambil dari pedoman Kemenkes/WHO."
        ),
        "tidak_dapat_ditentukan": (
            "[PERLU DIVERIFIKASI AHLI GIZI] Rekomendasi untuk ibu berstatus "
            "tidak_dapat_ditentukan belum diisi -- akan diambil dari pedoman "
            "Kemenkes/WHO."
        ),
    },
}


# ---------------------------------------------------------------------------
# 7. Foto tidak layak -- bahasa sederhana untuk petugas
# ---------------------------------------------------------------------------

FOTO_TIDAK_LAYAK: Final[dict[str, str]] = {
    "buram": (
        "Fotonya blur. Minta diambil ulang dengan tangan lebih stabil atau "
        "cahaya yang lebih terang, lalu pastikan wajah terlihat tajam."
    ),
    "wajah_tidak_terdeteksi": (
        "Wajahnya tidak terbaca. Minta subjek menghadap kamera dengan seluruh "
        "wajah terlihat jelas dan tidak tertutup tangan atau rambut."
    ),
    "pencahayaan_kurang": (
        "Cahayanya kurang atau tidak merata. Pindah ke tempat dengan cahaya "
        "cukup dan merata, jangan menyilaukan mata."
    ),
    "jarak_terlalu_jauh": (
        "Wajahnya terlalu kecil karena kamera terlalu jauh. Dekatkan kamera "
        "sampai wajah terlihat besar dan jelas."
    ),
    "jarak_terlalu_dekat": (
        "Wajahnya terlalu dekat sehingga terpotong. Mundurkan kamera sedikit "
        "sampai seluruh wajah muat dalam bingkai."
    ),
    "pose_miring": (
        "Wajahnya menghadap miring atau menunduk. Minta subjek menoleh lurus "
        "ke kamera dan menegakkan kepala."
    ),
    "beberapa_wajah": (
        "Fotonya memuat lebih dari satu wajah. Alat ini menilai satu subjek, "
        "jadi potret ulang hanya berisi wajah subjek yang diperiksa."
    ),
    "tidak_layak": (
        "Foto belum bisa dianalisis. Ambil ulang foto sesuai panduan, lalu "
        "coba lagi."
    ),
}


# ---------------------------------------------------------------------------
# 8. Teks perhitungan status risiko & indikator terukur (app/risk_status.py)
#
# Berupa TEMPLATE format -- angka (batas LiLA, usia, nilai) diisi oleh pemanggil
# dari app/thresholds.py, supaya kalimat dan angka tidak pernah terpisah.
# Isinya penjelasan cara status dihitung, BUKAN saran medis.
#
# ISTILAH PER KELOMPOK (jangan disamakan):
#   balita -> "status gizi balita menurut LiLA"  (mengikuti ambang 11.5/12.5 cm)
#   ibu    -> "risiko KEK (Kekurangan Energi Kronis)" (mengikuti ambang 23.5 cm)
# Ibu TIDAK pernah disebut "gizi buruk akut".
# ---------------------------------------------------------------------------

RISK_TEXTS: Final[dict[str, str]] = {
    # batas
    "risiko.batas_balita": (
        "Ambang {nama} untuk usia {min:g}-{max:g} bulan: LiLA < {merah:g} cm = "
        "merah, {merah:g} sampai < {aman:g} cm = kuning, >= {aman:g} cm = hijau."
    ),
    "risiko.batas_ibu": (
        "Ambang {nama}: LiLA < {aman:g} cm = merah, >= {aman:g} cm = hijau."
    ),
    # istilah per kelompok (lihat catatan di atas seksi ini)
    "risiko.nama_balita": "status gizi balita menurut LiLA",
    "risiko.nama_ibu": "risiko KEK (Kekurangan Energi Kronis)",
    # label indikator terukur
    "risiko.label_lila": "LiLA (lingkar lengan atas)",
    "risiko.label_lila_balita": "LiLA - status gizi balita",
    "risiko.label_lila_ibu": "LiLA - risiko KEK (Kekurangan Energi Kronis)",
    "risiko.label_berat": "Berat badan",
    # kategori & usia
    "risiko.kategori_dikenal": (
        "Kategori '{kategori}' dikenal -> kelompok '{kelompok}' untuk menilai "
        "{nama}."
    ),
    "risiko.kategori_salah": (
        "Kategori '{kategori}' tidak dikenal; harus balita, ibu_hamil, atau "
        "ibu_menyusui. Status tidak ditebak."
    ),
    "risiko.usia_ok": (
        "Usia {usia:g} bulan berada di rentang {min:g}-{max:g} bulan."
    ),
    "risiko.usia_terlalu_muda": (
        "Usia {usia:g} bulan kurang dari {min:g} bulan; status tidak dapat "
        "ditentukan."
    ),
    "risiko.usia_terlalu_tua": (
        "Usia {usia:g} bulan lebih dari {max:g} bulan; status tidak dapat "
        "ditentukan."
    ),
    "risiko.usia_kosong": (
        "Usia tidak diberikan untuk kategori balita; status tidak dapat "
        "ditentukan."
    ),
    # kelayakan angka
    "risiko.lila_ok": (
        "LiLA {lila:g} cm berada di rentang valid {min:g}-{max:g} cm."
    ),
    "risiko.lila_terlalu_kecil": (
        "LiLA {lila:g} cm kurang dari {min:g} cm; status tidak dapat ditentukan."
    ),
    "risiko.lila_terlalu_besar": (
        "LiLA {lila:g} cm lebih dari {max:g} cm; status tidak dapat ditentukan."
    ),
    "risiko.angka_tidak_valid": (
        "{nama} bukan angka yang valid (NaN atau inf); status tidak dapat "
        "ditentukan."
    ),
    "risiko.data_tidak_layak": (
        "Ada data yang tidak memenuhi syarat, jadi status antropometri "
        "'tidak_dapat_ditentukan' -- tidak ditebak."
    ),
    # data yang dicatat tapi tidak dipakai
    "risiko.berat_dicatat": (
        "Berat badan {berat:g} kg hanya dicatat; status LiLA tidak memakainya."
    ),
    "risiko.panjang_dicatat": (
        "Panjang/tinggi {panjang:g} cm hanya dicatat; status LiLA tidak "
        "memakainya."
    ),
    "risiko.jenis_kelamin_dicatat": (
        "Jenis kelamin '{jk}' hanya dicatat; status LiLA tidak memakainya."
    ),
    # hasil
    "risiko.hasil": "LiLA {lila:g} cm -> {nama}: '{status}'.",

    # keterangan indikator "lila" (nilai dibandingkan dengan ambang)
    "risiko.keterangan_bawah": "{nilai} cm, di bawah {ambang} cm",
    "risiko.keterangan_atas": "{nilai} cm, setara atau di atas {ambang} cm",
    "risiko.keterangan_rentang": (
        "{nilai} cm, antara {bawah} cm dan {atas} cm"
    ),
    "risiko.keterangan_nilai_tidak_valid": (
        "nilai LiLA bukan angka yang valid (NaN atau inf), jadi tidak "
        "dibandingkan dengan ambang"
    ),
    "risiko.keterangan_di_luar_rentang": (
        "{nilai} cm, di luar rentang wajar {min}-{max} cm, jadi tidak "
        "dibandingkan dengan ambang"
    ),
    "risiko.keterangan_tanpa_kategori": (
        "{nilai} cm, tidak dibandingkan dengan ambang karena kategori "
        "'{kategori}' tidak dikenal"
    ),
    "risiko.keterangan_usia_kosong": (
        "{nilai} cm, tidak dibandingkan dengan ambang karena usia balita "
        "tidak diberikan"
    ),
    "risiko.keterangan_usia_rentang": (
        "{nilai} cm, tidak dibandingkan dengan ambang karena usia {usia} "
        "bulan di luar {min}-{max} bulan"
    ),
    "risiko.keterangan_data_tidak_lengkap": (
        "{nilai} cm, tidak dibandingkan dengan ambang karena ada data lain "
        "yang tidak valid"
    ),
}



# ---------------------------------------------------------------------------
# 9. Indikator visual deskriptif (app/indicators.py)
#
# Modul ini HANYA melaporkan nilai mentah + deskripsi netral. Tidak ada
# klasifikasi, tidak ada skor, tidak ada penilaian "atas/bawah normal".
# Badge tiap baris selalu netral (lihat INDIKATOR_BADGE_LABEL).
# ---------------------------------------------------------------------------

#: Badge yang dipakai di SEMUA baris indikator visual.
INDIKATOR_BADGE_LABEL: Final[str] = "Eksperimental"

#: Keterangan untuk nilai yang NaN / tidak tersedia / tidak dievaluasi.
INDIKATOR_TIDAK_TERSEDIA: Final[str] = "tidak tersedia"

#: Pola keterangan baris. {label} dan {nilai} diisi oleh app/indicators.py.
INDIKATOR_KETERANGAN: Final[str] = "{label}: {nilai}"

#: Label per fitur. Kuncinya = nama fitur persis seperti di app/pipeline.py.
INDIKATOR_LABEL: Final[dict[str, str]] = {
    # konjungtiva (dari extract_sclera_features)
    "sclera_a_star": "Nilai a* sklera",
    "a_star_ratio": "Rasio a* sklera terhadap a* kulit",
    # rambut (dari extract_hair_features)
    "hair_l": "Nilai L* rambut",
    "hair_chroma": "Nilai chroma rambut",
    "l_delta": "Selisih L* kulit dengan L* rambut",
    "chroma_ratio": "Rasio chroma rambut terhadap chroma kulit",
    # pipi (dari extract_geometry_features)
    "cheek_width_px": "Lebar pipi",
    "jaw_width_px": "Lebar rahang",
    "cheek_to_jaw_ratio": "Rasio lebar pipi terhadap lebar rahang",
    "lower_face_height_ratio": (
        "Rasio tinggi bagian bawah wajah terhadap tinggi wajah"
    ),
    # kulit (nilai acuan dari ketiga extractor)
    "skin_l": "Nilai L* kulit",
    "skin_chroma": "Nilai chroma kulit",
    "skin_a_star": "Nilai a* kulit",
    "skin_a_star_raw": "Nilai a* kulit sebelum batas bawah",
}

#: Satuan yang bisa muncul di baris indikator.
INDIKATOR_SATUAN: Final[dict[str, str]] = {
    "rasio": "rasio",
    "piksel": "piksel",
    "l_star": "L*",
    "a_star": "a*",
    "chroma": "chroma",
}


# ---------------------------------------------------------------------------
# 10. Endpoint skrining baru (app/routers/screening.py)
#
# Ringkasan singkat per status (label sengaja dipisah: label = status.<status>,
# deskripsi = kalimat penjelas di kartu ringkasan). Mode layanan "model"
# balasannya 501 -- pesannya di sini.
# ---------------------------------------------------------------------------

RINGKASAN_DESKRIPSI: Final[dict[str, str]] = {
    "hijau": (
        "Tidak ada indikasi risiko pada skrining LiLA ini."
    ),
    "kuning": (
        "Perlu perhatian: hasil skrining LiLA berada pada rentang pemantauan."
    ),
    "merah": (
        "Perlu ditindaklanjuti: hasil skrining LiLA berada di bawah ambang aman."
    ),
    "tidak_dapat_ditentukan": (
        "Status tidak dapat ditentukan karena ada data yang tidak memenuhi "
        "syarat, jadi status tidak ditebak."
    ),
}

ENDPOINT_CV_MODE_MODEL: Final[str] = (
    "Mode layanan 'model' belum tersedia: model terlatih belum "
    "diimplementasikan. Pakai CV_MODE=indicator_only untuk indikator "
    "rule-based, atau CV_MODE=off untuk mematikan analisis visual."
)


# ---------------------------------------------------------------------------
# Teks yang sudah tampil di endpoint yang ada (dipindahkan KE sini verbatim,
# nilainya TIDAK berubah supaya kontrak respons yang sudah ada tetap sama)
# ---------------------------------------------------------------------------

ENDPOINT_DISCLAIMER_ANALISIS: Final[str] = (
    "Hasil berbasis rule-based threshold (lihat app/config.py), BUKAN model yang "
    "di-training dari dataset malnutrisi. 'kepucatan_mata' mengukur area mata "
    "yang terekspos sebagai PROXY konjungtiva pucat -- konjungtiva sejati hanya "
    "terlihat bila kelopak mata ditarik, jadi ini tidak setara dengan pemeriksaan "
    "konjungtiva klinis. Sensitivitas kepucatan konjungtiva terhadap anemia "
    "ringan rendah (<69%); untuk anemia sedang/berat lebih tinggi (>=70%)."
)

ENDPOINT_REKOMENDASI_ANALISIS: Final[str] = (
    "Konfirmasi dengan pengukuran MUAC dan/atau tes hemoglobin. "
    "Foto wajah tidak dapat menggantikan pemeriksaan anthropometri."
)

ENDPOINT_DISCLAIMER_AKAR: Final[str] = "Alat bantu skrining, bukan diagnosis."


# ---------------------------------------------------------------------------
# Registry: kunci datar yang dibaca get_text()
# ---------------------------------------------------------------------------

_TEXT: Final[dict[str, str]] = {
    # item 1-3
    "disclaimer.umum": DISCLAIMER_UMUM,
    "disclaimer.visual": DISCLAIMER_VISUAL,
    "disclaimer.privasi": DISCLAIMER_PRIVASI,
    # item 4
    "label.kartu_visual": LABEL_KARTU_VISUAL,
    # item 7 -- semua kondisi foto tidak layak (termasuk yang dipakai
    # endpoint skrining baru: beberapa_wajah, tidak_layak)
    # (diisi oleh loop di bawah)
    # teks endpoint yang sudah ada (nilai tidak berubah)
    "endpoint.disclaimer_analisis": ENDPOINT_DISCLAIMER_ANALISIS,
    "endpoint.rekomendasi_analisis": ENDPOINT_REKOMENDASI_ANALISIS,
    "endpoint.disclaimer_akar": ENDPOINT_DISCLAIMER_AKAR,
}

# item 5 -- kunci "status.<nama>"
for _status, _label in STATUS_LABELS.items():
    _TEXT[f"status.{_status}"] = _label

# item 7 -- kunci "foto.<kondisi>"
for _kondisi, _teks in FOTO_TIDAK_LAYAK.items():
    _TEXT[f"foto.{_kondisi}"] = _teks

# item 10 -- kunci "ringkasan.deskripsi.<status>"
for _status, _teks in RINGKASAN_DESKRIPSI.items():
    _TEXT[f"ringkasan.deskripsi.{_status}"] = _teks
_TEXT["endpoint.cv_mode_model"] = ENDPOINT_CV_MODE_MODEL

# item 8 -- kunci "risiko.<nama>" (template format)
for _kunci, _teks in RISK_TEXTS.items():
    _TEXT[_kunci] = _teks

# item 9 -- kunci "indikator.*" dan "satuan.*"
_TEXT["indikator.badge_label"] = INDIKATOR_BADGE_LABEL
_TEXT["indikator.tidak_tersedia"] = INDIKATOR_TIDAK_TERSEDIA
_TEXT["indikator.keterangan"] = INDIKATOR_KETERANGAN
for _kunci, _teks in INDIKATOR_LABEL.items():
    _TEXT[f"indikator.label.{_kunci}"] = _teks
for _kunci, _teks in INDIKATOR_SATUAN.items():
    _TEXT[f"satuan.{_kunci}"] = _teks


def _validate() -> None:
    """Nilai harus string terisi -- menangkap kesalahan ketik saat import."""
    semua = dict(_TEXT)
    for kelompok, daftar in REKOMENDASI.items():
        for status, teks in daftar.items():
            semua[f"rekomendasi.{kelompok}.{status}"] = teks
    bermasalah = [
        f"{key!r} = {value!r}"
        for key, value in semua.items()
        if not isinstance(value, str) or not value.strip()
    ]
    if bermasalah:
        raise RuntimeError(
            "Konten tidak valid (harus string terisi): " + "; ".join(bermasalah)
        )


_validate()


def available_keys() -> tuple[str, ...]:
    """Kunci yang bisa dipakai get_text(), terurut."""
    return tuple(sorted(_TEXT))


def get_text(key: str) -> str:
    """Ambil satu teks konten berdasarkan kunci.

    Raises:
        ContentKeyError: kunci tidak ada. Pesannya menyebut kunci yang
            dicoba dan daftar kunci yang tersedia -- bukan string kosong.
    """
    try:
        return _TEXT[key]
    except (KeyError, TypeError):
        raise ContentKeyError(
            f"Kunci konten tidak ditemukan: {key!r}. "
            f"Kunci yang tersedia: {', '.join(available_keys())}."
        ) from None


def get_recommendation(status: str, kelompok: str) -> str:
    """Ambil rekomendasi tampilan untuk satu status dan satu kelompok.

    Args:
        status: salah satu dari STATUS (hijau, kuning, merah,
            tidak_dapat_ditentukan).
        kelompok: "balita" atau "ibu".

    Raises:
        ContentKeyError: status atau kelompok tidak dikenal. Pesannya
            menyebut argumen bermasalah dan daftar yang tersedia.
    """
    if kelompok not in REKOMENDASI:
        raise ContentKeyError(
            f"Kelompok tidak dikenal: {kelompok!r}. "
            f"Kelompok yang tersedia: {', '.join(KELOMPOK)}."
        )
    try:
        return REKOMENDASI[kelompok][status]
    except (KeyError, TypeError):
        raise ContentKeyError(
            f"Status tidak dikenal: {status!r}. "
            f"Status yang tersedia: {', '.join(STATUS)}."
        ) from None
