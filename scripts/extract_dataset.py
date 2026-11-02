#!/usr/bin/env python3
"""
extract_dataset.py
------------------
Ekstraksi fitur wajah untuk dataset, dengan label biner dari LiLA
(lingkar lengan atas), bukan dari BMI.

Skrip ini PEMAKAI, bukan implementasi ulang: seluruh perhitungan landmark,
quality gate, dan ekstraksi fitur diambil apa adanya dari app/pipeline.py
dengan urutan persis seperti process_face_photo(). Kalau logika pipeline
berubah, hasil ekstraksi dataset ikut berubah tanpa perlu menyentuh skrip ini.

Alur per gambar:
    decode -> detect_landmarks -> sample_skin_lab -> assess_quality
           -> extract_{hair,sclera,geometry}_features -> score_* -> combine_status

Setiap gambar yang tidak lolos dicatat di reports/rejected.csv lengkap dengan
tahap gagalnya dan alasannya. Tidak ada gambar yang dibuang diam-diam: kalau
disimpan fiturnya tidak lengkap, itu ditandai lewat kolom `available`,
bukan lewat dihapus barisnya.

Contoh:
    python scripts/extract_dataset.py \
        --images data/images \
        --labels data/labels.csv \
        --out-features data/features.csv \
        --rejected reports/rejected.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from app import pipeline  # noqa: E402
from app.thresholds import (  # noqa: E402
    BALITA_MAX_MONTH,
    BALITA_MIN_MONTH,
    LILA_BALITA_AMAN_CM,
    LILA_BALITA_MERAH_CM,
    LILA_IBU_AMAN_CM,
    LILA_MAX_CM,
    LILA_MIN_CM,
)


# Kolom identitas + label anthropometri (LiLA, bukan BMI)
META_COLUMNS = [
    "subject_id",
    "image_path",        # apa adanya dari labels.csv -> join-friendly
    "image_abs_path",    # hasil resolve, untuk debugging
    "kategori",          # balita / ibu_hamil / ibu_menyusui
    "usia",              # nilai apa adanya (bulan untuk balita, tahun untuk ibu)
    "usia_bulan",        # hanya berarti untuk balita (cek 6-59 bulan)
    "jenis_kelamin",
    "lila_cm",           # lingkar lengan atas: sumber label
    "berat_kg",
    "panjang_tinggi_cm", # opsional, disimpan apa adanya; z-score BELUM dihitung
    "kategori_risiko",   # merah / kuning / hijau
    "label_method",      # lila_balita / lila_ibu
    "label_underweight",
]

# Kolom fitur. Dipisah per grup supaya CSV-nya enak dibaca dan supaya
# perubahan satu grup tidak menggeser kolom grup lain.
FEATURE_COLUMNS = [
    # pose (proxy, untuk audit gate)
    "yaw_proxy",
    "roll_deg",
    "pitch_ratio",
    "face_height_px",
    "face_width_px",
    # quality
    "quality_ok",
    "quality_score",
    "sharpness",
    "n_faces",
    # rambut
    "hair_available",
    "hair_l",
    "hair_chroma",
    "hair_skin_l",
    "hair_skin_chroma",
    "hair_l_delta",
    "hair_chroma_ratio",
    "hair_foreground_ratio",
    "hair_n_pixels",
    # sklera
    "sclera_available",
    "sclera_a_star",
    "sclera_skin_a_star",
    "sclera_skin_a_star_raw",
    "a_star_ratio",
    "sclera_n_eyes",
    "sclera_n_pixels",
    # geometri
    "geometry_available",
    "cheek_width_px",
    "jaw_width_px",
    "cheek_to_jaw_ratio",
    "lower_face_height_ratio",
    # scoring
    "score_hair",
    "score_sclera",
    "score_geometry",
    "status_cv",
    "cv_score",
    "notes",
]

OUTPUT_COLUMNS = META_COLUMNS + FEATURE_COLUMNS

REJECTED_COLUMNS = [
    "subject_id",
    "image_path",
    "image_abs_path",
    "stage",
    "reason",
    "hint",
]


class Rejected(Exception):
    """Gagal di tahap tertentu, dengan alasan yang bisa ditulis ke laporan."""

    def __init__(self, stage: str, reason: str, hint: str = ""):
        super().__init__(reason)
        self.stage = stage
        self.reason = reason
        self.hint = hint


# ---------------------------------------------------------------------------
# Parsing angka
# ---------------------------------------------------------------------------

def to_float(raw, field: str) -> float:
    """Parse angka dari CSV; kosong/tidak valid -> Rejected(stage='label').

    float("NaN") dan float("inf") secara teknis valid di Python, tapi tidak
    boleh lolos: LiLA NaN akan menghasilkan label yang Seenaknya dan baris
    seperti itu diam-diam merusak dataset.
    """
    if raw is None or str(raw).strip() == "":
        raise Rejected("label", f"kolom '{field}' kosong")
    try:
        value = float(str(raw).strip())
    except ValueError:
        raise Rejected("label", f"kolom '{field}' bukan angka: {raw!r}") from None
    if not math.isfinite(value):
        raise Rejected("label", f"kolom '{field}' bukan angka berhingga: {raw!r}")
    return value


# ---------------------------------------------------------------------------
# Tabel WHO 2007 BMI-for-age (5-19 tahun), sumber: data/who/
#
# Tabel ini tidak pernah dibuat sendiri. Kalau file-nya tidak ada, skrip
# berhenti dengan pesan yang jelas: label anak-anak mustahil dihitung tanpa
# L/M/S resmi, dan menebaknya akan menghasilkan label yang salah secara
# diam-diam.
# ---------------------------------------------------------------------------

WHO_DIR = ROOT / "data" / "who"
WHO_FILES = {
    "boys": WHO_DIR / "bmi_for_age_boys.csv",
    "girls": WHO_DIR / "bmi_for_age_girls.csv",
}
# Bulan kalendar (0-228 untuk bayi 19 tahun)
WHO_MIN_MONTH = 61            # >= 5 tahun
WHO_MAX_MONTH = 228           # tepat 19 tahun
# usia_bulan >= 229 dianggap dewasa. Untuk bulan bulat ini identik dengan
# "> 228"; bentuk > WHO_MAX_MONTH yang dipakai supaya bulan pecahan
# (mis. 228.6 dari usia 18.6 tahun) tidak jatuh di celah antara 228 dan 229
# dan tertolak sebagai "di luar rentang".
WHO_ADULT_MIN = WHO_MAX_MONTH + 1
WHO_REQUIRED_COLUMNS = ("Month", "L", "M", "S")
# TabelWHO-nya memang memuat bulan 61..228; kalau rentangnya tidak lengkap,
# lookup akan gagal di kemudian hari dengan pesan yang membingungkan.
WHO_EXPECTED_ROWS = WHO_MAX_MONTH - WHO_MIN_MONTH + 1


class WhoTableError(RuntimeError):
    """Tabel WHO tidak ada atau rusak. Fatal: skrip berhenti, bukan ditebak."""


def load_who_table(path: Path) -> dict[int, tuple[float, float, float]]:
    """Baca satu tabel WHO -> {bulan: (L, M, S)}.

    Validasi yang dijalankan (urutan pesan sengaja dari yang paling mudah
    diperbaiki):
      * file ada
      * kolom wajib Month, L, M, S ada
      * L/M/S terbaca sebagai desimal, semuanya berhingga (bukan NaN/inf)
      * S != 0 (pembagi pada rumus LMS)
      * bulan unik, berurutan tanpa celah, dan menutup rentang 61-228

    Kolom SD* sengaja tidak dipakai: angka itu hanya pembulatan dari LMS,
    dipakai di test sebagai pembanding.
    """
    if not path.exists():
        raise WhoTableError(
            f"Tabel WHO BMI-for-age tidak ditemukan: {path}\n"
            "Skrip TIDAK membuat tabel pengganti. Unduh WHO 2007 BMI-for-age\n"
            "(usia 5-19 tahun) lalu simpan sebagai:\n"
            f"  {WHO_FILES['boys']}\n"
            f"  {WHO_FILES['girls']}"
        )

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise WhoTableError(f"{path.name}: file kosong, tidak ada baris header")
        fields = [f.strip() for f in reader.fieldnames]
        missing = [c for c in WHO_REQUIRED_COLUMNS if c not in fields]
        if missing:
            raise WhoTableError(
                f"{path.name}: kolom wajib tidak ada: {', '.join(missing)}; "
                f"yang ditemukan: {', '.join(fields)}"
            )
        raw_rows = list(reader)

    if not raw_rows:
        raise WhoTableError(f"{path.name}: tidak ada baris data")

    table: dict[int, tuple[float, float, float]] = {}
    for lineno, raw in enumerate(raw_rows, start=2):
        row = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
        try:
            month = int(float(row["Month"]))
        except (ValueError, TypeError):
            raise WhoTableError(
                f"{path.name} baris {lineno}: Month bukan bulan bulat: {row['Month']!r}"
            ) from None
        if float(row["Month"]) != month:
            raise WhoTableError(
                f"{path.name} baris {lineno}: Month bukan bulan bulat: {row['Month']!r}"
            )

        params = []
        for col in ("L", "M", "S"):
            try:
                value = float(row[col])
            except (ValueError, TypeError):
                raise WhoTableError(
                    f"{path.name} baris {lineno}, kolom {col} bukan desimal "
                    f"yang terbaca: {row[col]!r}"
                ) from None
            if not math.isfinite(value):
                raise WhoTableError(
                    f"{path.name} baris {lineno}, kolom {col} bukan angka berhingga: "
                    f"{row[col]!r}"
                )
            params.append(value)
        if params[2] == 0:
            raise WhoTableError(f"{path.name} baris {lineno}: S = 0, rumus LMS dibagi nol")

        if month in table:
            raise WhoTableError(f"{path.name} baris {lineno}: bulan {month} ganda")
        table[month] = (params[0], params[1], params[2])

    months = sorted(table)
    if months != list(range(months[0], months[0] + len(months))):
        gaps = [
            m for m in range(months[0], months[-1] + 1) if m not in table
        ]
        raise WhoTableError(
            f"{path.name}: bulan tidak berurutan, ada celah di bulan "
            f"{', '.join(map(str, gaps[:10]))}{' ...' if len(gaps) > 10 else ''}"
        )
    if len(months) != WHO_EXPECTED_ROWS or months[0] > WHO_MIN_MONTH or months[-1] < WHO_MAX_MONTH:
        raise WhoTableError(
            f"{path.name}: rentang bulan {months[0]}-{months[-1]} "
            f"({len(months)} baris) tidak menutup {WHO_MIN_MONTH}-{WHO_MAX_MONTH} "
            f"({WHO_EXPECTED_ROWS} baris)"
        )

    return table


def load_who_tables() -> dict[str, dict[int, tuple[float, float, float]]]:
    """Baca kedua tabel WHO. Siapa pun gagal -> skrip berhenti."""
    return {sex: load_who_table(path) for sex, path in WHO_FILES.items()}


def who_zscore(bmi: float, L: float, M: float, S: float) -> float:
    """z-score BMI-for-age memakai rumus LMS.

        L != 0 : z = ((BMI/M)^L - 1) / (L * S)
        L == 0 : z = ln(BMI/M) / S

    Cabang L=0 adalah batas dari rumus pertama saat L mendekati 0, jadi bukan
    rumus kedua yang dikarang. Data WHO yang dipakai di sini ber L negatif
    semua, tapi cabang ini tetap diuji karena tabel lain bisa saja L=0.
    """
    if not (M > 0 and S > 0):
        raise WhoTableError(f"parameter WHO tidak valid: L={L}, M={M}, S={S}")
    if not (bmi > 0 and math.isfinite(bmi)):
        raise Rejected("label", f"BMI tidak valid untuk z-score: {bmi!r}")
    if L == 0:
        return math.log(bmi / M) / S
    return ((bmi / M) ** L - 1.0) / (L * S)


def lms_for_month(table: dict[int, tuple[float, float, float]],
                  month: float) -> tuple[float, float, float]:
    """Ambil (L, M, S) pada `month`.

    Bulan bulat diambil persis dari barisnya. Bulan pecahan (dari usia desimal)
    diinterpolasi linear antar dua bulan yang berdampingan, supaya tidak perlu
    membulatkan ke bulan terdekat. Luar tabel = error; ekstrapolasi dilarang.
    """
    if month < WHO_MIN_MONTH or month > WHO_MAX_MONTH:
        raise WhoTableError(f"bulan {month} di luar tabel WHO {WHO_MIN_MONTH}-{WHO_MAX_MONTH}")
    lo = math.floor(month)
    hi = math.ceil(month)
    if lo == hi:
        return table[lo]
    # diinterpolasi hanya kalau kedua ujungnya ada; kalau tidak, berarti
    # rentang tabel memang belum menutup bulan ini -> sudah divalidasi di load
    if lo not in table or hi not in table:
        raise WhoTableError(f"bulan {lo}/{hi} tidak ada di tabel WHO")
    t = month - lo
    return tuple(a + (b - a) * t for a, b in zip(table[lo], table[hi]))  # type: ignore[return-value]


def categorize_z(z: float) -> str:
    """Kategori WHO BMI-for-age. Ambangnya literal dari definisinya."""
    if z < -3:
        return "severe_thinness"
    if z < -2:
        return "thinness"
    return "normal"


# Terima nilai yang konsisten; selain ini ditolak supaya typo tidak berubah
# diam-diam jadi kategori jenis kelamin yang salah.
SEX_TO_TABLE = {
    "l": "boys", "laki": "boys", "laki-laki": "boys", "male": "boys", "m": "boys",
    "p": "girls", "perempuan": "girls", "pr": "girls", "female": "girls", "f": "girls",
}
SEX_ACCEPTED = "L / P / laki-laki / perempuan (juga laki, male, female)"


def parse_jenis_kelamin(raw: str) -> str:
    """-> 'boys' | 'girls'. Selain nilai yang konsisten -> Rejected."""
    value = re.sub(r"[\s_]+", "-", str(raw or "").strip().lower())
    if value in SEX_TO_TABLE:
        return SEX_TO_TABLE[value]
    raise Rejected(
        "label",
        f"jenis_kelamin tidak dikenali: {raw!r} (terima {SEX_ACCEPTED})",
    )


# ---------------------------------------------------------------------------
# Label LiLA (lingkar lengan atas) per kategori subjek
#
# Label TIDAK dihitung dari BMI. Tabel WHO BMI-for-age di atas tetap ada dan
# diuji, tapi tidak dipakai di alur ini.
# ---------------------------------------------------------------------------

KATEGORI_BALITA = "balita"
KATEGORI_HAMIL = "ibu_hamil"
KATEGORI_MENYUSUI = "ibu_menyusui"
KATEGORI_VALID = (KATEGORI_BALITA, KATEGORI_HAMIL, KATEGORI_MENYUSUI)

# Angka ambang LiLA dan rentang usia balita TIDAK ditulis ulang di sini:
# semuanya diimpor dari app/thresholds.py, modul yang sama yang dipakai
# app/risk_status.py. Supaya label dataset dan status di layar tidak bisa
# berbeda. Test test_risk_status.py memeriksa bahwa kedua modul membaca
# konstanta yang sama (identik, bukan salinan).

# Alasan ini persis diminta untuk kasus usia di luar rentang.
ALASAN_BALITA_LUAR_RENTANG = "LiLA tidak valid di luar 6-59 bulan"


def parse_kategori(raw: str) -> str:
    """-> 'balita' | 'ibu_hamil' | 'ibu_menyusui'. Selain itu -> Rejected."""
    text = str(raw or "").strip()
    if not text:
        raise Rejected(
            "label",
            "kategori kosong (harus salah satu dari: "
            f"{', '.join(KATEGORI_VALID)})",
        )
    value = re.sub(r"[\s-]+", "_", text.lower())
    if value in KATEGORI_VALID:
        return value
    raise Rejected(
        "label",
        f"kategori tidak dikenali: {raw!r} "
        f"(harus salah satu dari: {', '.join(KATEGORI_VALID)})",
    )


def risiko_balita(lila_cm: float) -> str:
    """Merah < 11.5, kuning 11.5 sampai < 12.5, hijau >= 12.5."""
    if lila_cm < LILA_BALITA_MERAH_CM:
        return "merah"
    if lila_cm < LILA_BALITA_AMAN_CM:
        return "kuning"
    return "hijau"


def risiko_ibu(lila_cm: float) -> str:
    """Merah < 23.5, hijau >= 23.5. Ibu hamil/menyusui tidak ada kategori kuning."""
    return "merah" if lila_cm < LILA_IBU_AMAN_CM else "hijau"


def resolve_age(row: dict, kategori: str) -> tuple[float | None, str]:
    """Tentukan usia balita dalam bulan.

    -> (usia_bulan_dipakai, nilai_kolom_usia)

    Satuan `usia` bergantung kategori, sesuai skema labels.csv:
      * balita : `usia` sudah dalam BULAN (atau pakai kolom `usia_bulan`)
      * ibu    : `usia` dalam TAHUN dan TIDAK dipakai menentukan label,
                 jadi usia_bulan dikembalikan None kecuali kolom
                 `usia_bulan` memang diisi.

    Kolom `usia_bulan` selalu didahulukan kalau ada isinya.
    """
    raw_bulan = str(row.get("usia_bulan") or "").strip()
    raw_usia = str(row.get("usia") or "").strip()

    if raw_bulan:
        months = to_float(raw_bulan, "usia_bulan")
        if months < 0:
            raise Rejected("label", f"usia_bulan tidak masuk akal: {months}")
        return months, raw_usia

    if not raw_usia:
        if kategori == KATEGORI_BALITA:
            raise Rejected(
                "label",
                "usia kosong dan tidak ada kolom usia_bulan "
                "(untuk balita usia diisi dalam bulan)",
            )
        return None, ""

    value = to_float(raw_usia, "usia")
    if value < 0:
        raise Rejected("label", f"usia tidak masuk akal: {value}")

    if kategori == KATEGORI_BALITA:
        return value, raw_usia     # sudah bulan
    return None, raw_usia          # ibu: tahun, tidak ada aturan usia


def build_label_row(row: dict, images_dir: Path) -> dict:
    """Hitung label LiLA dari satu baris labels.csv.

      * balita         -> usia_bulan 6-59, label=1 bila LiLA < 12.5
      * ibu_hamil      -> label=1 bila LiLA < 23.5
      * ibu_menyusui   -> label=1 bila LiLA < 23.5

    BMI tidak dipakai di sini sama sekali, termasuk untuk ibu hamil.
    """
    subject_id = (row.get("subject_id") or "").strip()
    if not subject_id:
        raise Rejected("label", "subject_id kosong")

    raw_path = (row.get("image_path") or "").strip()
    if not raw_path:
        raise Rejected("label", "image_path kosong")

    # image_path relatif terhadap folder gambar; absolut dipakai apa adanya
    p = Path(raw_path)
    image_path = p if p.is_absolute() else (images_dir / p)

    kategori = parse_kategori(row.get("kategori"))

    lila_cm = to_float(row.get("lila_cm"), "lila_cm")
    if not (LILA_MIN_CM <= lila_cm <= LILA_MAX_CM):
        raise Rejected(
            "label",
            f"lila_cm di luar rentang wajar {LILA_MIN_CM:g}-{LILA_MAX_CM:g} cm: "
            f"{lila_cm}",
        )

    berat_kg = to_float(row.get("berat_kg"), "berat_kg")
    if berat_kg <= 0:
        raise Rejected("label", f"berat_kg tidak masuk akal: {berat_kg}")

    # Disimpan apa adanya. Z-score-nya sengaja belum dihitung.
    raw_panjang = str(row.get("panjang_tinggi_cm") or "").strip()
    panjang_tinggi_cm: float | str = ""
    if raw_panjang:
        panjang_tinggi_cm = to_float(raw_panjang, "panjang_tinggi_cm")
        if panjang_tinggi_cm <= 0:
            raise Rejected(
                "label", f"panjang_tinggi_cm tidak masuk akal: {panjang_tinggi_cm}"
            )

    jenis_kelamin = str(row.get("jenis_kelamin") or "").strip()
    usia_bulan, usia_value = resolve_age(row, kategori)

    if kategori == KATEGORI_BALITA:
        # Usia dulu sebelum jenis kelamin: usia menentukan apakah LiLA-nya
        # sah dipakai sama sekali.
        if usia_bulan is None or not (BALITA_MIN_MONTH <= usia_bulan <= BALITA_MAX_MONTH):
            raise Rejected("label", ALASAN_BALITA_LUAR_RENTANG)
        if not jenis_kelamin:
            raise Rejected("label", "jenis_kelamin kosong (wajib untuk balita)")
        label_method = "lila_balita"
        kategori_risiko = risiko_balita(lila_cm)
        label_underweight = 1 if lila_cm < LILA_BALITA_AMAN_CM else 0
    else:
        # Ibu hamil / menyusui: usia dan jenis kelamin tidak mengubah label.
        label_method = "lila_ibu"
        kategori_risiko = risiko_ibu(lila_cm)
        label_underweight = 1 if lila_cm < LILA_IBU_AMAN_CM else 0

    return {
        "subject_id": subject_id,
        "image_path": raw_path,
        "image_abs_path": str(image_path),
        "kategori": kategori,
        "usia": usia_value,
        "usia_bulan": "" if usia_bulan is None else round(usia_bulan, 2),
        "jenis_kelamin": jenis_kelamin,
        "lila_cm": round(lila_cm, 4),
        "berat_kg": berat_kg,
        "panjang_tinggi_cm": panjang_tinggi_cm,
        "kategori_risiko": kategori_risiko,
        "label_method": label_method,
        "label_underweight": label_underweight,
    }


# ---------------------------------------------------------------------------
# Pipeline: reuse fungsi yang sudah ada, jangan diimplementasikan ulang
# ---------------------------------------------------------------------------

def analyze_image(image_bgr: np.ndarray) -> dict:
    """landmark -> quality -> features -> scoring.

    Urutannya sama persis dengan pipeline.process_face_photo(); fungsi yang
    dipanggil juga sama. Bedanya hanya: yang dikembalikan dict CSV, bukan
    CVAnalysisResult, dan errornya dibungkus jadi Rejected supaya bisa dilog.
    """
    # Step 1+2: landmark
    try:
        landmarks_px, n_faces = pipeline.detect_landmarks(image_bgr)
    except pipeline.FaceNotFoundError as exc:
        raise Rejected("no_face", str(exc)) from None
    # ModelMissingError sengaja dibiarkan naik: tanpa model, NOTHING bisa
    # diproses. Mencontinue dengan menandai semua gambar 'error' hanya
    # menghasilkan 500 baris failures yang berguna.

    pose = pipeline.estimate_pose(landmarks_px)

    # Warna kulit dipakai untuk gate exposure DAN sebagai acuan fitur warna
    skin_lab = pipeline.sample_skin_lab(image_bgr, landmarks_px)

    # Step 0: quality gate
    try:
        quality = pipeline.assess_quality(image_bgr, landmarks_px, n_faces, skin_lab)
    except pipeline.ImageRejectedError as exc:
        raise Rejected("quality", exc.reason, exc.detail) from None

    # Step 4: fitur
    notes = []
    if skin_lab is None:
        notes.append("Sampling warna kulit gagal; fitur warna dinonaktifkan.")
        hair_features = {"available": False, "reason": "warna kulit tidak tersedia"}
        sclera_features = {"available": False, "reason": "warna kulit tidak tersedia"}
    else:
        hair_features = pipeline.extract_hair_features(image_bgr, landmarks_px, skin_lab)
        sclera_features = pipeline.extract_sclera_features(image_bgr, landmarks_px, skin_lab)
    geometry_features = pipeline.extract_geometry_features(landmarks_px)

    # Step 5: scoring
    indicators = [
        pipeline.score_hair(hair_features),
        pipeline.score_sclera(sclera_features),
        pipeline.score_geometry(geometry_features),
    ]
    status, score = pipeline.combine_status(indicators)
    if status == "tidak_valid":
        notes.append("Status TIDAK_VALID: ada indikator yang tidak bisa dievaluasi.")

    return _flatten(landmarks_px, n_faces, pose, quality, sharpness_of(quality),
                    hair_features, sclera_features, geometry_features,
                    indicators, status, score, notes)


def sharpness_of(quality: "pipeline.QualityReport"):
    """Ambil sharpness dari checks quality gate (sudah dihitung di dalam)."""
    return quality.checks.get("sharpness")


def _flatten(landmarks_px, n_faces, pose, quality, sharpness,
             hair, sclera, geometry, indicators, status, score, notes) -> dict:
    """Ratakan semua keluaran pipeline menjadi satu baris CSV."""
    by_name = {i.name: i for i in indicators}
    checks = quality.checks

    def num(value, digits=4):
        if value is None:
            return ""
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return ""
        return round(float(value), digits)

    def avail(feature):
        return int(bool(feature.get("available")))

    row = {
        "yaw_proxy": num(pose["yaw_proxy"]),
        "roll_deg": num(pose["roll_deg"], 3),
        "pitch_ratio": num(pose["pitch_ratio"]),
        "face_height_px": num(pose["face_height_px"], 2),
        "face_width_px": num(pose["face_width_px"], 2),

        "quality_ok": int(bool(quality.ok)),
        "quality_score": num(quality.score, 4),
        "sharpness": num(sharpness, 2),
        "n_faces": int(n_faces),

        "hair_available": avail(hair),
        "hair_l": num(hair.get("hair_l")),
        "hair_chroma": num(hair.get("hair_chroma")),
        "hair_skin_l": num(hair.get("skin_l")),
        "hair_skin_chroma": num(hair.get("skin_chroma")),
        "hair_l_delta": num(hair.get("l_delta")),
        "hair_chroma_ratio": num(hair.get("chroma_ratio")),
        "hair_foreground_ratio": num(hair.get("foreground_ratio")),
        "hair_n_pixels": int(hair.get("n_pixels", 0) or 0),

        "sclera_available": avail(sclera),
        "sclera_a_star": num(sclera.get("sclera_a_star")),
        "sclera_skin_a_star": num(sclera.get("skin_a_star")),
        "sclera_skin_a_star_raw": num(sclera.get("skin_a_star_raw")),
        "a_star_ratio": num(sclera.get("a_star_ratio")),
        "sclera_n_eyes": int(sclera.get("n_eyes", 0) or 0),
        "sclera_n_pixels": int(sclera.get("n_pixels", 0) or 0),

        "geometry_available": avail(geometry),
        "cheek_width_px": num(geometry.get("cheek_width_px"), 2),
        "jaw_width_px": num(geometry.get("jaw_width_px"), 2),
        "cheek_to_jaw_ratio": num(geometry.get("cheek_to_jaw_ratio")),
        "lower_face_height_ratio": num(geometry.get("lower_face_height_ratio")),

        "score_hair": num(by_name["rambut"].normalized),
        "score_sclera": num(by_name["kepucatan_mata"].normalized),
        "score_geometry": num(by_name["geometri_wajah"].normalized),
        "status_cv": status,
        "cv_score": num(score),
        "notes": " | ".join(notes),
    }
    return row


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_labels(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"labels.csv tidak ditemukan: {path}")
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"labels.csv kosong (tidak ada data baris): {path}")

    required = {"subject_id", "image_path", "kategori", "lila_cm", "berat_kg"}
    missing = required - set(rows[0].keys())
    if missing:
        raise SystemExit(
            f"labels.csv kurang kolom wajib: {', '.join(sorted(missing))}\n"
            f"kolom yang ada: {', '.join(rows[0].keys())}"
        )
    return rows


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" supaya tidak ada baris kosong kelar di antara baris data
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Ringkasan
# ---------------------------------------------------------------------------

def print_summary(accepted: list, rejected: list,
                  features_path: Path, rejected_path: Path) -> None:
    total = len(accepted) + len(rejected)
    print()
    print("=" * 68)
    print("RINGKASAN EKSTRAKSI DATASET")
    print("=" * 68)
    print(f"Total baris di labels.csv : {total}")
    print(f"Diterima (lolos gate)     : {len(accepted)}")
    print(f"Ditolak                  : {len(rejected)}")
    if total:
        print(f"Rate penerimaan           : {len(accepted)/total:.1%}")

    print()
    print("METODE LABEL (diterima saja)")
    if not accepted:
        print("  (tidak ada data diterima)")
    else:
        by_method = Counter(r["label_method"] for r in accepted)
        for method, count in by_method.most_common():
            print(f"  {method:12s} {count:5d}")

        print()
        print("KATEGORI SUBJEK (diterima)")
        for key, count in Counter(r["kategori"] for r in accepted).most_common():
            print(f"  {key:16s} {count:5d}")

        print()
        print("DISTRIBUSI LABEL (diterima saja)")
        ambang = {"lila_balita": f"LiLA < {LILA_BALITA_AMAN_CM:g} cm",
                  "lila_ibu": f"LiLA < {LILA_IBU_AMAN_CM:g} cm"}
        dist = Counter((r["label_method"], r["label_underweight"]) for r in accepted)
        for (method, label), count in sorted(dist.items()):
            name = ambang.get(method, method) if label else "tidak berisiko"
            pct = count / len(accepted) * 100
            print(f"  {method:10s} label={label} {name:24s} {count:5d}  ({pct:.1f}%)")

        print()
        print("KATEGORI RISIKO LiLA (diterima)")
        for key, count in Counter(r["kategori_risiko"] for r in accepted).most_common():
            print(f"  {key:20s} {count:5d}")

        avail = Counter()
        for r in accepted:
            for key in ("hair_available", "sclera_available", "geometry_available"):
                if r[key] == 1:
                    avail[key] += 1
        print()
        print("KETERSEDIAAN FITUR (diterima)")
        print(f"  hair_available     : {avail['hair_available']}/{len(accepted)}")
        print(f"  sclera_available   : {avail['sclera_available']}/{len(accepted)}")
        print(f"  geometry_available : {avail['geometry_available']}/{len(accepted)}")

        status = Counter(r["status_cv"] for r in accepted)
        print()
        print("STATUS CV (proxy rule-based, BUKAN diagnosis)")
        for key, count in status.most_common():
            print(f"  {key:12s} {count:5d}")

    print()
    print("ALASAN PENOLAKAN (paling sering)")
    if not rejected:
        print("  (tidak ada)")
    else:
        by_stage = Counter(r["stage"] for r in rejected)
        for stage, count in by_stage.most_common():
            print(f"  [{stage}] total {count}")
            reasons = Counter(r["reason"] for r in rejected if r["stage"] == stage)
            for reason, n in reasons.most_common(5):
                print(f"      {n:5d}x {reason}")

    print()
    print(f"Tulis  : {features_path}")
    print(f"Tulis  : {rejected_path}")
    print("=" * 68)
    print("CATATAN: label ditetapkan dari LiLA, BUKAN dari BMI.")
    print(f"  balita         : LiLA < {LILA_BALITA_AMAN_CM:g} cm -> label 1 "
          f"(merah < {LILA_BALITA_MERAH_CM:g} cm)")
    print(f"  ibu hamil/menyusui : LiLA < {LILA_IBU_AMAN_CM:g} cm -> label 1")
    print("LiLA adalah alat skrining, bukan diagnosis kekurangan gizi.")
    print("=" * 68)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Ekstraksi fitur wajah + label LiLA dari dataset foto.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--images", type=Path, default=ROOT / "data" / "images",
                    help="folder gambar")
    ap.add_argument("--labels", type=Path, default=ROOT / "data" / "labels.csv",
                    help="CSV: subject_id, image_path, kategori, lila_cm, berat_kg, "
                         "usia/[usia_bulan], [jenis_kelamin], [panjang_tinggi_cm]")
    ap.add_argument("--out-features", type=Path, default=ROOT / "data" / "features.csv",
                    help="tujuan CSV fitur")
    ap.add_argument("--rejected", type=Path, default=ROOT / "reports" / "rejected.csv",
                    help="tujuan CSV gambar yang ditolak")
    ap.add_argument("--limit", type=int, default=None,
                    help="batasi jumlah baris (untuk smoke test)")
    args = ap.parse_args(argv)

    if not args.images.is_dir():
        raise SystemExit(f"folder gambar tidak ditemukan: {args.images}")

    rows = read_labels(args.labels)
    if args.limit:
        rows = rows[: args.limit]

    print(f"Model   : {pipeline.config.MODEL_PATH}")
    print(f"Gambar  : {args.images}")
    print(f"Labels  : {args.labels} ({len(rows)} baris)")
    print(f"Ambang LiLA: balita {LILA_BALITA_AMAN_CM:g} cm "
          f"(merah < {LILA_BALITA_MERAH_CM:g}), ibu {LILA_IBU_AMAN_CM:g} cm")

    # Cek model sekali di awal supaya gagal keras sebelum processing berjalan
    if not pipeline.config.MODEL_PATH.exists():
        raise SystemExit(
            f"Model tidak ditemukan: {pipeline.config.MODEL_PATH}\n"
            "Jalankan: python download_model.py"
        )

    accepted: list[dict] = []
    rejected: list[dict] = []

    bar = tqdm(rows, desc="ekstrak", unit="img", ncols=72)
    for raw_row in bar:
        # 1. label
        try:
            meta = build_label_row(raw_row, args.images)
        except Rejected as exc:
            rejected.append({
                "subject_id": (raw_row.get("subject_id") or "").strip(),
                "image_path": (raw_row.get("image_path") or "").strip(),
                "stage": exc.stage, "reason": exc.reason, "hint": exc.hint,
            })
            continue

        # 2. decode
        path = Path(meta["image_abs_path"])
        if not path.exists():
            rejected.append({**meta, "stage": "missing_file",
                             "reason": f"gambar tidak ditemukan: {meta['image_abs_path']}",
                             "hint": ""})
            continue
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            rejected.append({**meta, "stage": "decode",
                             "reason": f"gambar tidak bisa didekode: {meta['image_abs_path']}",
                             "hint": "mungkin file bukan gambar atau rusak"})
            continue

        # 3. pipeline
        try:
            features = analyze_image(image_bgr)
        except Rejected as exc:
            rejected.append({**meta, "stage": exc.stage,
                             "reason": exc.reason, "hint": exc.hint})
            bar.set_postfix(rej=len(rejected))
            continue
        except Exception as exc:  # noqa: BLE001
            # Satu gambar rusak tidak boleh menghentikan seluruh ekstraksi
            rejected.append({**meta, "stage": "error",
                             "reason": f"{type(exc).__name__}: {exc}",
                             "hint": traceback.format_exc(limit=3).replace("\n", " | ")})
            bar.set_postfix(rej=len(rejected))
            continue

        accepted.append({**meta, **features})
        bar.set_postfix(ok=len(accepted), rej=len(rejected))

    write_csv(args.out_features, OUTPUT_COLUMNS, accepted)
    write_csv(args.rejected, REJECTED_COLUMNS, rejected)

    print_summary(accepted, rejected, args.out_features, args.rejected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())