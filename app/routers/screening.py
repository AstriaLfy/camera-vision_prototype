"""
screening.py
------------
Endpoint skrining: upload foto wajah + data antropometri -> satu respons.

Rute (prefix `/api/v1`):

  * POST /screening/analyze  -> analisis lengkap (ringkasan + pengukuran +
                                data pendukung visual)
  * POST /debug-landmarks    -> overlay titik landmark (validasi visual saja)

Logika decode/upload ada di sini karena khusus domain screening; main.py hanya
mendaftarkan router.

ATURAN PENTING
  * `status` / `ringkasan` / `pengukuran` MURNI dari LiLA
    (app/risk_status.py). Foto tidak pernah mengubah status.
  * `status_cv` dan skor apa pun dari app/pipeline.py TIDAK dipakai dan TIDAK
    pernah muncul di respons. Modul ini hanya memakai fungsi fitur mentah
    pipeline (detect_landmarks, sample_skin_lab, assess_quality, extract_*),
    bukan process_face_photo / combine_status.
  * Foto tidak layak atau wajah tidak terdeteksi -> HTTP 200 dengan
    `kualitas_foto.ok = false`, `data_pendukung = []`, status tetap dikirim.
  * STATELESS: foto diproses di memori, tidak ditulis ke disk/DB, dan tidak
    ada nama/foto/nilai ukur subjek yang masuk log.

Jalankan:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import io
from typing import Final, Literal

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .. import __version__, content, indicators, pipeline, risk_status
from ..deps import CurrentUser
from ..risk_status import IndikatorTerukur
from ..settings import get_settings

router = APIRouter(prefix="/api/v1", tags=["screening"])

#: Router terpisah untuk endpoint debug. main.py mendaftarkannya HANYA bila
#: settings.enable_debug_endpoints true, supaya rutenya benar-benar tidak ada
#: (bukan sekadar 403) saat API dipublikasikan. Logika endpoint tidak berubah.
debug_router = APIRouter(prefix="/api/v1", tags=["debug"])

MODEL_MISSING_HINT = (
    "Model FaceLandmarker belum tersedia di server. Jalankan "
    "`python download_model.py` sekali pada mesin ini."
)

#: Kategori subjek yang diterima (dipakai juga sebagai validasi Pydantic).
Kategori = Literal["balita", "ibu_hamil", "ibu_menyusui"]

#: Sumber item pengukuran: nilai diukur langsung dari subjek.
SUMBER_TERUKUR: Final[str] = "terukur"

#: Alasan penolakan pipeline (string stabil dari app/pipeline.py) -> kunci
#: pesan pengguna di app/content.py. Teks pengguna TIDAK ditulis di sini.
_KUNCI_FOTO_DARI_ALASAN: Final[dict[str, str]] = {
    "Foto terlalu buram.": "foto.buram",
    "Wajah terlalu kecil di frame.": "foto.jarak_terlalu_jauh",
    "Kepala tidak menghadap kamera lurus.": "foto.pose_miring",
    "Kepala miring.": "foto.pose_miring",
    "Kepala menengadah atau mendongak.": "foto.pose_miring",
    "Pencahayaan tidak sesuai protokol.": "foto.pencahayaan_kurang",
}
_AKHIRAN_ALASAN_MAJEMUK: Final[str] = "wajah terdeteksi dalam satu foto."
_KUNCI_FOTO_MAJEMUK: Final[str] = "foto.beberapa_wajah"
_KUNCI_FOTO_UMUM: Final[str] = "foto.tidak_layak"


# ---------------------------------------------------------------------------
# Model respons (urutan field persis seperti kontrak)
# ---------------------------------------------------------------------------


class Ringkasan(BaseModel):
    status: str
    label: str
    deskripsi: str


class Subjek(BaseModel):
    kategori: Kategori
    kelompok: Literal["balita", "ibu"] | None


class PengukuranItem(IndikatorTerukur):
    """Satu pengukuran LiLA/berat; sumbernya selalu "terukur"."""

    sumber: Literal["terukur"] = SUMBER_TERUKUR


class KualitasFoto(BaseModel):
    ok: bool
    pesan: list[str]


class Meta(BaseModel):
    cv_mode: str
    versi: str


class ScreeningResponse(BaseModel):
    """Respons POST /api/v1/screening/analyze."""

    ringkasan: Ringkasan
    subjek: Subjek
    pengukuran: list[PengukuranItem]
    data_pendukung: list[indicators.Indicator]
    kualitas_foto: KualitasFoto
    rekomendasi: list[str]
    alasan: list[str]
    disclaimer: list[str]
    meta: Meta


# ---------------------------------------------------------------------------
# Decode upload
# ---------------------------------------------------------------------------


def _decode_upload(contents: bytes, content_type: str | None) -> np.ndarray:
    """Bytes -> BGR ndarray, dengan perbaikan di sini.

    Perbaikan yang dulu tidak ada (dan bikin hasil diam-diam salah):
      - batas ukuran upload  -> cegah OOM
      - batas dimensi gambar ->landmark presisi & kecepatan
      - EXIF orientation    -> cv2.imdecode() MENGABAIKAN EXIF, jadi foto
                               portrait dari HP akan diproses miring
      - downscale           -> foto 12MP tidak perlu diproses ukuran penuh

    Batas ukuran, dimensi, dan tipe file diambil dari settings (env / .env);
    default-nya jpeg/png dan angka lama di app/config.py::QUALITY.
    """
    settings = get_settings()

    max_bytes = settings.max_upload_bytes
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File terlalu besar ({len(contents)/1024/1024:.1f} MB). "
                   f"Maksimal {max_bytes/1024/1024:.0f} MB.",
        )

    allowed = settings.allowed_upload_types
    if content_type and content_type.lower() not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Content-Type '{content_type}' tidak didukung. "
                   f"Yang diterima: {', '.join(sorted(allowed))}.",
        )

    # 1. decode
    image_bgr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid.")

    # 2. EXIF orientation (harus sebelum downscale, atau sebelum apa pun)
    image_bgr = _apply_exif_orientation(contents, image_bgr)

    # 3. downscale sisi terpanjang
    h, w = image_bgr.shape[:2]
    max_dim = settings.max_dimension_px
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        image_bgr = cv2.resize(
            image_bgr, (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        # resize mengubah gamma; rapikan sedikit noise
        image_bgr = cv2.bilateralFilter(image_bgr, 5, 25, 25)

    return image_bgr


def _apply_exif_orientation(contents: bytes, image_bgr: np.ndarray) -> np.ndarray:
    """Terapkan orientasi EXIF.

    Dulu implementasi manual memetakan 2/3/4/5/6/8 satu per satu. Dua masalah:
    (a) nilai 7 (transverse) tidak ditangani sama sekali sehingga foto tersebut
    lolos dalam orientasi yang salah, dan (b) pemetaan 5 (transpose) mudah
    tertukar dengan 7. Sekarang delegasi ke Pillow ImageOps.exif_transpose
    yang menangani seluruh 8 nilai EXIF sesuai spesifikasi.

    Kalau tidak ada tag Orientation (kasus paling umum), gambar BGR hasil
    cv2.imdecode dipakai langsung supaya tidak decoding dua kali.
    """
    try:
        from PIL import Image, ExifTags, ImageOps
    except ImportError:
        return image_bgr  # fallback: tanpa koreksi EXIF

    try:
        with Image.open(io.BytesIO(contents)) as im:
            exif = im.getexif()
            orientation = exif.get(ExifTags.Base.Orientation) if exif else None
            if orientation in (None, 1):
                return image_bgr
            fixed = ImageOps.exif_transpose(im)
            if fixed is None:
                return image_bgr
            rgb = np.asarray(fixed.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        # Pillow gagal parsing -> jangan gagalkan request
        return image_bgr


# ---------------------------------------------------------------------------
# Analisis visual (fitur mentah saja, tanpa status_cv / skor)
# ---------------------------------------------------------------------------


def _analisis_visual(image_bgr: np.ndarray) -> tuple[KualitasFoto, dict | None]:
    """Jalankan gate kualitas + ekstraksi fitur mentah.

    Mengembalikan (kualitas_foto, features). `features` None bila foto tidak
    layak. TIDAK memanggil process_face_photo/combine_status, jadi status_cv
    dan skor tidak pernah dihitung.

    Raise:
        pipeline.ModelMissingError: model belum ada (ditangani pemanggil -> 503).
    """
    try:
        landmarks_px, n_faces = pipeline.detect_landmarks(image_bgr)
        skin_lab = pipeline.sample_skin_lab(image_bgr, landmarks_px)
        pipeline.assess_quality(image_bgr, landmarks_px, n_faces, skin_lab)
    except pipeline.FaceNotFoundError:
        return _tidak_layak("foto.wajah_tidak_terdeteksi"), None
    except pipeline.ImageRejectedError as exc:
        return _tidak_layak(_kunci_pesan_foto(exc.reason)), None

    # Fitur mentah: geometri selalu ada; rambut & sklera butuh warna kulit.
    features: dict = dict(pipeline.extract_geometry_features(landmarks_px))
    if skin_lab is not None:
        features.update(
            pipeline.extract_hair_features(image_bgr, landmarks_px, skin_lab)
        )
        features.update(
            pipeline.extract_sclera_features(image_bgr, landmarks_px, skin_lab)
        )
    return KualitasFoto(ok=True, pesan=[]), features


def _tidak_layak(kunci: str) -> KualitasFoto:
    return KualitasFoto(ok=False, pesan=[content.get_text(kunci)])


def _kunci_pesan_foto(reason: str) -> str:
    """Petakan alasan penolakan pipeline ke kunci pesan di app/content.py."""
    kunci = _KUNCI_FOTO_DARI_ALASAN.get(reason)
    if kunci is not None:
        return kunci
    if reason.endswith(_AKHIRAN_ALASAN_MAJEMUK):
        return _KUNCI_FOTO_MAJEMUK
    return _KUNCI_FOTO_UMUM


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/screening/analyze",
    response_model=ScreeningResponse,
    summary="Skrining LiLA + data pendukung visual",
    description=(
        "Menghitung status risiko MURNI dari LiLA, lalu (kalau foto layak) "
        "menambahkan data pendukung visual eksperimental. Foto tidak pernah "
        "mengubah status. Foto tidak layak tidak menggagalkan permintaan: "
        "kualitas_foto.ok bernilai false dan status tetap dikembalikan."
    ),
)
async def analyze_screening(
    current_user: CurrentUser = None,
    foto: UploadFile = File(..., description="Foto wajah subjek (jpeg/png)"),
    kategori: Kategori = Form(..., description="balita | ibu_hamil | ibu_menyusui"),
    lila_cm: float = Form(..., gt=0, description="Lingkar lengan atas (cm)"),
    berat_kg: float = Form(..., gt=0, description="Berat badan (kg)"),
    usia_bulan: float | None = Form(None, ge=0, description="Usia balita (bulan)"),
    usia: float | None = Form(None, ge=0, description="Usia ibu (tahun)"),
    jenis_kelamin: str | None = Form(None),
    panjang_tinggi_cm: float | None = Form(None, ge=0),
):
    """Skrining dari data antropometri + foto wajah.

    `current_user` memanggil dependency autentikasi yang belum aktif (selalu
    None) supaya auth tinggal diisi di app/deps.py tanpa menyentuh fungsi ini.

    `usia` (ibu) diterima sebagai data subjek, tapi status tetap murni LiLA,
    jadi nilainya tidak dipakai untuk menghitung status.
    """
    settings = get_settings()

    if settings.cv_mode == "model":
        raise HTTPException(
            status_code=501,
            detail=content.get_text("endpoint.cv_mode_model"),
        )

    # Status murni LiLA -- tidak melihat foto sama sekali.
    risk = risk_status.evaluate_risk(
        kategori=kategori,
        lila_cm=lila_cm,
        berat_kg=berat_kg,
        usia_bulan=usia_bulan,
        panjang_tinggi_cm=panjang_tinggi_cm,
        jenis_kelamin=jenis_kelamin,
    )

    # Data pendukung visual (hanya kalau mode indikator aktif).
    if settings.cv_mode == "off":
        kualitas = KualitasFoto(ok=True, pesan=[])
        data_pendukung: list[indicators.Indicator] = []
    else:
        contents = await foto.read()
        image_bgr = _decode_upload(contents, foto.content_type)
        # CPU-bound & blocking (OpenCV + MediaPipe): jangan blokir event loop.
        try:
            kualitas, features = await run_in_threadpool(
                _analisis_visual, image_bgr
            )
        except pipeline.ModelMissingError as exc:
            # Pesan aman: jangan bocorkan path model internal ke klien.
            raise HTTPException(
                status_code=503, detail=MODEL_MISSING_HINT
            ) from exc
        data_pendukung = (
            indicators.build_indicators(features).items
            if features is not None
            else []
        )

    disclaimer = [content.get_text("disclaimer.umum")]
    if data_pendukung:
        disclaimer.append(content.get_text("disclaimer.visual"))

    return ScreeningResponse(
        ringkasan=Ringkasan(
            status=risk.status_akhir,
            label=risk.label_status,
            deskripsi=content.get_text(
                f"ringkasan.deskripsi.{risk.status_akhir}"
            ),
        ),
        subjek=Subjek(kategori=kategori, kelompok=risk.kelompok),
        pengukuran=[
            PengukuranItem.model_validate(item.model_dump())
            for item in risk.indikator_terukur
        ],
        data_pendukung=data_pendukung,
        kualitas_foto=kualitas,
        rekomendasi=[risk.rekomendasi] if risk.rekomendasi else [],
        alasan=list(risk.alasan),
        disclaimer=disclaimer,
        meta=Meta(cv_mode=settings.cv_mode, versi=__version__),
    )


@debug_router.post("/debug-landmarks")
async def debug_landmarks(
    current_user: CurrentUser = None,
    image: UploadFile = File(..., description="Foto wajah subjek"),
):
    """Foto yang sama + titik landmark, kotak region, mask rambut, annulus sklera.

    Untuk validasi visual SAJA -- tidak dipakai di alur produksi.
    """
    contents = await image.read()
    image_bgr = _decode_upload(contents, image.content_type)

    try:
        landmarks_px, n_faces = await run_in_threadpool(
            pipeline.detect_landmarks, image_bgr
        )
    except pipeline.FaceNotFoundError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except pipeline.ModelMissingError as e:
        raise HTTPException(status_code=503, detail=MODEL_MISSING_HINT) from e

    skin_lab = await run_in_threadpool(
        pipeline.sample_skin_lab, image_bgr, landmarks_px
    )

    overlay = await run_in_threadpool(
        pipeline.draw_debug_overlay, image_bgr, landmarks_px, skin_lab
    )
    if n_faces > 1:
        overlay = overlay.copy()
        cv2.putText(overlay, f"WARNING: {n_faces} wajah terdeteksi", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    ok, encoded = cv2.imencode(".jpg", overlay)
    if not ok:
        raise HTTPException(status_code=500, detail="Gagal encode gambar debug.")
    return Response(content=encoded.tobytes(), media_type="image/jpeg")
