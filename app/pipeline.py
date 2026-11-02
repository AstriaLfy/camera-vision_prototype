"""
pipeline.py
-----------
Step 0 -> 5 untuk deteksi/skrining malnutrisi dari foto wajah:

  [Foto wajah]
      -> Step 0: IMAGE QUALITY GATE  (resolusi, blur, exposure, pose, jumlah wajah)
      -> Step 1+2: Face Detection + Landmark Extraction (MediaPipe FaceLandmarker)
      -> Step 3: Region Cropping + Segmentation (rambut via segmentasi kulit, sklera via iris)
      -> Step 4: Feature Extraction (CIELAB, dinormalisasi terhadap kulit subjek sendiri)
      -> Step 5: Scoring/Classification (bobot kontinu, fail-closed)

PRINSIP DESAIN (perubahan penting dari versi lama):

1. TIDAK ADA fitur warna/brightness absolut. Semua fitur warna dihitung dalam
   CIELAB lalu dinormalisasi terhadap warna kulit subjek itu sendiri.
   Alasannya: foto anak berkulit gelap dengan pencahayaan berbeda menghasilkan
   angka HSV/grayscale yang sama sekali berbeda, padahal kondisi klinisnya sama.

2. TIDAK ADA rata-rata hue. Hue bersifat melingkar (wrap-around di 0/179),
   jadi np.mean() atas hue menghasilkan hijau/yellow untuk campuran
   merah+kuning. Semua statistik warna pakai median atau mean pada LAB.

3. FAIL-CLOSED. Kalau sebuah region tidak bisa dianalisis (crop kosong,
   kontras terlalu rendah, rambut tidak terlihat), indikator mengembalikan
   is_flagged=None -- BUKAN False. Di versi lama data hilang dilaporkan
   sebagai "normal", dan itu bahaya untuk alat skrining.

4. QUALITY GATE DULU. Foto blur, gelap, atau 3/4 samping ditolak dengan pesan
   yang jelas, bukan diproses diam-diam lalu dilaporkan seolah valid.
"""

from dataclasses import dataclass, field
import math
import threading

import cv2
import numpy as np

from . import config


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class FaceNotFoundError(Exception):
    """Wajah tidak terdeteksi di frame."""


class ImageRejectedError(Exception):
    """Foto tidak memenuhi syarat protokol -> tolak, jangan diproses."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


class ModelMissingError(Exception):
    """File model MediaPipe belum ada di disk."""


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class IndicatorResult:
    name: str
    value: float | None
    is_flagged: bool | None          # None = tidak bisa dievaluasi (bukan "normal")
    detail: str = ""
    normalized: float | None = None  # 0..1, untuk agregasi skor

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "flagged": self.is_flagged,
            "value": None if self.value is None else round(self.value, 4),
            "normalized": None if self.normalized is None else round(self.normalized, 4),
            "detail": self.detail,
        }


@dataclass
class QualityReport:
    ok: bool = True
    score: float = 0.0                 # 0..1
    checks: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "score": round(self.score, 3),
            "checks": {k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in self.checks.items()},
            "warnings": self.warnings,
        }


@dataclass
class CVAnalysisResult:
    indicators: list = field(default_factory=list)
    status_cv: str = "normal"           # normal | sedang | tinggi | tidak_valid
    score: float = 0.0
    quality: QualityReport | None = None
    raw_landmarks_count: int = 0
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 1+2: Face Detection + Landmark Extraction
# ---------------------------------------------------------------------------

_landmarker = None
_landmarker_lock = threading.Lock()


def _get_landmarker():
    """Lazy-load FaceLandmarker, thread-safe.

    Dipisah dari import supaya modul ini bisa di-import (dan Step 3-5 diuji)
    tanpa model ter-download.
    """
    global _landmarker
    if _landmarker is not None:
        return _landmarker

    with _landmarker_lock:
        if _landmarker is not None:      # double-checked locking
            return _landmarker

        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = config.MODEL_PATH
        if not model_path.exists():
            raise ModelMissingError(
                "Model FaceLandmarker belum ada. Jalankan `python download_model.py` "
                "sekali, lalu ulangi request."
            )

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            # Minta lebih dari 1 supaya bisa mendeteksi & menolak foto yang
            # berisi >1 wajah, alih-alih diam-diam memilih satu.
            num_faces=5,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        _landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        return _landmarker


def detect_landmarks(image_bgr: np.ndarray) -> tuple:
    """Return (landmarks_px, n_faces).

    landmarks_px: ndarray (N, 3) = x, y dalam pixel + z relatif.
    n_faces     : jumlah wajah yang terdeteksi (untuk quality gate).
    """
    import mediapipe as mp

    landmarker = _get_landmarker()
    h, w = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise FaceNotFoundError(
            "Wajah tidak terdeteksi. Pastikan wajah terlihat jelas, menghadap kamera, "
            "dan pencahayaan cukup."
        )

    # Kalau >1 wajah, ambil yang paling besar (paling dekat dengan kamera = paling
    # mungkin subjek), lalu biarkan quality gate menolak dengan pesan jelas.
    faces = []
    for lm_list in result.face_landmarks:
        pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in lm_list], dtype=np.float64)
        xs, ys = pts[:, 0], pts[:, 1]
        area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
        faces.append((area, pts))
    faces.sort(key=lambda t: t[0], reverse=True)
    _, landmarks_px = faces[0]

    return landmarks_px, len(faces)


# ---------------------------------------------------------------------------
# Step 0: Image Quality Gate
# ---------------------------------------------------------------------------

def estimate_pose(landmarks_px: np.ndarray) -> dict:
    """Estimasi yaw, roll, dan pitch (SEMUA proxy) dari landmark.

    yaw_proxy: jarak centroid iris dari titik tengah kedua tepi wajah,
               dibagi setengah lebar wajah. 0 = frontal, ~1 = profile.
    roll_deg: sudut garis penghubung kedua iris terhadap horizontal.
    pitch_ratio: (garis rambut -> garis iris) : (garis iris -> dagu).
                 Wajah frontal ~0.40-0.43; bergeser saat kepala menengadah
                 atau mendongak.

    Catatan: semua ini PROXY, bukan angle Euler, dan belum terkalibrasi.
    Cukup untuk tolak/terima foto, tidak untuk reporting sudut presisi.
    """
    edge_l = landmarks_px[config.FACE_EDGE_CAM_LEFT][:2]
    edge_r = landmarks_px[config.FACE_EDGE_CAM_RIGHT][:2]
    iris_l = landmarks_px[config.IRIS_CAM_LEFT_CENTER][:2]
    iris_r = landmarks_px[config.IRIS_CAM_RIGHT_CENTER][:2]

    face_mid_x = (edge_l[0] + edge_r[0]) / 2.0
    half_w = abs(edge_r[0] - edge_l[0]) / 2.0
    iris_mid_x = (iris_l[0] + iris_r[0]) / 2.0

    yaw_proxy = (iris_mid_x - face_mid_x) / half_w if half_w > 0 else 0.0

    dx = iris_r[0] - iris_l[0]
    dy = iris_r[1] - iris_l[1]
    roll_deg = math.degrees(math.atan2(dy, dx)) if (dx or dy) else 0.0
    # normalisasi ke [-90, 90] supaya 170 == -10 (miring ke arah lain)
    if roll_deg > 90:
        roll_deg -= 180
    elif roll_deg < -90:
        roll_deg += 180

    face_top_y = landmarks_px[config.FOREHEAD_TOP][1]
    chin_y = landmarks_px[config.CHIN][1]
    face_h = abs(chin_y - face_top_y)

    iris_mid_y = (iris_l[1] + iris_r[1]) / 2.0
    upper = abs(iris_mid_y - face_top_y)
    lower = abs(chin_y - iris_mid_y)
    pitch_ratio = (upper / lower) if lower > 1e-6 else 1.0

    return {
        "yaw_proxy": yaw_proxy,
        "roll_deg": roll_deg,
        "pitch_ratio": pitch_ratio,
        "face_height_px": face_h,
        "face_width_px": half_w * 2.0,
    }


def measure_sharpness(image_bgr: np.ndarray) -> float:
    """Variance of Laplacian -- proxy blur. Budget: dipakai hanya untuk gate."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_quality(image_bgr: np.ndarray, landmarks_px: np.ndarray, n_faces: int,
                   skin_lab: np.ndarray | None) -> QualityReport:
    """Gate protokol foto. Mengembalikan QualityReport; ok=False = jangan diproses."""
    q = config.QUALITY
    report = QualityReport()
    checks = {}

    if n_faces > q["max_faces"]:
        report.ok = False
        report.checks["n_faces"] = n_faces
        raise ImageRejectedError(
            f"{n_faces} wajah terdeteksi dalam satu foto.",
            "Alat ini menilai SATU subjek. Crop foto agar hanya berisi satu wajah, "
            "lalu ambil ulang.",
        )
    checks["n_faces"] = n_faces

    pose = estimate_pose(landmarks_px)
    checks["face_height_px"] = pose["face_height_px"]

    if pose["face_height_px"] < q["min_face_height_px"]:
        report.ok = False
        report.checks.update(checks)
        raise ImageRejectedError(
            "Wajah terlalu kecil di frame.",
            f"Tinggi wajah {pose['face_height_px']:.0f}px, minimal {q['min_face_height_px']}px. "
            "Mendekatkan kamera atau naikkan resolusi foto.",
        )

    if abs(pose["yaw_proxy"]) > q["max_yaw_proxy"]:
        report.ok = False
        report.checks.update(checks)
        raise ImageRejectedError(
            "Kepala tidak menghadap kamera lurus.",
            f"yaw_proxy={pose['yaw_proxy']:.2f} (maks {q['max_yaw_proxy']}). "
            "Minta subjek menghadap lurus ke kamera.",
        )

    if abs(pose["roll_deg"]) > q["max_roll_deg"]:
        report.ok = False
        report.checks.update(checks)
        raise ImageRejectedError(
            "Kepala miring.",
            f"roll={pose['roll_deg']:.1f}deg (maks {q['max_roll_deg']:.1f}deg). "
            "Minta subjek menegakkan kepala.",
        )

    checks["pitch_ratio"] = pose["pitch_ratio"]
    if not (q["pitch_ratio_min"] <= pose["pitch_ratio"] <= q["pitch_ratio_max"]):
        report.ok = False
        report.checks.update(checks)
        raise ImageRejectedError(
            "Kepala menengadah atau mendongak.",
            f"pitch_ratio={pose['pitch_ratio']:.2f} "
            f"(harus [{q['pitch_ratio_min']:.2f}, {q['pitch_ratio_max']:.2f}]). "
            "Minta subjek menyetel kepala setinggi mata kamera.",
        )

    sharpness = measure_sharpness(image_bgr)
    checks["sharpness"] = sharpness
    if sharpness < q["min_laplacian_variance"]:
        report.ok = False
        report.checks.update(checks)
        raise ImageRejectedError(
            "Foto terlalu buram.",
            f"sharpness={sharpness:.0f} (min {q['min_laplacian_variance']:.0f}). "
            "Gunakan cahaya lebih baik atau kamera lebih stabil.",
        )

    if skin_lab is not None:
        skin_l = float(skin_lab[0])
        checks["skin_l"] = skin_l
        if not (q["skin_l_min"] <= skin_l <= q["skin_l_max"]):
            report.ok = False
            report.checks.update(checks)
            raise ImageRejectedError(
                "Pencahayaan tidak sesuai protokol.",
                f"L* kulit={skin_l:.0f}, harus di [{q['skin_l_min']:.0f}, {q['skin_l_max']:.0f}]. "
                "Pindah ke tempat dengan cahaya merata (hindari cahaya matahari langsung).",
            )

    report.checks.update(checks)
    report.ok = True
    # skor kualitas soft, hanya untuk laporan -- tidak dipakai gating
    parts = [
        np.clip(sharpness / (q["min_laplacian_variance"] * 4), 0, 1),
        1.0 - np.clip(abs(pose["yaw_proxy"]) / q["max_yaw_proxy"], 0, 1),
        1.0 - np.clip(abs(pose["roll_deg"]) / q["max_roll_deg"], 0, 1),
    ]
    report.score = float(np.mean(parts))
    return report


# ---------------------------------------------------------------------------
# Sampling warna kulit (landmark -> median LAB)
# ---------------------------------------------------------------------------

def sample_skin_lab(image_bgr: np.ndarray, landmarks_px: np.ndarray) -> np.ndarray | None:
    """Median CIELAB dari pixel kulit di sekitar titik-titik pipi/dahi.

    Mengembalikan ndarray [L, a, b] atau None kalau tidak cukup pixel valid.
    Median dipakai (bukan mean) supaya pixelkompresi/artefak tidak menggeser
    nilai tengah.
    """
    h, w = image_bgr.shape[:2]
    face_w = abs(landmarks_px[config.FACE_EDGE_CAM_RIGHT][0]
                 - landmarks_px[config.FACE_EDGE_CAM_LEFT][0])
    radius = max(2, int(face_w * config.SKIN_SAMPLE_RADIUS_RATIO))

    lab = opencv_lab_to_cielab(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB))
    samples = []

    for idx in config.SKIN_SAMPLE_POINTS:
        if idx >= len(landmarks_px):
            continue
        cx, cy = int(round(landmarks_px[idx][0])), int(round(landmarks_px[idx][1]))
        if not (radius <= cx < w - radius and radius <= cy < h - radius):
            continue
        patch = lab[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1]
        if patch.size:
            samples.append(patch.reshape(-1, 3))

    if not samples:
        return None
    stacked = np.concatenate(samples, axis=0)
    if stacked.shape[0] < 50:
        return None
    return np.median(stacked, axis=0)   # L*, a*, b*


def opencv_lab_to_cielab(lab8: np.ndarray) -> np.ndarray:
    """LAB byte OpenCV -> CIELAB asli (L* 0..100, a*/b* sekitar 0).

    OpenCV menyimpan hasil COLOR_BGR2LAB sebagai uint8 dengan L diskalakan ke
    0..255 dan a*/b* digeser +128. Nilai itu TIDAK sama dengan CIELAB.

    dulu dipakai apa adanya, dan itu merusak dua hal:
      - quality gate exposure membandingkan L=0..255 ke ambang 38..94 sehingga
        hampir semua foto normal ditolak;
      - rasio a* sklera terhadap a* kulit (dasar indikator anemia) jadi ~1.0
        tidak peduli apa pun warnanya, sehingga ambang 0.50 tidak pernah tercapai
        dan indikator sklera praktis mati.

    Semua konversi di modul ini harus lewat fungsi ini.
    """
    out = np.asarray(lab8).astype(np.float64).copy()
    out[..., 0] *= 100.0 / 255.0
    out[..., 1] -= 128.0
    out[..., 2] -= 128.0
    return out


def lab_chroma(lab: np.ndarray) -> float:
    """Chroma = sqrt(a*^2 + b*^2) dari CIELAB (a*/b* sudah digeser ke 0)."""
    return float(math.hypot(lab[1], lab[2]))


def lab_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance per-pixel antara dua array LAB (N,3)."""
    d = a.astype(np.float64) - b.astype(np.float64).reshape(1, 3)
    return np.sqrt((d ** 2).sum(axis=1))


# ---------------------------------------------------------------------------
# Step 3: Region cropping + segmentation
# ---------------------------------------------------------------------------

def build_hair_roi_box(image_shape, landmarks_px: np.ndarray) -> tuple:
    """Bounding box SEARCH untuk rambut (persegi panjang, hanya untuk 없다고).

    Untuk mask yang benar-benar dipakai, pakai build_hair_roi_mask() -- Kotak
    persegi panjang di atas wajah yang BULAT selalu menyorotKAN sudut-sudut
    ke latar belakang (dinding/langit). Itu sebab utama rata-rata warna
    rambut di versi lama tidak bermakna: sudut-sudut box berisi dinding.
    """
    box = _hair_search_extent(image_shape, landmarks_px)
    return box


def _hair_search_extent(image_shape, landmarks_px: np.ndarray) -> tuple:
    """(x_min, y_min, x_max, y_max) area search di atas dahi."""
    h, w = image_shape[:2]
    face_top = landmarks_px[config.FOREHEAD_TOP]
    chin = landmarks_px[config.CHIN]
    face_h = abs(chin[1] - face_top[1])
    edge_l = landmarks_px[config.FACE_EDGE_CAM_LEFT]
    edge_r = landmarks_px[config.FACE_EDGE_CAM_RIGHT]
    mid_x = (edge_l[0] + edge_r[0]) / 2.0

    # pita tengah: 0.42 dari lebar wajah tiap sisi dari garis tengah
    face_w = abs(edge_r[0] - edge_l[0])
    band_half = face_w * 0.42
    x_min = int(max(0, mid_x - band_half))
    x_max = int(min(w, mid_x + band_half))

    y_max = int(face_top[1])
    y_min = int(max(0, face_top[1] - face_h * config.MARGIN_HAIR_REGION_RATIO))
    # buang 12% baris teratas: paling mungkin murni background di atas kepala
    trim = int((y_max - y_min) * 0.12)
    y_min += trim

    if y_max <= y_min or x_max <= x_min:
        return (0, 0, 0, 0)
    return (x_min, y_min, x_max, y_max)


def build_hair_roi_mask(image_shape, landmarks_px: np.ndarray):
    """Mask BENTUK WAJAH untuk area rambut (bukan persegi panjang).

    Lebar yang diizinkan di tiap baris mengikuti lebar wajah di ketinggian itu,
    lalu menyempit ke arah atas mengikuti oval kepala. Hasilnya sebuah trapesium
    membulat, jadi tidak ada sudut kotak yang jatuh ke dinding/latar belakang.

    Kembalikan (mask bool sesuai ukuran box, box) atau (None, box) kalau tidak
    valid.
    """
    h, w = image_shape[:2]
    box = _hair_search_extent(image_shape, landmarks_px)
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None, box

    top_y = landmarks_px[config.FOREHEAD_TOP][1]
    edge_l = landmarks_px[config.FACE_EDGE_CAM_LEFT][0]
    edge_r = landmarks_px[config.FACE_EDGE_CAM_RIGHT][0]
    mid_x = (edge_l + edge_r) / 2.0
    face_w = abs(edge_r - edge_l)

    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    box_h = y1 - y0
    for i in range(box_h):
        yy = y0 + i
        # t = 0 tepat di garis rambut, t = 1 di baris paling atas kotak
        t = (top_y - yy) / max(top_y - y0, 1e-6)
        t = float(np.clip(t, 0.0, 1.0))
        # Taper BENTUK KAP: lebar penuh di garis rambut, menyempit ke arah atas
        # mengikuti oval kepala. Kalau dibuat lebar, sudut kotak akan
        # keluar dari kepala dan masuk ke dinding.
        frac = 0.72 * math.sqrt(max(0.05, 1.0 - 1.35 * t * t))
        half = face_w * frac / 2.0
        col_lo = int(max(0, math.floor(mid_x - half - x0)))
        col_hi = int(min(x1 - x0, math.ceil(mid_x + half - x0)))
        if col_hi > col_lo:
            mask[i, col_lo:col_hi] = True

    if not mask.any():
        return None, box
    return mask, box


def segment_hair_mask(image_bgr: np.ndarray, landmarks_px: np.ndarray,
                      skin_lab: np.ndarray) -> tuple:
    """Pisahkan rambut dari kulit di dalam area search di atas dahi.

    Returns (mask uint8, fg_ratio). mask=1 berarti dianggap rambut.

    Dua lapis perlindungan terhadap latar belakang:

    1. GEOMETRI: hanya pixel di dalam mask build_hair_roi_mask() yang boleh
       dipakai. Mask itu mengikuti oval wajah, jadi sudut kotak yang biasa
       berisi dinding sudah tercoret SEBELUM segmentasi berjalan. Ini jauh
       lebih andal daripada menebak-nebak lewat ukuran blob.
    2. WARNA: di dalam area yang diizinkan, pixel dibedakan dari kulit lewat
       jarak CIELAB, lalu komponen terhubung terbesar diambil.

    Hairline (garis rambut) membagi dua: di bawahnya kulit, di atasnya rambut.
    Karena itu mask hanya dipakai sebagai pagar, bukan sebagai penentu rambut.

    Guard terakhir: foreground terlalu sedikit berarti rambut tidak bisa
    dianalisis (mis. kerontok) -> indikator dinonaktifkan (fail-closed).
    """
    roi_mask, box = build_hair_roi_mask(image_bgr.shape, landmarks_px)
    if roi_mask is None:
        return None, 0.0
    x0, y0, x1, y1 = box

    patch = image_bgr[y0:y1, x0:x1]
    lab = opencv_lab_to_cielab(
        cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)).reshape(-1, 3)

    tol = config.HAIR["skin_distance_tolerance"]
    dist = lab_distance(lab, skin_lab).reshape(roi_mask.shape)
    not_skin = ((dist > tol) & roi_mask).astype(np.uint8)

    allowed = int(roi_mask.sum())
    if allowed == 0:
        return None, 0.0
    # fg_ratio relatif terhadap area yang diizinkan, bukan terhadap kotak --
    # kalau relatif terhadap kotak, lekukan oval ikutpires sebagai "tidak ada".
    fg_ratio = float(not_skin.sum()) / float(allowed)

    mask_img = not_skin
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_img, connectivity=8)
    if n_labels <= 1:
        return None, fg_ratio

    # component 0 = yang dianggap kulit
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return None, fg_ratio
    largest = int(np.argmax(areas)) + 1
    area = int(stats[largest, cv2.CC_STAT_AREA])

    # terlalu sedikit non-kulit untuk dianalisis (rambut tipis / kerontok)
    if fg_ratio < config.HAIR["min_foreground_ratio"] or area < 0.01 * allowed:
        return None, fg_ratio

    return (labels == largest).astype(np.uint8), fg_ratio


def build_sclera_annulus(landmarks_px: np.ndarray, iris_center_idx: int,
                         iris_ring: list) -> tuple:
    """Geometri annulus di sekitar iris (dalam pixel).

    Annulus 1.05r .. 2.2r mengelilingi iris. Ini memakai landmark iris
    (468-477) yang di versi lama TIDAK PERNAH dipakai, dan jauh lebih
    terkandal daripada bounding box yang menyatukan kedua mata.

    Returns (center, r_inner, r_outer) atau None.
    """
    if iris_center_idx >= len(landmarks_px):
        return None
    center = landmarks_px[iris_center_idx][:2]

    ring_pts = [landmarks_px[i][:2] for i in iris_ring if i < len(landmarks_px)]
    if len(ring_pts) < 3:
        return None

    radii = [float(np.linalg.norm(p - center)) for p in ring_pts]
    iris_r = float(np.median(radii))
    if iris_r < 2.0:
        return None
    return center, iris_r * 1.05, iris_r * 2.2


def sclera_roi_from_iris(image_shape, landmarks_px: np.ndarray,
                         iris_center_idx: int, iris_ring: list):
    """Mask boolean untuk pixel sklera di sekitar iris."""
    geo = build_sclera_annulus(landmarks_px, iris_center_idx, iris_ring)
    if geo is None:
        return None
    center, r_in, r_out = geo

    h, w = image_shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    r_out_i = int(np.ceil(r_out))
    x0, x1 = max(0, cx - r_out_i), min(w, cx + r_out_i + 1)
    y0, y1 = max(0, cy - r_out_i), min(h, cy + r_out_i + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.hypot(xx - center[0], yy - center[1])

    return (
        (slice(y0, y1), slice(x0, x1)),
        ((dist >= r_in) & (dist <= r_out)),
    )


# ---------------------------------------------------------------------------
# Step 4: Feature extraction (CIELAB, dinormalisasi ke kulit subjek)
# ---------------------------------------------------------------------------

def extract_hair_features(image_bgr: np.ndarray, landmarks_px: np.ndarray,
                          skin_lab: np.ndarray) -> dict:
    """Fitur rambut: L* dan chroma median, relatif terhadap kulit subjek."""
    mask, fg_ratio = segment_hair_mask(image_bgr, landmarks_px, skin_lab)
    if mask is None:
        return {"available": False, "reason": "tidak ada massa non-kulit yang kompak di region rambut",
                "foreground_ratio": fg_ratio}

    if fg_ratio < config.HAIR["min_foreground_ratio"]:
        return {"available": False, "reason": "region rambut terlalu kecil / rambut tidak terlihat",
                "foreground_ratio": fg_ratio}

    box = build_hair_roi_box(image_bgr.shape, landmarks_px)
    x0, y0, x1, y1 = box
    lab = opencv_lab_to_cielab(
        cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2LAB))

    sel = lab[mask.astype(bool)]
    if sel.shape[0] < 30:
        return {"available": False, "reason": "terlalu sedikit pixel rambut",
                "foreground_ratio": fg_ratio}

    l_med = float(np.median(sel[:, 0]))
    a_med = float(np.median(sel[:, 1]))
    b_med = float(np.median(sel[:, 2]))
    hair_chroma = float(math.hypot(a_med, b_med))

    skin_l = float(skin_lab[0])
    skin_chroma = lab_chroma(skin_lab)

    return {
        "available": True,
        "hair_l": l_med,
        "hair_chroma": hair_chroma,
        "skin_l": skin_l,
        "skin_chroma": skin_chroma,
        "l_delta": skin_l - l_med,                       # kecil = rambut pucat
        "chroma_ratio": hair_chroma / skin_chroma if skin_chroma > 1e-6 else None,
        "foreground_ratio": fg_ratio,
        "n_pixels": int(sel.shape[0]),
    }


def extract_sclera_features(image_bgr: np.ndarray, landmarks_px: np.ndarray,
                            skin_lab: np.ndarray) -> dict:
    """Fitur kepucatan dari a* sklera, dinormalisasi terhadap a* kulit.

    a* di CIELAB mengukur "kemerahan" -- proposional kadar hemoglobin di
    jaringan. a* kulit > 0 (darah), a* sklera sehat relatif lebih rendah.
    Rasio a*_sklera / a*_kulit naik mendekati 1 saat pucat.

    Kedua mata dianalisis terpisah lalu digabung, supaya satu mata tertutup/
    berkacamata tidak menghapus informasi mata yang lain.
    """
    t = config.SCLERA
    skin_a_raw = float(skin_lab[1])
    if skin_a_raw < t["min_skin_a_star"]:
        return {"available": False, "reason": "warna kulit tidak punya a* positif (data tidak valid)"}

    # Penyebut rasio dibatasi. Tanpa floor ini, subyek berkulit netral (a* ~ 2)
    # membikin rasio meledak: sklera a*=-1 -> rasio -0.5 -> dihitung kepucatan
    # berat, padahal itu artefak pembagian, bukan kondisi klinis.
    skin_a = max(skin_a_raw, t["skin_a_star_floor"])

    per_eye = []
    total_px = 0
    # PENTING: teruskan RING IRIS (4 titik), bukan ring mata (16 titik).
    # Ring mata jauh lebih lebar -> radius iris terbaca terlalu besar ->
    # annulus sklera menapu bridge hidung & kulit -> a* yang diukur bukan sklera.
    for center_idx, ring in (
        (config.IRIS_CAM_LEFT_CENTER, config.IRIS_CAM_LEFT_RING),
        (config.IRIS_CAM_RIGHT_CENTER, config.IRIS_CAM_RIGHT_RING),
    ):
        roi = sclera_roi_from_iris(image_bgr.shape, landmarks_px, center_idx, ring)
        if roi is None:
            continue
        (ys, xs), mask = roi
        lab = opencv_lab_to_cielab(cv2.cvtColor(image_bgr[ys, xs], cv2.COLOR_BGR2LAB))
        sel = lab[mask]
        if sel.shape[0] < 20:
            continue
        # buang 8% terrendah L* (pupil/rilis eyelash yang masuk annulus)
        l_floor = np.percentile(sel[:, 0], 8)
        sel = sel[sel[:, 0] >= l_floor]
        if sel.shape[0] < 20:
            continue
        # buang 8% tertinggi a* (caruncle / pelipit kemerahan) -- trim kecil
        # dan berbatas, BUKAN memotong 60% pixel seperti versi lama
        a_ceil = np.percentile(sel[:, 1], 92)
        sel = sel[sel[:, 1] <= a_ceil]
        if sel.shape[0] < 20:
            continue
        per_eye.append(float(np.mean(sel[:, 1])))
        total_px += int(sel.shape[0])

    if not per_eye:
        return {"available": False, "reason": "tidak ada region sklera yang cukup besar"}
    if total_px < t["min_pixels"]:
        return {"available": False, "reason": f"pixel sklera kurang dari minimum ({total_px})",
                "n_pixels": total_px}

    sclera_a = float(np.mean(per_eye))
    return {
        "available": True,
        "sclera_a_star": sclera_a,
        "skin_a_star": skin_a,
        "skin_a_star_raw": skin_a_raw,
        "a_star_ratio": sclera_a / skin_a,
        "n_eyes": len(per_eye),
        "n_pixels": total_px,
    }


def extract_geometry_features(landmarks_px: np.ndarray) -> dict:
    """Rasio geometri wajah (lihat config.GEOMETRY untuk literatur)."""
    cheek_l = landmarks_px[config.CHEEK_CAM_LEFT][:2]
    cheek_r = landmarks_px[config.CHEEK_CAM_RIGHT][:2]
    jaw_l = landmarks_px[config.JAW_ANGLE_CAM_LEFT][:2]
    jaw_r = landmarks_px[config.JAW_ANGLE_CAM_RIGHT][:2]
    top = landmarks_px[config.FOREHEAD_TOP][:2]
    chin = landmarks_px[config.CHIN][:2]
    lip = landmarks_px[config.LOWER_LIP_BOTTOM][:2]

    cheek_w = float(np.linalg.norm(cheek_r - cheek_l))
    jaw_w = float(np.linalg.norm(jaw_r - jaw_l))
    total_h = abs(chin[1] - top[1])
    lower_h = abs(chin[1] - lip[1])

    out = {"available": total_h > 1e-6, "cheek_width_px": cheek_w, "jaw_width_px": jaw_w}
    if jaw_w > 1e-6:
        out["cheek_to_jaw_ratio"] = cheek_w / jaw_w
    if total_h > 1e-6:
        out["lower_face_height_ratio"] = lower_h / total_h
    return out


# ---------------------------------------------------------------------------
# Step 5: Scoring
# ---------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def score_hair(features: dict) -> IndicatorResult:
    """Indikator hipopigmentasi rambut.

    ARAH KLINIS YANG BENAR: kwashiorkor mengurangi melanin rambut sehingga
    rambut menjadi "reddish yellow to white" (Medscape, Protein-Energy
    Malnutrition) / "gray-blond" (dokumentasi foto CDC). Sinyal CV-nya bukan
    "hue kemerahan", melainkan L* rambut mendekati L* kulit DAN chroma rambut
    turun tajam terhadap kulit.

    Versi lama memakai `hue <= 20 or hue >= 160` yang terbukti salah arah:
    ia MISS pada warna rambut yang justru dideskripsikan di literatur
    ("reddish yellow") dan FALSE-POSITIVE pada rambut coklat Asia normal.
    """
    if not features.get("available"):
        return IndicatorResult("rambut", None, None,
                               detail=f"tidak dapat dievaluasi: {features.get('reason', 'tidak diketahui')}")

    l_delta = features["l_delta"]
    chroma_ratio = features.get("chroma_ratio")
    h = config.HAIR

    # arah 1: L* rambut mendekati L* kulit (rambut pucat).
    # severity 0 di l_delta_max (atau di atasnya), severity 1 di <= 0.
    l_component = _clamp01((h["l_delta_max"] - l_delta)
                           / max(h["l_delta_max"] - h["l_delta_flag"], 1e-6))

    # arah 2: chroma rambut anjlok dibanding kulit (hipopigmentasi).
    # severity 0 di chroma_ratio_max, severity 1 di <= chroma_ratio_flag.
    if chroma_ratio is None:
        c_component = None
    else:
        c_component = _clamp01((h["chroma_ratio_max"] - chroma_ratio)
                               / max(h["chroma_ratio_max"] - h["chroma_ratio_flag"], 1e-6))

    # FLAG hanya kalau KEDUA arah melewati titik flag-nya (min). Satu arah
    # saja tidak cukup: rambut hitam pekat memang low-chroma secara normal,
    # dan rambut terang memang L*-nya tinggi -- kalau salah satu arah saja
    # sudah memicu, hampir semua rambut akan ter-flag.
    #
    # `normalized` memakai RATA-RATA (bukan min) supaya informasi "mendekati
    # abnormal di salah satu arah" tidak hilang total. Kalau pakai min,
    # subyek dengan (1.0, 0.0) dapat skor 0.0 yang identik dengan subyek
    # sehat -- persis masalah yang ditemukan di score_sclera.
    if c_component is None:
        normalized = l_component
        l_flagged = l_delta <= h["l_delta_flag"]
        flagged = l_flagged
    elif l_component <= 0.0:
        # Rambut JAUH lebih gelap dari kulit (l_delta di atas l_delta_max):
        # tidak ada sinyal hipopigmentasi sama sekali. Tanpa cabang ini,
        # subyek berambut hitam tetap dapat severity > 0 hanya karena chroma
        # rambut hitam memang rendah -- jadi rambut hitam yang normal ikut
        # menyumbang skor.
        normalized = 0.0
        flagged = False
    else:
        normalized = _clamp01((l_component + c_component) / 2.0)
        l_flagged = l_delta <= h["l_delta_flag"]
        c_flagged = chroma_ratio <= h["chroma_ratio_flag"]
        flagged = bool(l_flagged and c_flagged)

    detail = (f"L_delta={l_delta:.1f} (flag bila <= {h['l_delta_flag']:.0f}), "
              f"chroma_ratio={chroma_ratio:.3f} (flag bila <= {h['chroma_ratio_flag']:.2f})"
              if chroma_ratio is not None else f"L_delta={l_delta:.1f}")
    return IndicatorResult("rambut", l_delta, flagged, detail, normalized)


def score_sclera(features: dict) -> IndicatorResult:
    """Indikator kepucatan (proxy anemia) via a* CIELAB."""
    if not features.get("available"):
        return IndicatorResult("kepucatan_mata", None, None,
                               detail=f"tidak dapat dievaluasi: {features.get('reason', 'tidak diketahui')}")

    ratio = features["a_star_ratio"]
    t = config.SCLERA["a_star_ratio_max"]
    flagged = ratio > t

    # Severity dihitung dari baseline "sehat", bukan dari threshold.
    # Kalau severity hanya dihitung di atas threshold, maka rasio tepat di bawah
    # ambang (mis. 0.79 vs 0.80) mendapat normalized=0.0 -- identik dengan
    # subyek yang benar-benar sehat -- sehingga skor agregat kehilangan informasi
    # "mendekati abnormal" dan dua kondisi berbeda bisa mendapat status sama.
    baseline = config.SCLERA["a_star_ratio_normal"]
    normalized = _clamp01((ratio - baseline) / max(t - baseline, 1e-6))

    detail = (f"a*_sklera/a*_kulit={ratio:.3f} (flag > {t}), "
              f"a*_kulit={features.get('skin_a_star'):.1f}, "
              f"mata={features.get('n_eyes')}, px={features.get('n_pixels')}")
    return IndicatorResult("kepucatan_mata", ratio, flagged, detail, normalized)


def score_geometry(features: dict) -> IndicatorResult:
    """Indikator geometri wajah (rasio cheek-to-jaw & lower-face height)."""
    if not features.get("available"):
        return IndicatorResult("geometri_wajah", None, None,
                               detail="landmark tidak lengkap")

    g = config.GEOMETRY
    ck = features.get("cheek_to_jaw_ratio")
    lf = features.get("lower_face_height_ratio")
    if ck is None or lf is None:
        return IndicatorResult("geometri_wajah", None, None, detail="rasio tidak terhitung")

    # pipi relatif terhadap rahang terlalu sempit
    c_component = _clamp01((g["cheek_to_jaw_ratio_min"] - ck)
                           / max(g["cheek_to_jaw_ratio_min"], 1e-6))
    # lower-face height di luar rentang normal (terlalu pendek ATAU terlalu panjang)
    if lf < g["lower_face_height_ratio_min"]:
        l_component = _clamp01((g["lower_face_height_ratio_min"] - lf)
                               / max(g["lower_face_height_ratio_min"], 1e-6))
    elif lf > g["lower_face_height_ratio_max"]:
        l_component = _clamp01((lf - g["lower_face_height_ratio_max"])
                               / max(1.0 - g["lower_face_height_ratio_max"], 1e-6))
    else:
        l_component = 0.0

    severity = max(c_component, l_component)
    normalized = severity
    flagged = severity > 0
    value = float((ck + lf) / 2)
    detail = (f"cheek/jaw={ck:.3f} (min {g['cheek_to_jaw_ratio_min']}), "
              f"lower_face_ratio={lf:.3f} (normal {g['lower_face_height_ratio_min']}-{g['lower_face_height_ratio_max']})")
    return IndicatorResult("geometri_wajah", value, flagged, detail, normalized)


def combine_status(indicators: list) -> tuple:
    """Agregasi berbobot -> (status, score 0..1).

    BERUBAH dari versi lama. Versi lama cuma menghitung "berapa yang flagged",
    yang punya dua masalah:
      1. indikator gagal (None) ikut dihitung seolah normal -- fail-open
      2. satu indikator kuat bisa tertutupi indikator lain yang normal

    Sekarang: skor kontinu berbobot (config.WEIGHTS), indikator yang gagal
    dihitung 0 tapi MELETAKKAN status jadi "tidak_valid" supaya tidak pernah
    dilaporkan sebagai "normal" berdasarkan data yang tidak lengkap.
    """
    evaluated = [i for i in indicators if i.normalized is not None]
    if not evaluated or len(evaluated) < len(indicators):
        return "tidak_valid", 0.0

    total_w = sum(config.WEIGHTS.get(i.name, 1.0) for i in evaluated)
    if total_w <= 0:
        return "tidak_valid", 0.0
    score = sum(i.normalized * config.WEIGHTS.get(i.name, 1.0) for i in evaluated) / total_w

    st = config.STATUS_THRESHOLDS
    if score >= st["tinggi"]:
        status = "tinggi"
    elif score >= st["sedang"]:
        status = "sedang"
    else:
        status = "normal"

    # Konsistensi: kalau ada indikator yang FLAG, status overall TIDAK BOLEH
    # "normal". Tanpa aturan ini, operator bisa melihat satu indikator
    # is_flagged=True sementara status overall tetap "normal" -- dan karena ambang
    # status maupun ambang flag sama-sama belum terkalibrasi, lebih aman
    # menaikkan status minimal ke "sedang" daripada melaporkan "normal"
    # padahal ada temuan positif.
    if status == "normal" and any(i.is_flagged for i in evaluated):
        status = "sedang"

    return status, score


# ---------------------------------------------------------------------------
# Pipeline lengkap
# ---------------------------------------------------------------------------

def draw_debug_overlay(image_bgr: np.ndarray, landmarks_px: np.ndarray,
                       skin_lab: np.ndarray | None = None) -> np.ndarray:
    """Overlay visual untuk validasi landmark & region. BUKAN untuk produksi."""
    overlay = image_bgr.copy()
    h, w = image_bgr.shape[:2]

    def pt(idx):
        return int(round(landmarks_px[idx][0])), int(round(landmarks_px[idx][1]))

    labels = {
        "FOREHEAD_TOP": config.FOREHEAD_TOP,
        "CHIN": config.CHIN,
        "EDGE_CAM_LEFT": config.FACE_EDGE_CAM_LEFT,
        "EDGE_CAM_RIGHT": config.FACE_EDGE_CAM_RIGHT,
        "CHEEK_CAM_LEFT": config.CHEEK_CAM_LEFT,
        "CHEEK_CAM_RIGHT": config.CHEEK_CAM_RIGHT,
        "JAW_CAM_LEFT": config.JAW_ANGLE_CAM_LEFT,
        "JAW_CAM_RIGHT": config.JAW_ANGLE_CAM_RIGHT,
        "NOSE_TIP": config.NOSE_TIP,
        "IRIS_L": config.IRIS_CAM_LEFT_CENTER,
        "IRIS_R": config.IRIS_CAM_RIGHT_CENTER,
    }
    for label, idx in labels.items():
        if idx < len(landmarks_px):
            x, y = pt(idx)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(overlay, (x, y), 4, (0, 0, 255), -1)
                cv2.putText(overlay, label, (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

    # garis pipi & rahang
    cv2.line(overlay, pt(config.CHEEK_CAM_LEFT), pt(config.CHEEK_CAM_RIGHT), (0, 255, 0), 2)
    cv2.line(overlay, pt(config.JAW_ANGLE_CAM_LEFT), pt(config.JAW_ANGLE_CAM_RIGHT), (255, 128, 0), 2)
    cv2.line(overlay, pt(config.FACE_EDGE_CAM_LEFT), pt(config.FACE_EDGE_CAM_RIGHT), (255, 0, 0), 2)
    cv2.line(overlay, pt(config.IRIS_CAM_LEFT_CENTER), pt(config.IRIS_CAM_RIGHT_CENTER), (255, 0, 255), 2)

    # kotak search rambut
    box = build_hair_roi_box(image_bgr.shape, landmarks_px)
    if box[2] > box[0] and box[3] > box[1]:
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
        if skin_lab is not None:
            mask, _ = segment_hair_mask(image_bgr, landmarks_px, skin_lab)
            if mask is not None:
                sub = overlay[box[1]:box[3], box[0]:box[2]].copy()
                tint = np.zeros_like(sub)
                tint[mask.astype(bool)] = (0, 0, 255)
                overlay[box[1]:box[3], box[0]:box[2]] = cv2.addWeighted(sub, 0.6, tint, 0.4, 0)

    # annulus sklera
    for center_idx in (config.IRIS_CAM_LEFT_CENTER, config.IRIS_CAM_RIGHT_CENTER):
        iris_ring = (config.IRIS_CAM_LEFT_RING if center_idx == config.IRIS_CAM_LEFT_CENTER
                     else config.IRIS_CAM_RIGHT_RING)
        geo = build_sclera_annulus(landmarks_px, center_idx, iris_ring)
        if geo:
            c, r_in, r_out = geo
            cx, cy = int(round(c[0])), int(round(c[1]))
            cv2.circle(overlay, (cx, cy), int(round(r_out)), (255, 0, 255), 1)
            cv2.circle(overlay, (cx, cy), int(round(r_in)), (255, 0, 255), 1)

    pose = estimate_pose(landmarks_px)
    cv2.putText(overlay,
                f"yaw={pose['yaw_proxy']:.2f} roll={pose['roll_deg']:.1f}deg face_h={pose['face_height_px']:.0f}px",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def process_face_photo(image_bgr: np.ndarray) -> CVAnalysisResult:
    """Step 0 -> 5."""
    landmarks_px, n_faces = detect_landmarks(image_bgr)          # Step 1+2

    skin_lab = sample_skin_lab(image_bgr, landmarks_px)           # warna kulit acuan
    quality = assess_quality(image_bgr, landmarks_px, n_faces, skin_lab)   # Step 0 (tolak)

    notes = []
    if skin_lab is None:
        notes.append("Sampling warna kulit gagal; fitur warna dinonaktifkan.")

    # Step 4 features
    hair_features = extract_hair_features(image_bgr, landmarks_px, skin_lab) \
        if skin_lab is not None else {"available": False, "reason": "warna kulit tidak tersedia"}
    sclera_features = extract_sclera_features(image_bgr, landmarks_px, skin_lab) \
        if skin_lab is not None else {"available": False, "reason": "warna kulit tidak tersedia"}
    geometry_features = extract_geometry_features(landmarks_px)

    # Step 5 scoring
    indicators = [
        score_hair(hair_features),
        score_sclera(sclera_features),
        score_geometry(geometry_features),
    ]

    status, score = combine_status(indicators)
    if status == "tidak_valid":
        notes.append("Status TIDAK_VALID: ada indikator yang tidak bisa dievaluasi. "
                     "Jangan perlakukan sebagai 'normal'.")

    return CVAnalysisResult(
        indicators=indicators,
        status_cv=status,
        score=score,
        quality=quality,
        raw_landmarks_count=len(landmarks_px),
        notes=notes,
    )