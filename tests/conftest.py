"""
conftest.py
-----------
Fixture bersama: renderer wajah sintetis, path model, import mediapipe opsional.

Test yang butuh model asli akan skip otomatis kalau model/dep tidak ada --
bukan GAGAL. Ini penting supaya logic murni (gate, segmentasi, scoring) tetap
bisa diuji meskipun model belum di-download.

Kenapa wajah di-render, bukan memakai foto?
  - Foto wajah pihak ketiga di repo ini berarti Provenance lisensi yang tidak
    jelas (salinan lena.jpg OpenCV sudah dihapus karena itu).
  - Foto 3/4 view yang ada diinternet tidak bisa dipakai sebagai end-to-end
    POSITIF karena quality gate memang harus menolaknya.
  - Synthetic landmark saja tidak membuktikan apa-apa soal model: model bisa
    mengembalikan mesh yang sama sekali tidak masuk akal dan semua test tetap hijau.

Kalau renderer ini suatu saat tidak terdeteksi MediaPipe (mis. setelah upgrade
model), test akan skip dengan pesan jelas -- BUKAN diam-diam lolos.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config, pipeline  # noqa: E402


MODEL_AVAILABLE = config.MODEL_PATH.exists()
MEDIAPIPE_AVAILABLE = True
try:
    import mediapipe  # noqa: F401
except Exception:
    MEDIAPIPE_AVAILABLE = False

requires_model = pytest.mark.skipif(
    not (MODEL_AVAILABLE and MEDIAPIPE_AVAILABLE),
    reason=(
        "Model FaceLandmarker / mediapipe tidak tersedia. "
        "Jalankan: pip install -r requirements.txt && python download_model.py"
    ),
)


# ---------------------------------------------------------------------------
# Renderer wajah frontal prosedural
# ---------------------------------------------------------------------------

SKIN = (150, 168, 196)
HAIR = (55, 45, 40)
SCLERA = (238, 228, 232)
IRIS = (70, 60, 50)
MOUTH = (110, 105, 175)


def render_frontal_face(size: int = 640) -> np.ndarray:
    """Render wajah frontal bertekstur cukup agar dideteksi FaceLandmarker."""
    import cv2

    img = np.zeros((size, size, 3), np.uint8)
    img[:, :] = (72, 78, 84)  # dinding

    cx, cy = size // 2, int(size * 0.52)
    fw, fh = int(size * 0.26), int(size * 0.36)

    # leher + kepala
    cv2.rectangle(img, (cx - fw // 3, cy + fh // 2), (cx + fw // 3, size), SKIN, -1)
    cv2.ellipse(img, (cx, cy), (fw, fh), 0, 0, 360, SKIN, -1)

    # rambut menutupi bagian atas kepala
    cv2.ellipse(img, (cx, cy - int(fh * 0.42)), (int(fw * 1.02), int(fh * 0.52)),
                0, 180, 360, HAIR, -1)
    cv2.ellipse(img, (cx, cy - int(fh * 0.40)), (int(fw * 0.99), int(fh * 0.50)),
                0, 200, 340, HAIR, -1)

    eye_dx = int(fw * 0.40)
    eye_y = cy - int(fh * 0.10)
    eye_w, eye_h = int(fw * 0.30), int(fh * 0.16)

    # alis
    for sx in (-1, 1):
        ex = cx + sx * eye_dx
        cv2.ellipse(img, (ex, eye_y - int(eye_h * 1.5)),
                    (int(eye_w * 0.6), int(eye_h * 0.45)), 0, 200, 340, HAIR, -1)

    # mata: sklera + iris + pupil + kelopak
    for sx in (-1, 1):
        ex = cx + sx * eye_dx
        cv2.ellipse(img, (ex, eye_y), (eye_w, eye_h), 0, 0, 360, SCLERA, -1)
        cv2.circle(img, (ex, eye_y), int(eye_h * 0.85), IRIS, -1)
        cv2.circle(img, (ex, eye_y), int(eye_h * 0.40), (30, 28, 26), -1)
        cv2.circle(img, (ex, eye_y - int(eye_h * 0.10)), int(eye_h * 0.18),
                   (245, 245, 245), -1)
        cv2.ellipse(img, (ex, eye_y), (eye_w, eye_h), 0, 190, 350, (60, 50, 48), 3)
        cv2.ellipse(img, (ex, eye_y), (eye_w, eye_h), 0, 15, 165, (60, 50, 48), 2)

    # hidung
    nose_top = cy + int(fh * 0.05)
    nose_bot = cy + int(fh * 0.30)
    cv2.line(img, (cx, nose_top), (cx - int(fw * 0.10), nose_bot - 6),
             (128, 148, 178), 3)
    cv2.line(img, (cx, nose_top), (cx + int(fw * 0.10), nose_bot - 6),
             (128, 148, 178), 3)
    cv2.ellipse(img, (cx, nose_bot), (int(fw * 0.17), int(fh * 0.07)), 0, 0, 360,
                SKIN, -1)
    for sx in (-1, 1):
        cv2.ellipse(img, (cx + sx * int(fw * 0.08), nose_bot + 2),
                    (int(fw * 0.045), int(fh * 0.025)), 0, 0, 360, (95, 105, 120), -1)

    # mulut
    mouth_y = cy + int(fh * 0.52)
    cv2.ellipse(img, (cx, mouth_y), (int(fw * 0.34), int(fh * 0.13)), 0, 0, 360,
                MOUTH, -1)
    cv2.line(img, (cx - int(fw * 0.34), mouth_y), (cx + int(fw * 0.34), mouth_y),
             (80, 70, 130), 2)

    # sedikit tekstur kulit supaya tidak terlalu planar
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 2.0, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def face_image_bgr() -> np.ndarray:
    """Gambar wajah frontal hasil render (deteksi MediaPipe sudah diverifikasi)."""
    return render_frontal_face()


@pytest.fixture
def synthetic_landmarks():
    """Landmark sintetis yang bisa diuji TANPA MediaPipe.

    Mengembalikan ndarray (478, 3) berisi z=0.446539 (nilai z khas FaceMesh)
    supaya bentuk array sama persis dengan keluaran detect_landmarks.
    """
    return make_landmarks()


def make_landmarks(
    image_w: int = 640,
    image_h: int = 640,
    face_h: float = 300.0,
    face_w: float = 220.0,
    yaw_shift: float = 0.0,
    roll_deg: float = 0.0,
    lip_y_ratio: float = 0.62,
) -> np.ndarray:
    """Landmark wajah menghadap depan (kecuali optional yaw/roll).

    yaw_shift menggeser kedua iris ke kanan (dalam px) untuk menyimulasikan
    kepala tidak menghadap lurus.
    """
    lm = np.zeros((478, 3), dtype=np.float64)
    lm[:, 2] = 0.446539

    cx = image_w / 2.0
    cy = image_h / 2.0

    # apply roll: rotasi semua titik yang kita taruh di sekitar pusat
    rad = np.radians(roll_deg)
    cos_r, sin_r = np.cos(rad), np.sin(rad)

    def place(idx, x, y):
        dx, dy = x - cx, y - cy
        lm[idx, 0] = cx + dx * cos_r - dy * sin_r
        lm[idx, 1] = cy + dx * sin_r + dy * cos_r

    # oval wajah
    place(config.FOREHEAD_TOP, cx, cy - face_h / 2)
    place(config.CHIN, cx, cy + face_h / 2)
    place(config.FACE_EDGE_CAM_LEFT, cx - face_w / 2, cy)
    place(config.FACE_EDGE_CAM_RIGHT, cx + face_w / 2, cy)

    # pipi (simetris kiri-kanan gambar)
    place(config.CHEEK_CAM_LEFT, cx - face_w * 0.45, cy)
    place(config.CHEEK_CAM_RIGHT, cx + face_w * 0.45, cy)
    # rahang
    place(config.JAW_ANGLE_CAM_LEFT, cx - face_w * 0.40, cy + face_h * 0.30)
    place(config.JAW_ANGLE_CAM_RIGHT, cx + face_w * 0.40, cy + face_h * 0.30)
    # ujung dagu untuk lebar dagu
    place(config.CHIN_WIDTH_CAM_LEFT, cx - face_w * 0.18, cy + face_h * 0.46)
    place(config.CHIN_WIDTH_CAM_RIGHT, cx + face_w * 0.18, cy + face_h * 0.46)
    # hidung + bibir bawah
    place(config.NOSE_TIP, cx, cy + face_h * 0.08)
    place(config.LOWER_LIP_BOTTOM, cx, cy - face_h / 2 + face_h * lip_y_ratio)

    # iris: dipisah kiri/kanan, digeser oleh yaw_shift.
    # Posisi vertikal: pada wajah frontal, garis iris letaknya ~30% dari
    # landmark 10 (garis rambut) ke dagu. Angka ini penting: kalau mata
    # diletakkan terlalu dekat garis rambut, region search rambut akan menimpa
    # sklera dan test segmentasi jadi mengukur yang salah.
    iris_r = face_w * 0.075
    iris_y = cy - face_h * 0.20
    place(config.IRIS_CAM_LEFT_CENTER, cx - face_w * 0.21 + yaw_shift, iris_y)
    place(config.IRIS_CAM_RIGHT_CENTER, cx + face_w * 0.21 + yaw_shift, iris_y)
    for k, idx in enumerate((469, 470, 471, 472)):
        ang = k * np.pi / 2
        place(idx,
              cx - face_w * 0.21 + yaw_shift + iris_r * np.cos(ang),
              iris_y + iris_r * np.sin(ang))
    for k, idx in enumerate((474, 475, 476, 477)):
        ang = k * np.pi / 2
        place(idx,
              cx + face_w * 0.21 + yaw_shift + iris_r * np.cos(ang),
              iris_y + iris_r * np.sin(ang))

    # ring mata (isi saja supaya shape valid; tidak dipakai langsung)
    for idx in config.EYE_CAM_LEFT_RING:
        place(idx, cx - face_w * 0.21 + yaw_shift, iris_y)
    for idx in config.EYE_CAM_RIGHT_RING:
        place(idx, cx + face_w * 0.21 + yaw_shift, iris_y)

    # titik sampel kulit
    for i, idx in enumerate(config.SKIN_SAMPLE_POINTS):
        ang = 2 * np.pi * i / len(config.SKIN_SAMPLE_POINTS)
        place(idx,
              cx + np.cos(ang) * face_w * 0.32,
              cy - face_h * 0.22 + np.sin(ang) * face_h * 0.22)

    return lm


def bgr_to_lab(bgr):
    """BGR -> CIELAB asli (L* 0..100, a*/b* sekitar 0).

    PENTING: harus lewat opencv_lab_to_cielab. Nilai mentah OpenCV (L 0..255,
    a*/b* +128) bukan CIELAB, dan fixture yang memakainya membuat test lulus
    untuk kode yang salah.
    """
    import cv2
    from app.pipeline import opencv_lab_to_cielab

    px = np.array([[list(bgr)]], dtype=np.uint8)
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0]
    return opencv_lab_to_cielab(lab)


def bgr_to_hsv_opencv(bgr):
    """BGR -> (hue 0-179, sat 0-255, val 0-255) seperti OpenCV."""
    import cv2
    px = np.array([[list(bgr)]], dtype=np.uint8)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
    return np.array(hsv, dtype=np.float64)


def flat_lab_image(size=200, lab=(60.0, 14.0, 18.0), noise_seed=None):
    """Kanvas warna solid dalam BGR yang setara dengan nilai LAB tertentu."""
    import cv2
    bgr = cv2.cvtColor(
        np.array([[[lab[0], lab[1], lab[2]]]], dtype=np.uint8), cv2.COLOR_LAB2BGR
    )[0, 0]
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = bgr
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        noise = rng.normal(0, 2.0, img.shape)
        img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return img


def paint_region(img, box, bgr):
    x0, y0, x1, y1 = box
    img[y0:y1, x0:x1] = bgr