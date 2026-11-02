"""
test_quality_gate.py
--------------------
Test quality gate (Step 0). Ini yang versi lama tidak ada sama sekali --
versi lama memproses foto blur/3-4 samping/gelap lalu melaporkan hasilnya
seolah-olah valid.
"""

import numpy as np
import pytest

from app import config, pipeline
from tests.conftest import make_landmarks, flat_lab_image


def _sharp_image(h=480, w=640, seed=0):
    """Gambar tajam: noise fine-grained -> variance of Laplacian tinggi.
    Tidak butuh wajah sungguhan; quality gate hanya mengukur ketajaman,
    sedangkan exposure dihitung dari skin_lab yang kita berikan manual.
    """
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return img


def _blurry_image(h=480, w=640):
    import cv2
    flat = np.full((h, w, 3), 128, dtype=np.uint8)
    return cv2.GaussianBlur(flat, (0, 0), 12)


def _good_skin_lab(l=62.0):
    return np.array([l, 14.0, 18.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------

def test_frontal_landmarks_pass_pose_gate():
    lm = make_landmarks(yaw_shift=0.0, roll_deg=0.0)
    pose = pipeline.estimate_pose(lm)
    assert abs(pose["yaw_proxy"]) < 1e-6, "landmark frontal harus yaw_proxy ~ 0"
    assert abs(pose["roll_deg"]) < 1e-6, "landmark frontal harus roll ~ 0"


def test_yaw_shift_increases_yaw_proxy():
    frontal = pipeline.estimate_pose(make_landmarks(yaw_shift=0.0))
    turned = pipeline.estimate_pose(make_landmarks(yaw_shift=25.0))
    assert turned["yaw_proxy"] > frontal["yaw_proxy"], (
        "menggeser iris harus menaikkan yaw_proxy"
    )


def test_roll_shift_increases_roll_deg():
    flat = pipeline.estimate_pose(make_landmarks(roll_deg=0.0))
    tilted = pipeline.estimate_pose(make_landmarks(roll_deg=10.0))
    assert abs(tilted["roll_deg"]) > abs(flat["roll_deg"])


def _tilt_eye_landmarks(lm, deg):
    """Putar HANYA landmark mata/iris di sekitar titik tengah kedua iris.

    Gate roll dihitung dari garis penghubung iris, jadi memutar seluruh wajah
    sekaligus memalsukan gate yaw juga (titik tepi wajah ikut bergeser) -- test
    jadi ditolak karena yaw, bukan karena roll. Memutar hanya bagian mata
    mengisolasi sinyal roll yang memang diukur gate.
    """
    lm = lm.copy()
    eye_idx = (list(config.EYE_CAM_LEFT_RING) + list(config.EYE_CAM_RIGHT_RING)
               + list(config.IRIS_CAM_LEFT_RING) + list(config.IRIS_CAM_RIGHT_RING)
               + [config.IRIS_CAM_LEFT_CENTER, config.IRIS_CAM_RIGHT_CENTER])
    cx = (lm[config.IRIS_CAM_LEFT_CENTER][0] + lm[config.IRIS_CAM_RIGHT_CENTER][0]) / 2.0
    cy = (lm[config.IRIS_CAM_LEFT_CENTER][1] + lm[config.IRIS_CAM_RIGHT_CENTER][1]) / 2.0
    rad = np.radians(deg)
    cos_r, sin_r = np.cos(rad), np.sin(rad)
    for i in eye_idx:
        dx, dy = lm[i][0] - cx, lm[i][1] - cy
        lm[i, 0] = cx + dx * cos_r - dy * sin_r
        lm[i, 1] = cy + dx * sin_r + dy * cos_r
    return lm


def test_rolled_face_is_rejected():
    lm = _tilt_eye_landmarks(make_landmarks(), 30.0)
    img = _sharp_image()
    with pytest.raises(pipeline.ImageRejectedError) as exc:
        pipeline.assess_quality(img, lm, n_faces=1, skin_lab=_good_skin_lab())
    assert "miring" in str(exc.value).lower()


def test_profile_face_is_rejected():
    # geser iris jauh ke kanan -> simulates 3/4 profile
    lm = make_landmarks(yaw_shift=45.0)
    img = _sharp_image()
    with pytest.raises(pipeline.ImageRejectedError) as exc:
        pipeline.assess_quality(img, lm, n_faces=1, skin_lab=_good_skin_lab())
    assert "lurus" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Pitch
# ---------------------------------------------------------------------------

def _shift_eyes_vertically(lm, delta_px):
    """Geser garis iris ke atas/bawah untuk mensimulasikan kepala menengadah/
    mendongak. Hanya landmark mata yang digeser; geometri wajah tetap."""
    lm = lm.copy()
    eye_idx = (list(config.EYE_CAM_LEFT_RING) + list(config.EYE_CAM_RIGHT_RING)
               + list(config.IRIS_CAM_LEFT_RING) + list(config.IRIS_CAM_RIGHT_RING)
               + [config.IRIS_CAM_LEFT_CENTER, config.IRIS_CAM_RIGHT_CENTER])
    lm[eye_idx, 1] += delta_px
    return lm


def test_pitch_ratio_shifts_when_eyes_move_vertically():
    """dy > 0 = iris bergeser ke BAWAH di gambar.

    Kepala mendongak (chin naik, dahi menjauh) membuat jarak rambut-ke-iris
    mengecil sehingga rasio turun. Kepala menengadah sebaliknya.
    """
    neutral = pipeline.estimate_pose(make_landmarks())
    dy_down = pipeline.estimate_pose(_shift_eyes_vertically(make_landmarks(), 60.0))
    dy_up = pipeline.estimate_pose(_shift_eyes_vertically(make_landmarks(), -60.0))
    assert dy_down["pitch_ratio"] > neutral["pitch_ratio"]
    assert dy_up["pitch_ratio"] < neutral["pitch_ratio"]


def test_head_tilted_up_is_rejected():
    lm = _shift_eyes_vertically(make_landmarks(), -60.0)
    img = _sharp_image()
    with pytest.raises(pipeline.ImageRejectedError) as exc:
        pipeline.assess_quality(img, lm, n_faces=1, skin_lab=_good_skin_lab())
    assert "menengadah" in str(exc.value).lower()


def test_head_tilted_down_is_rejected():
    lm = _shift_eyes_vertically(make_landmarks(), 60.0)
    img = _sharp_image()
    with pytest.raises(pipeline.ImageRejectedError) as exc:
        pipeline.assess_quality(img, lm, n_faces=1, skin_lab=_good_skin_lab())
    assert "mendongak" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Multi-face
# ---------------------------------------------------------------------------

def test_multiple_faces_rejected():
    """Foto anak + orang dewasa. Versi lama (num_faces=1) diam-diam memilih
    satu wajah dan melaporkan hasil seolah itu subjek yang dimaksud."""
    lm = make_landmarks()
    img = _sharp_image()
    with pytest.raises(pipeline.ImageRejectedError) as exc:
        pipeline.assess_quality(img, lm, n_faces=2, skin_lab=_good_skin_lab())
    assert "wajah" in str(exc.value).lower()


def test_single_face_accepted():
    lm = make_landmarks()
    img = _sharp_image()
    report = pipeline.assess_quality(img, lm, n_faces=1, skin_lab=_good_skin_lab())
    assert report.ok is True


# ---------------------------------------------------------------------------
# Resolution & blur
# ---------------------------------------------------------------------------

def test_tiny_face_rejected():
    lm = make_landmarks(face_h=40.0)   # << min_face_height_px (160)
    img = _sharp_image()
    with pytest.raises(pipeline.ImageRejectedError) as exc:
        pipeline.assess_quality(img, lm, n_faces=1, skin_lab=_good_skin_lab())
    assert "kecil" in str(exc.value).lower()


def test_blurry_image_rejected():
    lm = make_landmarks()
    img = _blurry_image()
    with pytest.raises(pipeline.ImageRejectedError) as exc:
        pipeline.assess_quality(img, lm, n_faces=1, skin_lab=_good_skin_lab())
    assert "buram" in str(exc.value).lower()


def test_sharp_image_accepted():
    lm = make_landmarks()
    img = _sharp_image()
    report = pipeline.assess_quality(img, lm, n_faces=1, skin_lab=_good_skin_lab())
    assert report.ok is True


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------

def test_too_dark_skin_rejected():
    lm = make_landmarks()
    img = _sharp_image()
    # L*=15 jauh di bawah skin_l_min (38)
    with pytest.raises(pipeline.ImageRejectedError) as exc:
        pipeline.assess_quality(img, lm, n_faces=1, skin_lab=np.array([15.0, 14.0, 18.0]))
    assert "pencahayaan" in str(exc.value).lower()


def test_overexposed_skin_rejected():
    lm = make_landmarks()
    img = _sharp_image()
    # L*=120 jauh di atas skin_l_max (94)
    with pytest.raises(pipeline.ImageRejectedError) as exc:
        pipeline.assess_quality(img, lm, n_faces=1, skin_lab=np.array([120.0, 14.0, 18.0]))
    assert "pencahayaan" in str(exc.value).lower()


def test_good_exposure_accepted():
    lm = make_landmarks()
    img = _sharp_image()
    report = pipeline.assess_quality(img, lm, n_faces=1, skin_lab=_good_skin_lab())
    assert report.ok is True


# ---------------------------------------------------------------------------
# Urutan gate: gambar buruk (blur + profile + gelap) pasti ditolak
# yang penting DITOLAK, bukan diproses diam-diam.
# ---------------------------------------------------------------------------

def test_multiple_gate_failures_still_rejected():
    lm = make_landmarks(face_h=20.0, roll_deg=45.0, yaw_shift=60.0)
    img = _blurry_image()
    with pytest.raises(pipeline.ImageRejectedError):
        pipeline.assess_quality(img, lm, n_faces=3, skin_lab=np.array([5.0, 0.5, 1.0]))