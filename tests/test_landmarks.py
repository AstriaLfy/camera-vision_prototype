"""
test_landmarks.py
-----------------
Verifikasi index landmark terhadap model MediaPipe ASLI.

Ini test yang tidak bisa dilakukan versi lama: dulu index landmark cuma
"diasumsikan benar" dari dokumentasi. Sekarang dicek empiris ke
models/face_landmarker.task, jadi kalau model/Dokumentasi berubah kita tahu.

Yang diuji:
  1. detectado 478 landmark (bukan 468)
  2. iris center 468 milik ring mata KIRI gambar, 473 milik ring KANAN gambar
  3. landmark oval wajah berada di posisi yang masuk akal
  4. pasangan pipi & rahang simetris
"""

import numpy as np
import pytest

from app import config, pipeline
from tests.conftest import requires_model


@requires_model
def test_detect_returns_478_landmarks(face_image_bgr):
    landmarks, n_faces = pipeline.detect_landmarks(face_image_bgr)
    assert landmarks.shape == (478, 3), (
        f"Harusnya 478 landmark (termasuk 468-477 untuk iris), "
        f"dapat {landmarks.shape}"
    )
    assert n_faces >= 1


@requires_model
def test_iris_centers_pair_with_correct_eye_ring(face_image_bgr):
    """Yang paling penting untuk integritas pipeline.

    Landmark 468 & 473 adalah pusat iris. Kalau keduanya tertukar, seluruh
    fitur kepucatan dihitung dari mata yang salah -- dan karena kedua mata
    biasanya serupa, gejalanya subtle: pipeline tetap jalan tapi tidak
    mengukur apa yang dikira diukur.
    """
    landmarks, _ = pipeline.detect_landmarks(face_image_bgr)
    h, w = face_image_bgr.shape[:2]
    x = lambda i: landmarks[i][0]
    y = lambda i: landmarks[i][1]

    iris_l = x(config.IRIS_CAM_LEFT_CENTER)   # index 468
    iris_r = x(config.IRIS_CAM_RIGHT_CENTER)  # index 473

    ring_l = np.mean([x(i) for i in config.EYE_CAM_LEFT_RING])   # ring 33..133
    ring_r = np.mean([x(i) for i in config.EYE_CAM_RIGHT_RING])  # ring 263..362

    # 468 harus dekat ring KIRI gambar; 473 dekat ring KANAN gambar
    assert abs(iris_l - ring_l) < abs(iris_l - ring_r), (
        f"iris {config.IRIS_CAM_LEFT_CENTER} (x={iris_l:.0f}) ternyata dekat ring "
        f"kanan (x={ring_r:.0f}), bukan ring kiri (x={ring_l:.0f}). "
        "Konvensi CAM_LEFT/CAM_RIGHT di config.py kemungkinan terbalik."
    )
    assert abs(iris_r - ring_r) < abs(iris_r - ring_l)

    # dan iris kiri harus memang di kiri iris kanan pada subyek frontal
    assert iris_l < iris_r, (
        f"iris kiri (x={iris_l:.0f}) tidak di kiri iris kanan (x={iris_r:.0f}); "
        "foto mungkin bukan frontal, atau konvensi terbalik."
    )

    # jarak iris kiri-ke-kanan harus masuk akal (bukan 0, bukan separuh wajah)
    assert 0.1 * w < (iris_r - iris_l) < 0.6 * w

    _ = y


@requires_model
def test_face_oval_landmarks_are_where_they_claim(face_image_bgr):
    """Verifikasi klaim di config.py: 10=garis rambut dahi, 152=dagu, dll."""
    landmarks, _ = pipeline.detect_landmarks(face_image_bgr)

    forehead = landmarks[config.FOREHEAD_TOP]
    chin = landmarks[config.CHIN]
    edge_l = landmarks[config.FACE_EDGE_CAM_LEFT]
    edge_r = landmarks[config.FACE_EDGE_CAM_RIGHT]
    nose = landmarks[config.NOSE_TIP]

    # forehead di atas dagu
    assert forehead[1] < chin[1], "landmark 'dahi' harus di atas landmark 'dagu'"
    # edge kiri benar-benar di kiri edge kanan (dalam koordinat gambar)
    assert edge_l[0] < edge_r[0], "tepi kiri gambar harus punya x lebih kecil"
    # hidung di antara dua tepi wajah
    assert edge_l[0] < nose[0] < edge_r[0], "hidung harus di antara dua tepi wajah"
    # Landmark 10 di face oval = garis rambut, BUKAN puncak kepala.
    # Kalau ada uplift berarti 10 bukan batas atas kepala, dan asumsi
    # "landmark 10 = puncak dahi" di config.py salah.
    assert nose[1] < chin[1], "ujung hidung harus di atas dagu"


@requires_model
def test_cheek_and_jaw_pairs_are_symmetric_in_y(face_image_bgr):
    """Pasangan pipi & rahang harus simetris terhadap sumbu tengah wajah."""
    landmarks, _ = pipeline.detect_landmarks(face_image_bgr)

    mid_x = (landmarks[config.FACE_EDGE_CAM_LEFT][0]
             + landmarks[config.FACE_EDGE_CAM_RIGHT][0]) / 2.0

    for left_i, right_i, name in [
        (config.CHEEK_CAM_LEFT, config.CHEEK_CAM_RIGHT, "pipi"),
        (config.JAW_ANGLE_CAM_LEFT, config.JAW_ANGLE_CAM_RIGHT, "rahang"),
    ]:
        l, r = landmarks[left_i], landmarks[right_i]
        assert l[0] < mid_x < r[0], f"pasangan {name} tidak mengapit sumbu wajah"
        # simetri vertikal dalam toleransi:landmark bisa bergeser karena pose
        tol = abs(landmarks[config.CHIN][1] - landmarks[config.FOREHEAD_TOP][1]) * 0.08
        assert abs(l[1] - r[1]) < tol, (
            f"pasangan {name} tidak setinggi: y={l[1]:.1f} vs {r[1]:.1f} "
            f"(toleransi {tol:.1f})"
        )


@requires_model
def test_estimate_pose_is_small_for_frontal_photo(face_image_bgr):
    landmarks, _ = pipeline.detect_landmarks(face_image_bgr)
    pose = pipeline.estimate_pose(landmarks)

    assert pose["face_height_px"] > 0
    assert pose["face_width_px"] > 0
    # Fixture adalah wajah frontal yang benar-benar dirender frontal, jadi di
    # sini kita boleh assert ketat. Dulu fixture-nya foto 3/4 view sehingga nilai
    # hanya bisa checked longgar -- sekarang tidak perlu toleransi broad lagi.
    assert abs(pose["yaw_proxy"]) < 0.05, pose
    assert abs(pose["roll_deg"]) < 2.0, pose
    assert config.QUALITY["pitch_ratio_min"] < pose["pitch_ratio"] \
        < config.QUALITY["pitch_ratio_max"], pose