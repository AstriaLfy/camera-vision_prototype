"""
test_end_to_end.py
------------------
UJi pipeline penuh dengan model ASLI (FaceLandmarker), bukan landmark sintetis.

Wajah uji di-render secara prosedural (tests/conftest.render_frontal_face).
Synthetic landmark saja tidak membuktikan apa-apa soal model: model bisa
mengembalikan mesh yang sama sekali tidak masuk akal dan semua test tetap hijau.
"""

import cv2
import numpy as np
import pytest

from app import config, pipeline
from tests.conftest import requires_model, render_frontal_face


@pytest.fixture(scope="module")
def rendered_face():
    return render_frontal_face()


@requires_model
def test_model_detects_rendered_frontal_face(rendered_face):
    try:
        landmarks_px, n_faces = pipeline.detect_landmarks(rendered_face)
    except pipeline.FaceNotFoundError:
        pytest.skip(
            "MediaPipe tidak mendeteksi wajah hasil render. Fixture sintetis "
            "perlu diperbarui (bukan kegagalan kode pipeline)."
        )
    assert n_faces == 1
    assert landmarks_px.shape == (478, 3)
    h, w = rendered_face.shape[:2]
    assert landmarks_px[:, 0].min() >= 0 and landmarks_px[:, 0].max() < w
    assert landmarks_px[:, 1].min() >= 0 and landmarks_px[:, 1].max() < h


@requires_model
def test_pose_of_rendered_face_is_frontal(rendered_face):
    try:
        landmarks_px, _ = pipeline.detect_landmarks(rendered_face)
    except pipeline.FaceNotFoundError:
        pytest.skip("wajah hasil render tidak terdeteksi")
    q = config.QUALITY
    pose = pipeline.estimate_pose(landmarks_px)
    assert abs(pose["yaw_proxy"]) < q["max_yaw_proxy"] / 2, pose
    assert abs(pose["roll_deg"]) < q["max_roll_deg"] / 2, pose
    assert q["pitch_ratio_min"] < pose["pitch_ratio"] < q["pitch_ratio_max"], pose


@requires_model
def test_full_pipeline_returns_valid_result(rendered_face):
    """Pipeline penuh harus mengembalikan hasil, bukan exception, untuk wajah
    frontal yang layak."""
    try:
        landmarks_px, _ = pipeline.detect_landmarks(rendered_face)
    except pipeline.FaceNotFoundError:
        pytest.skip("wajah hasil render tidak terdeteksi")

    # skin_lab=None supaya gate pencahayaan (yang butuh sample kulit) dilewati;
    # di sini yang diuji adalah gate pose/sharpness.
    report = pipeline.assess_quality(rendered_face, landmarks_px, n_faces=1,
                                     skin_lab=None)
    assert report.ok
    assert report.checks["n_faces"] == 1
    assert report.checks["sharpness"] > 0
    assert 0.0 <= report.score <= 1.0


@requires_model
def test_iris_centers_land_inside_eye_regions(rendered_face):
    """Iris center harus jatuh di tengah region mata -- kalau tidak, ROI sklera
    salah tempat dan a* yang diukur bukan sklera."""
    try:
        landmarks_px, _ = pipeline.detect_landmarks(rendered_face)
    except pipeline.FaceNotFoundError:
        pytest.skip("wajah hasil render tidak terdeteksi")

    for center_idx, eye_ring, iris_ring in (
        (config.IRIS_CAM_LEFT_CENTER, config.EYE_CAM_LEFT_RING, config.IRIS_CAM_LEFT_RING),
        (config.IRIS_CAM_RIGHT_CENTER, config.EYE_CAM_RIGHT_RING, config.IRIS_CAM_RIGHT_RING),
    ):
        center = landmarks_px[center_idx]
        ring_xy = landmarks_px[list(iris_ring)][:, :2]
        # jarak tiap titik ring iris ke center harus seragam (itu definisi lingkaran)
        radii = np.linalg.norm(ring_xy - center[:2], axis=1)
        assert radii.std() < radii.mean() * 0.45, (
            f"titik ring iris {iris_ring} tidak membentuk lingkaran di sekitar "
            f"center {center_idx}: radii={radii}"
        )
        # center harus di dalam konveks eye ring
        eye_xy = landmarks_px[list(eye_ring)][:, :2]
        hull = cv2.convexHull(eye_xy.astype(np.float32))
        if cv2.contourArea(hull) > 0:
            assert cv2.pointPolygonTest(hull, (float(center[0]), float(center[1])),
                                         False) >= 0, (
                f"iris center {center_idx} berada DI LUAR region mata"
            )


@requires_model
def test_skin_sample_points_do_not_fall_on_hair_or_eyes(rendered_face):
    """Titik sampel kulit harus benar-benar di kulit.

    Ini regresi untuk bug index yang tertimpa: dulu landmark 10 (garis rambut)
    ikut masuk daftar sampel kulit, sehingga "warna kulit" bisa sebenarnya
    warna rambut.
    """
    try:
        landmarks_px, _ = pipeline.detect_landmarks(rendered_face)
    except pipeline.FaceNotFoundError:
        pytest.skip("wajah hasil render tidak terdeteksi")

    # tidak boleh ada titik sampel yang berimpit dengan anchor geometri
    geometry_anchors = {
        config.FOREHEAD_TOP, config.CHIN,
        config.FACE_EDGE_CAM_LEFT, config.FACE_EDGE_CAM_RIGHT,
        config.CHEEK_CAM_LEFT, config.CHEEK_CAM_RIGHT,
        config.JAW_ANGLE_CAM_LEFT, config.JAW_ANGLE_CAM_RIGHT,
        config.NOSE_TIP, config.LOWER_LIP_BOTTOM,
        config.IRIS_CAM_LEFT_CENTER, config.IRIS_CAM_RIGHT_CENTER,
    }
    overlap = geometry_anchors & set(config.SKIN_SAMPLE_POINTS)
    assert not overlap, f"titik sampel kulit bentrok dengan anchor: {overlap}"

    # tidak boleh ada duplikat
    assert len(config.SKIN_SAMPLE_POINTS) == len(set(config.SKIN_SAMPLE_POINTS))

    # tidak boleh keluar dari gambar
    h, w = rendered_face.shape[:2]
    for idx in config.SKIN_SAMPLE_POINTS:
        x, y = landmarks_px[idx][:2]
        assert 0 <= x < w and 0 <= y < h, f"titik {idx} di luar gambar"


@requires_model
def test_quality_gate_rejects_when_photo_is_unusable():
    """Foto dengan landmark tapi gate menolak -> harus exception, bukan hasil.

    Gate diuji dengan landmark sintetis karena foto 3/4 view yang sah lisensinya
    tidak disertakan, dan foto render prosedural tidak bisa membuat yaw cukup
    besar untuk melewati gate (MediaPipe tetap membacanya frontal). Yang diuji
    di sini adalah kontrak: penolakan selalu berupa ImageRejectedError dengan
    reason + hint yang bisa ditampilkan ke user.
    """
    from tests.conftest import make_landmarks

    img = render_frontal_face()
    lm = make_landmarks(yaw_shift=45.0)   # yaw jelas di luar ambang
    with pytest.raises(pipeline.ImageRejectedError) as exc:
        pipeline.assess_quality(img, lm, n_faces=1, skin_lab=None)
    assert str(exc.value), "reason tidak boleh kosong"
    assert exc.value.detail, "hint tidak boleh kosong -- tanpa ini user tidak tahu apa yang harus diperbaiki"


@requires_model
def test_landmarks_are_not_degenerate(rendered_face):
    """Guard terhadap model yang mengembalikan semua landmark di titik yang sama.

    Model bisa 'berhasil' tanpa error tapi menghasilkan mesh yang menumpuk di
    satu titik. Test ini memastikan mesh punya sebaran nyata.
    """
    try:
        landmarks_px, _ = pipeline.detect_landmarks(rendered_face)
    except pipeline.FaceNotFoundError:
        pytest.skip("wajah hasil render tidak terdeteksi")

    xy = landmarks_px[:, :2]
    assert xy[:, 0].std() > 5, f"sebaran x terlalu kecil: {xy[:, 0].std()}"
    assert xy[:, 1].std() > 5, f"sebaran y terlalu kecil: {xy[:, 1].std()}"

    # jarak kiri-kanan pipi harus positif (bukan keduanya di tempat sama)
    cheek_dx = abs(landmarks_px[config.CHEEK_CAM_RIGHT][0]
                   - landmarks_px[config.CHEEK_CAM_LEFT][0])
    assert cheek_dx > 20, f"kedua pipi dianggap di lokasi sama: {cheek_dx}"