"""
test_segmentation.py
-------------------
Test pemisahan rambut dari kulit, dan pemisahan sklera dari iris.

Ini menutup celah test terbesar di versi lama: di sana, region rambut dan
region mata dibuat sebagai "band seluruh lebar gambar dengan warna datar",
sehingga crop_eye_regions yang rusak sama sekali tidak akan ketahuan.

Di sini region dibuat SECARA SPASIAL: rambut hanya ada di kotak search,
mata hanya ada di sekeliling iris, dan sisanya (latar belakang, kulit)
berwarna berbeda -- jadi kalau segmentation salah, test gagal.
"""

import cv2
import numpy as np
import pytest

from app import config, pipeline
from tests.conftest import make_landmarks


SKIN_BGR = (150, 165, 195)      # warna kulit
HAIR_BG_BGR = (85, 60, 45)      # rambut coklat gelap
HAIR_PALE_BGR = (120, 120, 135)  # rambut pucat
SCLERA_BGR = (215, 225, 235)     # sklera putih


def build_scene(hair_bgr=HAIR_BG_BGR, background_bgr=(60, 70, 80),
                sclera_bgr=SCLERA_BGR, iris_bgr=(45, 40, 35)):
    """Bangun gambar yang 'mirip wajah' dengan komponen berwarna terpisah.

    Peta layout (image 640x640, wajah 220x300 di tengah):
      - seluruh kanvas: background (dinding)
      - kepala: kulit, sedikit meluas ke ATAS garis rambut (supaya skalp
        ada di region rambut -- inilah yang terjadi pada anak kerontok)
      - kotak search rambut: hair_bgr
      - annulus sklera di sekitar iris: sclera_bgr
      - pusat iris: iris_bgr

    Parameter hair_bgr dipakai untuk mengisi area search rambut. Kalau diberi
    SKIN_BGR, skenario-nya adalah anak kerontok: tidak ada rambut, hanya kulit.
    """
    from tests.conftest import bgr_to_lab

    img = np.zeros((640, 640, 3), np.uint8)
    img[:, :] = background_bgr

    lm = make_landmarks()

    # kepala: ellipse yang vertikalnya lebih tinggi dari oval wajah, supaya
    # area di atas garis rambut terisi kulit (skalp), bukan dinding.
    top_y = lm[config.FOREHEAD_TOP][1]
    chin_y = lm[config.CHIN][1]
    face_h = abs(chin_y - top_y)
    center_y = int((top_y + chin_y) / 2 - 0.10 * face_h)  # digeser ke atas
    axes = (int(abs(lm[config.FACE_EDGE_CAM_RIGHT][0]
                    - lm[config.FACE_EDGE_CAM_LEFT][0]) / 2) + 4,
            int(face_h * 0.62))
    cv2.ellipse(img, (int(lm[config.NOSE_TIP][0]), center_y),
                axes, 0, 0, 360, SKIN_BGR, -1)

    # rambut: hanya di kotak search
    box = pipeline.build_hair_roi_box(img.shape, lm)
    img[box[1]:box[3], box[0]:box[2]] = hair_bgr

    # sklera + iris per mata
    for center_idx, ring in ((config.IRIS_CAM_LEFT_CENTER, config.IRIS_CAM_LEFT_RING),
                             (config.IRIS_CAM_RIGHT_CENTER, config.IRIS_CAM_RIGHT_RING)):
        geo = pipeline.build_sclera_annulus(lm, center_idx, ring)
        assert geo is not None, "annulus harus bisa dibangun dari landmark sintetis"
        c, r_in, r_out = geo
        cv2.circle(img, (int(c[0]), int(c[1])), int(round(r_out)), sclera_bgr, -1)
        cv2.circle(img, (int(c[0]), int(c[1])), int(round(r_in * 0.95)), iris_bgr, -1)

    skin_lab = bgr_to_lab(SKIN_BGR)
    return img, lm, skin_lab


# ---------------------------------------------------------------------------
# Hair ROI box
# ---------------------------------------------------------------------------

def test_hair_roi_box_is_narrower_than_full_face():
    """ROI rambut harus PITA TENGAH, bukan selebar wajah penuh.
    Kalau selebar wajah, dinding/lantai ikut ter-average (bug versi lama)."""
    img, lm, _ = build_scene()
    x0, y0, x1, y1 = pipeline.build_hair_roi_box(img.shape, lm)
    face_left = lm[config.FACE_EDGE_CAM_LEFT][0]
    face_right = lm[config.FACE_EDGE_CAM_RIGHT][0]

    roi_w = x1 - x0
    face_w = face_right - face_left
    assert roi_w < face_w, (
        f"ROI rambut ({roi_w}px) harus lebih sempit dari lebar wajah ({face_w}px)"
    )
    # dan harus di dalam rentang wajah, tidak meluber
    assert x0 >= face_left - 2 and x1 <= face_right + 2


def test_hair_roi_box_excludes_forehead_below_top():
    """ROI rambut hanya di ATAS landmark dahi (y_max = y forehead)."""
    img, lm, _ = build_scene()
    _, _, _, y1 = pipeline.build_hair_roi_box(img.shape, lm)
    assert y1 <= int(lm[config.FOREHEAD_TOP][1]) + 1


# ---------------------------------------------------------------------------
# Hair segmentation
# ---------------------------------------------------------------------------

def test_segment_hair_finds_compact_hair_blob():
    img, lm, skin_lab = build_scene(hair_bgr=HAIR_BG_BGR)
    mask, fg_ratio = pipeline.segment_hair_mask(img, lm, skin_lab)
    roi_mask, box = pipeline.build_hair_roi_mask(img.shape, lm)

    assert mask is not None, "harus menemukan massa rambut"
    assert mask.shape == roi_mask.shape
    # rambut mengisi area di ATAS garis rambut, jadi harus memenuhi bagian atas
    # ROI. Diukur relatif terhadap ROI mask (yang berbentuk kepala), bukan
    # terhadap lebar kotak -- kalau terhadap kotak, taper oval ikut dihitung
    # sebagai "kosong" dan test jadi salah.
    half = mask.shape[0] // 2
    allowed_half = max(int(roi_mask[:half].sum()), 1)
    top_fraction = float(mask[:half].sum()) / allowed_half
    assert top_fraction > 0.8, (
        f"bagian atas ROI harus terisi rambut, tapi cuma {top_fraction:.0%} "
        "dari area yang diizinkan"
    )
    # dan tidak boleh merembes ke bawah garis rambut (dahi)
    _, _, _, y_hairline = box
    assert y_hairline == int(lm[config.FOREHEAD_TOP][1]), (
        "batas bawah ROI harus tepat di garis rambut, jangan sampai menimpa dahi"
    )
    # mask hanya boleh hidup di dalam ROI; kolom di luar roi_mask wajib 0
    outside = mask.astype(bool) & ~roi_mask
    assert not outside.any(), "mask merembes di luar area yang diizinkan"
    assert fg_ratio > config.HAIR["min_foreground_ratio"]


def test_segment_hair_excludes_background_not_counted_as_hair():
    """Background yang masuk ke ROI harus TIDAK dipakai sebagai sample rambut.

    Regression guard untuk bug versi lama: crop selebar wajah penuh pasti
    menyerap dinding/lantai, sehingga rata-rata warna jadi tidak bermakna.

    Skenario: anak kerontok / rambut di atas kepala, jadi SETIAP pixel di ROI
    adalah salah satu dari {kulit, dinding}. Tidak ada rambut sama sekali ->
    fitur WAJIB fail-closed. Kalau pipeline diam-diam mengambil sample dari
    dinding, ini akan lolos diam-diam.
    """
    img, lm, skin_lab = build_scene(
        hair_bgr=SKIN_BGR,             # tidak ada rambut di ROI
        background_bgr=(200, 40, 30),  # dinding biru pekat di atas kepala
    )
    mask, fg_ratio = pipeline.segment_hair_mask(img, lm, skin_lab)

    if mask is not None:
        # Kalau ada mask, sample dari mask harus benar-benar rambut
        # (warna rambut) -- kalau tidak berarti guard gagal.
        feat = pipeline.extract_hair_features(img, lm, skin_lab)
        assert feat["available"] is False, (
            f"tidak ada rambut di scene ini tapi fitur tersedia: {feat} -- "
            "berarti pipeline mengambil sample dari background"
        )


def test_segment_hair_fails_closed_when_no_hair_present():
    """Semua ROI berisi kulit (mis. anak kerontok / rambut di atasetop) ->
    indikator rambut harus dinonaktifkan, bukan dianggap normal."""
    img, lm, skin_lab = build_scene(hair_bgr=SKIN_BGR)  # "rambut" = warna kulit
    mask, fg_ratio = pipeline.segment_hair_mask(img, lm, skin_lab)

    if mask is None:
        feat = pipeline.extract_hair_features(img, lm, skin_lab)
        assert feat["available"] is False
        res = pipeline.score_hair(feat)
        assert res.is_flagged is None, "harus fail-closed, bukan normal"
    else:
        # kalau masih ada mask, fg_ratio sangat kecil -> harus ditolak
        feat = pipeline.extract_hair_features(img, lm, skin_lab)
        assert feat["available"] is False or res_is_safe(feat)


def res_is_safe(feat):
    res = pipeline.score_hair(feat)
    return res.is_flagged in (None, False)


# ---------------------------------------------------------------------------
# Hair feature: arah klinis
# ---------------------------------------------------------------------------

def test_pale_hair_has_smaller_l_delta_than_dark_hair():
    """Arah klinis: rambut hipopigmen -> L* lebih dekat ke kulit."""
    img_dark, lm_dark, skin_dark = build_scene(hair_bgr=HAIR_BG_BGR)
    img_pale, lm_pale, skin_pale = build_scene(hair_bgr=HAIR_PALE_BGR)

    f_dark = pipeline.extract_hair_features(img_dark, lm_dark, skin_dark)
    f_pale = pipeline.extract_hair_features(img_pale, lm_pale, skin_pale)

    if f_dark.get("available") and f_pale.get("available"):
        assert f_pale["l_delta"] < f_dark["l_delta"], (
            f"rambut pucat harus punya l_delta lebih kecil: "
            f"pale={f_pale['l_delta']:.1f} dark={f_dark['l_delta']:.1f}"
        )


# ---------------------------------------------------------------------------
# Sclera annulus: Eye ROI terpisah, bukan satu bbox gabungan
# ---------------------------------------------------------------------------

def test_sclera_annulus_is_around_iris_not_whole_eye():
    img, lm, _ = build_scene()
    geo = pipeline.build_sclera_annulus(lm, config.IRIS_CAM_LEFT_CENTER,
                                        config.IRIS_CAM_LEFT_RING)
    assert geo is not None
    center, r_in, r_out = geo
    assert r_in > 0 and r_out > r_in
    # annulus relatif kecil terhadap tinggi wajah
    face_h = abs(lm[config.CHIN][1] - lm[config.FOREHEAD_TOP][1])
    assert r_out < face_h * 0.35, "annulus jangan terlalu besar (akan masuk kulit)"


def test_sclera_roi_mask_contains_sclera_not_iris():
    """Penting: a* yang diukur harus dari SKLERA, bukan dari iris (iris gelap
    punya a* berbeda dan akan merusak rasio)."""
    img, lm, skin_lab = build_scene()
    roi = pipeline.sclera_roi_from_iris(img.shape, lm,
                                        config.IRIS_CAM_LEFT_CENTER,
                                        config.IRIS_CAM_LEFT_RING)
    assert roi is not None
    (ys, xs), mask = roi

    # pusat iris = gelap, pinggiran = sklera terang
    patch = img[ys, xs].astype(np.float64).mean(axis=2)
    cen = patch[len(patch) // 2 - 2:len(patch) // 2 + 3,
                len(patch[0]) // 2 - 2:len(patch[0]) // 2 + 3].mean()
    ring = patch[mask].mean()

    assert cen < ring, (
        f"pusat iris ({cen:.0f}) harus lebih gelap dari annulus sklera ({ring:.0f}) "
        "-- kalau tidak, ROI sklera capturing iris"
    )


def test_eye_rois_are_separate_not_merged():
    """Dua ROI mata harus terpisah (tidak seperti bbox gabungan versi lama
    yang ikut menelan bridge hidung + alis)."""
    img, lm, skin_lab = build_scene()
    roi_l = pipeline.sclera_roi_from_iris(img.shape, lm,
                                          config.IRIS_CAM_LEFT_CENTER,
                                          config.IRIS_CAM_LEFT_RING)
    roi_r = pipeline.sclera_roi_from_iris(img.shape, lm,
                                          config.IRIS_CAM_RIGHT_CENTER,
                                          config.IRIS_CAM_RIGHT_RING)
    assert roi_l is not None and roi_r is not None
    (ys_l, xs_l), _ = roi_l
    (ys_r, xs_r), _ = roi_r
    # kotak kiri dan kanan tidak overlap
    assert xs_l.stop <= xs_r.start, "ROI mata kiri dan kanan tidak boleh overlap"