"""
test_color.py
-------------
Test fitur warna & scoring.

Bagian penting: file ini menguji hal-hal yang versi LAMA tidak pernah uji --
khususnya bahwa indikator "rambut kemerahan" versi lama salah arah.
"""

import numpy as np
import pytest

from app import config, pipeline
from tests.conftest import bgr_to_hsv_opencv, bgr_to_lab


# ---------------------------------------------------------------------------
# Regresi: aturan versi lama salah secara klinis
# ---------------------------------------------------------------------------

LEGACY_HAIR_RULE = dict(
    hair_reddish_hue_max=20,
    hair_reddish_hue_min_wrap=160,
    hair_saturation_min=60,
    hair_value_min=40,
)


def legacy_score_hair_hsv(hue, sat, val):
    t = LEGACY_HAIR_RULE
    is_reddish = (hue <= t["hair_reddish_hue_max"]) or (hue >= t["hair_reddish_hue_min_wrap"])
    return bool(is_reddish and sat > t["hair_saturation_min"] and val > t["hair_value_min"])


@pytest.mark.parametrize("name,bgr,legacy_should_flag,note", [
    # (nama, BGR, legacy_flag, catatan)
    ("kwashiorkor reddish yellow (deskripsi Medscape)", (40, 150, 180), False,
     "Ini warna rambut yang JUSTRUS dideskripsikan di literatur."),
    ("kwashiorkor hypopigmented pale", (110, 120, 140), False,
     "Pucat tapi kurang saturated."),
    ("coklat gelap Asia (normal)", (45, 45, 60), True,
     "Rambut normal_false positive di aturan lama."),
    ("rambut dicat merah (bukan tanda malnutrisi)", (20, 30, 190), True,
     "DYE, bukan kondisi klinis."),
    ("pirang terang", (90, 100, 150), True,
     "Albedo tinggi normal."),
])
def test_legacy_hair_rule_is_clinically_backwards(name, bgr, legacy_should_flag, note):
    """Dokumentasi executor- bug: aturan lama tidak menangkap tanda klinis."""
    hue, sat, val = bgr_to_hsv_opencv(bgr)
    legacy_flag = legacy_score_hair_hsv(hue, sat, val)

    assert legacy_flag == legacy_should_flag, (
        f"Fixtur tes tidak merepresentasikan yang diklaim: {name} -> {legacy_flag}"
    )
    if legacy_should_flag:
        pytest.xfail(
            f"Known legacy bug: '{name}' salah di-flag oleh aturan lama. {note}"
        )


def test_kwashiorkor_reddish_yellow_would_be_missed_by_legacy_rule():
    """Tegas: warna rambut yang literatur sebut sebagai tanda kwashiorkor
    justru TIDAK tertangkap aturan hue versi lama."""
    hue, sat, val = bgr_to_hsv_opencv((40, 150, 180))
    # Aturan lama butuh hue <= 20 (merah pekat).
    # Tapi "reddish yellow to white" itu hue oranye-kuning (~24).
    assert hue > 20, (
        f" unexpectedly hue={hue} <= 20 -- kalau ini berubah, "
        "claims ini perlu ditinjau ulang."
    )
    assert legacy_score_hair_hsv(hue, sat, val) is False


# ---------------------------------------------------------------------------
# Pipeline baru: arah klinis yang benar (L* naik, chroma turun)
# ---------------------------------------------------------------------------

def _hair_features_from_pixels(hair_lab, skin_lab_tuple, available=True):
    """Bypass segmentasi: langsung feed fitur ke scoring, supaya unit test
    fokus ke arah scoring bukan ke segmentasi."""
    hair_lab = np.array(hair_lab, dtype=np.float64)
    skin = np.array(skin_lab_tuple, dtype=np.float64)
    return {
        "available": available,
        "hair_l": float(hair_lab[0]),
        "hair_chroma": float(np.hypot(hair_lab[1], hair_lab[2])),
        "skin_l": float(skin[0]),
        "skin_chroma": float(np.hypot(skin[1], skin[2])),
        "l_delta": float(skin[0] - hair_lab[0]),
        "chroma_ratio": float(np.hypot(hair_lab[1], hair_lab[2]) / np.hypot(skin[1], skin[2])),
        "foreground_ratio": 0.4,
        "n_pixels": 500,
    }


def test_hair_scoring_flags_hypopigmented_hair():
    """Rambut hipopigmen: L* mendekati kulit DAN chroma anjlok -> flag.

    SKENARIO KLINIS YANG BENAR: anak yang rambutnya tadinya hitam, lalu
    pigmentation-nya berkurang karena defisiensi protein -> rambut pucat
    (L* naik mendekati kulit) DAN chroma turun. persis inilah yang
    dideskripsikan "reddish yellow to white" (Medscape) / "gray-blond" (CDC).

    Fixture dibuat ekstrem supaya ambang config (l_delta_flag=12,
    chroma_ratio_flag=0.30) terlampaui jelas.
    """
    skin = (60.0, 14.0, 18.0)
    # rambut hipopigmen: L*=58 (hampir sama dengan kulit 60 -> l_delta=2,
    # jauh di bawah l_delta_flag=12), chroma sangat rendah (rasio ~0.24,
    # di bawah chroma_ratio_flag=0.30). Kedua arah abnormal.
    hair = (58.0, 3.0, 4.0)
    feat = _hair_features_from_pixels(hair, skin)
    res = pipeline.score_hair(feat)
    assert res.is_flagged is True, (
        f"rambut hipopigmen harus flag; dapat {res}"
    )
    assert res.normalized and res.normalized > 0


def test_hair_scoring_partial_signal_keeps_partial_severity():
    """Kalau hanya SATU dari dua arah (L* ATAU chroma) melewati titik flag-nya,
    jangan langsung flag -- biar rambut hitam pekat yang memang low-chroma tidak
    otomatis dianggap malnourished. Severitynya tidak nol, karena L* sudah
    mendekati abnormal (l_delta di antara flag dan max)."""
    skin = (60.0, 14.0, 18.0)
    # l_delta=18 (antara l_delta_flag=12 dan l_delta_max=24 -> komponen L*
    # ~0.5) tapi chroma jeblok (rasio ~0.24, di bawah flag chroma 0.30).
    # Hanya 1 dari 2 arah melewati flag.
    hair = (42.0, 3.0, 4.0)
    feat = _hair_features_from_pixels(hair, skin)
    res = pipeline.score_hair(feat)
    # bukan flag: komponen chroma tidak melewati flag DAN L* tidak di bawah flag
    assert res.is_flagged is False
    # tapi normalized tetap di antara 0 dan 1 (ada info, tidak discard total)
    assert 0.0 < res.normalized < 1.0, res


def test_hair_scoring_chroma_alone_contributes_no_severity():
    """Rambut yang LEBIH GELAP dari kulit sama sekali (l_delta di atas
    l_delta_max) tidak mungkin hipopigmen, jadi chroma yang rendah tidak boleh
    menyumbang severity.

    Tanpa aturan ini, rambut hitam yang secara alami low-chroma selalu menambah
    skor -- artinya SETIAP anak berambut hitam dapat skor > 0."""
    skin = (62.0, 14.0, 18.0)
    hair = (20.0, 3.0, 4.0)   # l_delta=42, chroma_ratio=5/22.8=0.22
    feat = _hair_features_from_pixels(hair, skin)
    assert feat["l_delta"] > config.HAIR["l_delta_max"]
    assert feat["chroma_ratio"] <= config.HAIR["chroma_ratio_flag"]
    res = pipeline.score_hair(feat)
    assert res.is_flagged is False
    assert res.normalized == 0.0, (
        f"chroma saja tidak boleh menambah severity; dapat {res.normalized}"
    )


def test_hair_scoring_does_not_flag_normal_dark_hair():
    """Rambut hitam/coklat normal: L* jauh dari kulit, chroma tidak jeblok."""
    skin = (62.0, 14.0, 18.0)
    # rambut hitam: L*=25 (jauh lebih gelap dari kulit), chroma sangat rendah
    # tapi ini NORMAL -- harusnya tidak flag hanya karena chroma rendah.
    hair = (25.0, 4.0, 3.0)
    feat = _hair_features_from_pixels(hair, skin)
    res = pipeline.score_hair(feat)
    assert res.is_flagged is False, (
        f"rambut hitam normal tidak boleh flag; dapat {res}"
    )


def test_hair_scoring_is_fail_closed():
    feat = {"available": False, "reason": "rambut tidak terlihat"}
    res = pipeline.score_hair(feat)
    assert res.is_flagged is None, (
        "data hilang harus is_flagged=None (fail-closed), bukan False"
    )
    assert res.value is None


# ---------------------------------------------------------------------------
# Sklera: CIELAB a* dinormalisasi, bukan brightness absolut
# ---------------------------------------------------------------------------

def _sclera_features(sclera_a, skin_a):
    return {
        "available": True,
        "sclera_a_star": sclera_a,
        "skin_a_star": skin_a,
        "a_star_ratio": sclera_a / skin_a,
        "n_eyes": 2,
        "n_pixels": 400,
    }


def test_sclera_scoring_flags_high_a_ratio():
    """a* sklera mendekati a* kulit = pucat."""
    res = pipeline.score_sclera(_sclera_features(sclera_a=12.5, skin_a=14.0))
    assert res.is_flagged is True
    assert res.normalized and res.normalized > 0


def test_sclera_scoring_does_not_flag_healthy():
    """a* sklera jelas lebih rendah dari kulit = sehat."""
    res = pipeline.score_sclera(_sclera_features(sclera_a=6.0, skin_a=14.0))
    assert res.is_flagged is False


def test_sclera_scoring_is_skin_tone_invariant():
    """Feature yang sama secara klinis harus flagged sama untuk kulit gelap
    maupun kulit terang. Ini yang tidak bisa dilakukan brightness absolut."""
    # subyek kulit terang, sklera sehat
    bright_normal = pipeline.score_sclera(_sclera_features(sclera_a=6.0, skin_a=14.0))
    # subyek kulit gelap, sklera sehat (nilai absolut lebih rendah, tapi
    # RASIO terhadap kulit sama sehat)
    dark_normal = pipeline.score_sclera(_sclera_features(sclera_a=3.5, skin_a=7.0))
    assert bright_normal.is_flagged is False
    assert dark_normal.is_flagged is False


def test_sclera_scoring_is_fail_closed():
    feat = {"available": False, "reason": "kurang pixel"}
    res = pipeline.score_sclera(feat)
    assert res.is_flagged is None


def test_sclera_extract_fails_closed_on_invalid_skin():
    """a* kulit <= 0 -> sample warna tidak valid -> fail closed."""
    skin_lab = np.array([60.0, 0.2, 10.0])  # a* ~0
    out = pipeline.extract_sclera_features(
        np.zeros((100, 100, 3), np.uint8), np.zeros((478, 3)), skin_lab
    )
    assert out["available"] is False


# ---------------------------------------------------------------------------
# Geometri
# ---------------------------------------------------------------------------

def test_geometry_scoring_flags_narrow_cheeks():
    feat = {
        "available": True,
        "cheek_to_jaw_ratio": 0.90,     # di bawah min -> pipi relatif sempit
        "lower_face_height_ratio": 0.38,
    }
    res = pipeline.score_geometry(feat)
    assert res.is_flagged is True


def test_geometry_scoring_normal():
    """Nilai yang terukur pada dua wajah acuan (render 1.224/0.217 dan foto asli
    1.234/0.131) harus lolos tanpa flag."""
    for ck, lf in ((1.224, 0.217), (1.234, 0.131), (1.18, 0.167)):
        feat = {"available": True, "cheek_to_jaw_ratio": ck,
                "lower_face_height_ratio": lf}
        res = pipeline.score_geometry(feat)
        assert res.is_flagged is False, f"cheek/jaw={ck} lower={lf} -> {res}"
        assert res.normalized == 0.0


def test_geometry_scoring_flags_narrow_cheeks():
    """Pipi relatif terhadap rahang menyempit -> rasio turun."""
    feat = {"available": True, "cheek_to_jaw_ratio": 0.95,
            "lower_face_height_ratio": 0.167}
    res = pipeline.score_geometry(feat)
    assert res.is_flagged is True


def test_geometry_scoring_flags_out_of_range_lower_face():
    """lower face height terlalu pendek (di luar rentang normal)."""
    feat = {
        "available": True,
        "cheek_to_jaw_ratio": 1.22,
        "lower_face_height_ratio": 0.06,   # jauh di bawah min 0.11
    }
    res = pipeline.score_geometry(feat)
    assert res.is_flagged is True


def test_geometry_scoring_flags_too_long_lower_face():
    feat = {"available": True, "cheek_to_jaw_ratio": 1.22,
            "lower_face_height_ratio": 0.31}   # di atas max 0.24
    res = pipeline.score_geometry(feat)
    assert res.is_flagged is True


def test_geometry_scoring_is_fail_closed():
    feat = {"available": False}
    res = pipeline.score_geometry(feat)
    assert res.is_flagged is None


# ---------------------------------------------------------------------------
# Guard terhadap regresi grayscale-brightness (yang dulu dipakai)
# ---------------------------------------------------------------------------

def test_sclera_feature_is_not_just_grayscale_brightness():
    """Feature CIELAB a* harus membedakan 'Abu terang' vs 'putih tapi pucat'.
    Versi lama memakai brightness grayscale, yang tidak bisa membedakan."""
    # pucat: rasio 0.93, DI ATAS ambang flag 0.80
    pallid = _sclera_features(sclera_a=13.0, skin_a=14.0)
    # sehat: rasio 0.43
    healthy = _sclera_features(sclera_a=6.0, skin_a=14.0)

    r_pallid = pipeline.score_sclera(pallid)
    r_healthy = pipeline.score_sclera(healthy)

    assert r_pallid.normalized > r_healthy.normalized
    assert r_pallid.is_flagged is True
    assert r_healthy.is_flagged is False

# ---------------------------------------------------------------------------
# Agregasi status
# ---------------------------------------------------------------------------

def test_status_is_tidak_valid_when_any_indicator_unavailable():
    """Fail-closed: satu indikator gagal -> overall tidak boleh 'normal'."""
    inds = [
        pipeline.IndicatorResult("rambut", 5.0, True, "", 1.0),
        pipeline.IndicatorResult("kepucatan_mata", None, None, "tidak ada mata"),
        pipeline.IndicatorResult("geometri_wajah", 0.5, False, "", 0.0),
    ]
    status, score = pipeline.combine_status(inds)
    assert status == "tidak_valid"
    assert score == 0.0


def test_all_normal_indicators_give_normal_status():
    inds = [
        pipeline.IndicatorResult("rambut", 50.0, False, "", 0.0),
        pipeline.IndicatorResult("kepucatan_mata", 0.2, False, "", 0.0),
        pipeline.IndicatorResult("geometri_wajah", 1.1, False, "", 0.0),
    ]
    status, score = pipeline.combine_status(inds)
    assert status == "normal"
    assert score == 0.0


def test_flagged_indicator_never_yields_normal_status():
    """Skor agregat kecil TIDAK boleh menutupi satu indikator yang FLAG.

    Contoh nyata: indikator geometri punya severity 0.32 (di bawah ambang
    'sedang' 0.45) tapi is_flagged=True karena rasionya di luar rentang. Kalau
    status hanya dilihat dari skor, hasilnya 'normal' padahal ada temuan
    positif -- operator punya alasan untuk mengabaikannya.
    """
    inds = [
        pipeline.IndicatorResult("rambut", 50.0, False, "", 0.0),
        pipeline.IndicatorResult("kepucatan_mata", 0.2, False, "", 0.0),
        pipeline.IndicatorResult("geometri_wajah", 0.83, True, "", 0.32),
    ]
    status, score = pipeline.combine_status(inds)
    assert score < config.STATUS_THRESHOLDS["sedang"], "fixture harus memang di bawah ambang"
    assert status != "normal", "ada indikator yang flag, status tidak boleh normal"


def test_high_score_escalates_to_tinggi():
    inds = [
        pipeline.IndicatorResult("rambut", 2.0, True, "", 1.0),
        pipeline.IndicatorResult("kepucatan_mata", 0.95, True, "", 1.0),
        pipeline.IndicatorResult("geometri_wajah", 0.7, True, "", 1.0),
    ]
    status, score = pipeline.combine_status(inds)
    assert status == "tinggi"
    assert score == 1.0
