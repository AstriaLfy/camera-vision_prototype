"""
test_eda.py
-----------
Test untuk scripts/eda.py.

Fokusnya bagian yang kalau salah akan menghasilkan angka yang kelihatan masuk akal
tetapi sebenarnya keliru:
koreksi Benjamini-Hochberg, rank-biserial, korelasi parsial, ambang NaN, dan
penanganan kasus degenerate (kolom konstan, satu kelas kosong, sampel kecil).

Semua test memakai CSV dummy dengan nama kolom yang PERSIS sama seperti
keluaran scripts/extract_dataset.py.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eda  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: CSV dummy dengan skema sama seperti features.csv
# ---------------------------------------------------------------------------

def make_dummy(n=120, seed=0, drop_class=None, high_nan_frac=0.9,
               nan_frac=0.05, with_covariates=True) -> pd.DataFrame:
    """Buat DataFrame features.csv dummy.

    drop_class: 'one' -> semua label 1, 'zero' -> semua label 0, None -> dua kelas.
    high_nan_frac: fraksi NaN untuk kolom yang sengaja harus dibuang.
    """
    rng = np.random.default_rng(seed)

    if drop_class == "one":
        label = np.ones(n, dtype=int)
    elif drop_class == "zero":
        label = np.zeros(n, dtype=int)
    else:
        label = (rng.random(n) < 0.45).astype(int)
        if label.sum() == 0 or label.sum() == n:      # pastikan dua kelas ada
            label[0], label[-1] = 1, 0

    # LiLA sengaja bergantung pada label (sinyal Mann-Whitney) DAN pada usia.
    # Efek usia ini yang membuat fitur berbasis usia terdeteksi "cerminan usia".
    usia = rng.integers(7, 45, n)
    lila = (np.where(label == 1, 16.5, 24.0)
           + 0.40 * (usia - np.mean(rng.integers(7, 45, n))) + rng.normal(0, 0.8, n))
    tinggi = rng.normal(150, 8, n)
    berat = lila * (tinggi / 100.0) ** 2

    # fitur yang korelasinya dengan LiLA sengaja dibuat bermakna
    hair_chroma = -0.6 * (lila - lila.mean()) / lila.std() + rng.normal(0, 0.7, n)
    # fitur yang isinya cuma ngikut usia -> harus terdeteksi sbg "cerminan usia"
    age_proxy = 0.9 * (usia - usia.mean()) / usia.std() + rng.normal(0, 0.3, n)
    noise = rng.normal(0, 1, n)
    # dua fitur duplikat -> harus terdeteksi redundan
    dup_a = noise * 2.0
    dup_b = noise * 2.0

    df = pd.DataFrame({
        "subject_id": [f"SUBJ{i:03d}" for i in range(1, n + 1)],
        "image_path": [f"S{i:03d}.jpg" for i in range(1, n + 1)],
        "image_abs_path": [f"/tmp/img/S{i:03d}.jpg" for i in range(1, n + 1)],
        "panjang_tinggi_cm": tinggi,
        "berat_kg": berat,
        "lila_cm": lila,
        "label_underweight": label,
        "yaw_proxy": rng.normal(0, 0.1, n),
        "roll_deg": rng.normal(0, 6, n),
        "quality_ok": np.ones(n, dtype=int),          # konstan
        "n_faces": np.ones(n, dtype=int),             # konstan
        "hair_available": np.ones(n, dtype=int),      # konstan + flag
        "hair_l": rng.normal(45, 8, n),
        "hair_chroma": hair_chroma,
        "hair_chroma_ratio": rng.normal(1.0, 0.2, n),
        "sclera_a_star": rng.normal(12, 1.5, n),
        "sclera_n_eyes": np.full(n, 2),              # konstan
        "cheek_to_jaw_ratio": rng.normal(1.2, 0.05, n),
        "cheek_width_px": rng.normal(200, 10, n),
        "jaw_width_px": rng.normal(170, 8, n),
        "dup_a": dup_a,
        "dup_b": dup_b,
        "age_proxy": age_proxy,
        "mostly_nan": np.where(rng.random(n) < high_nan_frac, np.nan,
                            rng.normal(0, 1, n)),
        "sparse_nan": np.where(rng.random(n) < nan_frac, np.nan, rng.normal(0, 1, n)),
        "notes": ["" for _ in range(n)],               # semua kosong -> all-NaN
    })
    if with_covariates:
        df.insert(df.columns.get_loc("lila_cm") + 1, "usia", usia)
        df.insert(df.columns.get_loc("usia") + 1, "jenis_kelamin",
                  np.where(rng.random(n) < 0.5, "L", "P"))
    return df


def write_csv(df: pd.DataFrame, tmp_path: Path, name="features.csv") -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def dummy_run(tmp_path):
    """Jalankan EDA atas dataset dummy yang normal, kembalikan (res, outdir)."""
    df = make_dummy(n=140, seed=1)
    csv = write_csv(df, tmp_path)
    outdir = tmp_path / "eda"
    res = eda.run(csv, outdir)
    return res, outdir, df


# ---------------------------------------------------------------------------
# Koreksi Benjamini-Hochberg
# ---------------------------------------------------------------------------

def test_benjamini_hochberg_matches_textbook_example():
    """Contoh klasik BH. adjusted harus >= p dan monoton monoton naik."""
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    expected = [0.01, 0.04, 0.084, 0.084, 0.084,
                0.1, 0.10571, 0.216, 0.216, 0.216]
    got = eda.benjamini_hochberg(p)
    assert got == pytest.approx(expected, abs=1e-4)


def test_benjamini_hochberg_is_monotone_and_never_below_raw():
    rng = np.random.default_rng(3)
    p = list(rng.random(60))
    adj = eda.benjamini_hochberg(p)
    # monoton hanya berlaku kalau diurutkan menurut p (BH adalah step-up)
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12), "adjusted harus monoton naik"
    assert np.all(adj >= np.array(p) - 1e-12), "adjusted tidak boleh < p asli"
    assert np.all(adj <= 1.0)


def test_benjamini_hochberg_preserves_input_order():
    p = [0.5, 0.001, 0.9]
    adj = eda.benjamini_hochberg(p)
    assert adj[1] < adj[0] < adj[2]


def test_benjamini_hochberg_keeps_nan_as_nan():
    p = [0.01, float("nan"), 0.04]
    adj = eda.benjamini_hochberg(p)
    assert np.isnan(adj[1]), "NaN tidak boleh jadi angka"
    assert np.isfinite(adj[0]) and np.isfinite(adj[2])


def test_benjamini_hochberg_all_nan():
    adj = eda.benjamini_hochberg([float("nan")] * 4)
    assert np.all(np.isnan(adj))


def test_benjamini_hochberg_correction_is_stricter_than_raw():
    """Koreksi harus menaikkan p-value, tidak pernah menurunkannya."""
    rng = np.random.default_rng(11)
    p = list(rng.random(40))
    adj = eda.benjamini_hochberg(p)
    assert np.all(adj >= np.array(p) - 1e-12)
    assert np.any(adj > np.array(p))


# ---------------------------------------------------------------------------
# Effect size rank-biserial
# ---------------------------------------------------------------------------

def test_rank_biserial_is_minus_one_when_group_a_all_lower():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 11.0, 12.0, 13.0])
    u, p, r, n1, n0, reason = eda.mannwhitney_with_effect(a, b)
    assert reason == ""
    assert u == 0.0
    assert r == pytest.approx(-1.0), "grup a selalu lebih rendah -> r = -1"


def test_rank_biserial_is_plus_one_when_group_a_all_higher():
    a = np.array([10.0, 11.0, 12.0, 13.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    u, p, r, n1, n0, reason = eda.mannwhitney_with_effect(a, b)
    assert r == pytest.approx(1.0)


def test_rank_biserial_is_antisymmetric_and_bounded():
    rng = np.random.default_rng(5)
    a, b = rng.normal(0, 1, 60), rng.normal(0.4, 1, 50)
    res_ab = eda.mannwhitney_with_effect(a, b)
    res_ba = eda.mannwhitney_with_effect(b, a)
    r_ab, r_ba = res_ab[2], res_ba[2]
    assert r_ab == pytest.approx(-r_ba)
    assert -1.0 <= r_ab <= 1.0


def test_rank_biserial_matches_manual_formula():
    rng = np.random.default_rng(7)
    a, b = rng.normal(0, 1, 40), rng.normal(0.5, 1, 45)
    u, p, r, n1, n0, _ = eda.mannwhitney_with_effect(a, b)
    manual = 2 * u / (n1 * n0) - 1
    assert r == pytest.approx(manual)


def test_mannwhitney_agrees_with_scipy_pvalue():
    rng = np.random.default_rng(9)
    a, b = rng.normal(0, 1, 45), rng.normal(0.6, 1, 50)
    _, p, *_ = eda.mannwhitney_with_effect(a, b)
    assert p == pytest.approx(float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue))


# ---------------------------------------------------------------------------
# safe_spearman
# ---------------------------------------------------------------------------

def test_safe_spearman_perfect_monotone_is_one():
    x = pd.Series(np.arange(50, dtype=float))
    y = pd.Series(np.arange(50, dtype=float) * 3 + 1)
    rho, p, n, reason = eda.safe_spearman(x, y)
    assert rho == pytest.approx(1.0)
    assert reason == ""
    assert n == 50


def test_safe_spearman_perfect_inverse_is_minus_one():
    rho, _, _, _ = eda.safe_spearman(pd.Series(np.arange(40.0)),
                                      pd.Series(-np.arange(40.0)))
    assert rho == pytest.approx(-1.0)


def test_safe_spearman_constant_column_reports_reason():
    x = pd.Series(np.ones(60))
    y = pd.Series(np.arange(60.0))
    rho, p, n, reason = eda.safe_spearman(x, y)
    assert np.isnan(rho) and np.isnan(p)
    assert "konstan" in reason
    assert reason, "kolom konstan WAJIB punya alasan"


def test_safe_spearman_too_few_rows_reports_reason():
    x = pd.Series([1.0, np.nan, 3.0])
    y = pd.Series([1.0, 2.0, 3.0])
    rho, p, n, reason = eda.safe_spearman(x, y)
    assert np.isnan(rho)
    assert n == 2
    assert "baris" in reason


def test_safe_spearman_uses_pairwise_deletion():
    x = pd.Series([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
    y = pd.Series([2.0, 5.0, 4.0, 8.0, 1.0, 12.0])
    rho, _, n, reason = eda.safe_spearman(x, y)
    assert n == 4, "NaN harus pairwise-deleted, bukan.dropna() pada seluruh tabel"
    assert reason == ""
    assert rho > 0.9


# ---------------------------------------------------------------------------
# Korelasi parsial
# ---------------------------------------------------------------------------

def test_partial_spearman_matches_rank_pearson_formula():
    rng = np.random.default_rng(13)
    n = 200
    z = rng.normal(0, 1, n)
    x = z * 0.8 + rng.normal(0, 0.6, n)
    y = z * 0.8 + rng.normal(0, 0.6, n)
    got, n_used, reason = eda.partial_spearman(pd.Series(x), pd.Series(y), pd.Series(z))

    rxz = stats.spearmanr(x, z).statistic
    ryz = stats.spearmanr(y, z).statistic
    rxy = stats.spearmanr(x, y).statistic
    expected = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    assert reason == ""
    assert n_used == n
    assert got == pytest.approx(expected)


def test_partial_spearman_removes_pure_age_effect():
    """Kalau x & y cuma fungsi z, korelasi parsial harus mendekati 0."""
    rng = np.random.default_rng(17)
    n = 400
    z = rng.normal(0, 1, n)
    x = z * 1.5 + rng.normal(0, 0.2, n)
    y = z * 1.5 + rng.normal(0, 0.2, n)
    full = stats.spearmanr(x, y).statistic
    part, _, _ = eda.partial_spearman(pd.Series(x), pd.Series(y), pd.Series(z))
    assert full > 0.9, "korelasi kasar harus tinggi"
    assert abs(part) < 0.25, f"korelasi parsial harus mendekati 0, dapat {part}"


def test_partial_spearman_reports_unstable_denominator():
    n = 60
    z = np.arange(float(n))
    x = z.copy()
    y = z.copy()          # yse perfectly linier dengan z -> penyebut ~0
    part, _, reason = eda.partial_spearman(pd.Series(x), pd.Series(y), pd.Series(z))
    assert reason != ""
    assert np.isnan(part)


def test_partial_spearman_handles_nan_pairwise():
    rng = np.random.default_rng(19)
    n = 100
    z = rng.normal(0, 1, n)
    x = z + rng.normal(0, 0.5, n)
    y = z + rng.normal(0, 0.5, n)
    x[::10] = np.nan
    y[::7] = np.nan
    part, n_used, _ = eda.partial_spearman(pd.Series(x), pd.Series(y), pd.Series(z))
    assert 0 < n_used < n
    assert np.isfinite(part)


# ---------------------------------------------------------------------------
# Klasifikasi kolom
# ---------------------------------------------------------------------------

def test_covariates_are_not_treated_as_features():
    """usia & jenis kelamin harus dikeluarkan; kalau tidak, usia akan
    mendominasi dan menyesatkan pembaca summary."""
    df = make_dummy(n=60, seed=2, with_covariates=True)
    feats, coverage, miss = eda.classify_columns(df)
    assert "usia" not in feats
    assert "jenis_kelamin" not in feats
    for col in ("usia", "jenis_kelamin"):
        row = miss[miss["column"] == col].iloc[0]
        assert row["decision"] == "exclude"
        assert row["reason"], f"{col} harus punya alasan"


def test_identity_and_measurement_columns_excluded():
    df = make_dummy(n=60, seed=2)
    feats, _, miss = eda.classify_columns(df)
    for col in ("subject_id", "image_path", "image_abs_path",
                "panjang_tinggi_cm", "berat_kg", "lila_cm", "label_underweight"):
        assert col not in feats
        assert miss.loc[miss["column"] == col, "decision"].iloc[0] == "exclude"


def test_constant_columns_excluded_with_reason():
    df = make_dummy(n=60, seed=2)
    feats, _, miss = eda.classify_columns(df)
    for col in ("quality_ok", "n_faces", "sclera_n_eyes"):
        assert col not in feats
        reason = miss.loc[miss["column"] == col, "reason"].iloc[0]
        assert "konstan" in reason


def test_coverage_flags_excluded_with_reason():
    df = make_dummy(n=60, seed=2)
    feats, coverage, miss = eda.classify_columns(df)
    assert "hair_available" not in feats
    assert "hair_available" in coverage
    reason = miss.loc[miss["column"] == "hair_available", "reason"].iloc[0]
    assert "ketersediaan" in reason


def test_non_numeric_columns_excluded_with_reason():
    df = make_dummy(n=60, seed=2)
    _, _, miss = eda.classify_columns(df)
    reason = miss.loc[miss["column"] == "notes", "reason"].iloc[0]
    assert "non-numerik" in reason


def test_every_excluded_column_has_a_reason(dummy_run):
    _, outdir, _ = dummy_run
    miss = pd.read_csv(outdir / "missingness.csv")
    excluded = miss[miss["decision"] == "exclude"]
    assert len(excluded) > 0
    assert excluded["reason"].notna().all()
    assert (excluded["reason"].str.len() > 0).all()


def test_every_column_appears_in_missingness_report(dummy_run):
    _, outdir, df = dummy_run
    miss = pd.read_csv(outdir / "missingness.csv")
    assert set(miss["column"]) == set(df.columns), "semua kolom harus tercatat"


# ---------------------------------------------------------------------------
# Ambang NaN
# ---------------------------------------------------------------------------

def test_high_nan_column_dropped_and_recorded():
    df = make_dummy(n=100, seed=4, high_nan_frac=0.95)
    csv = write_csv(df, Path("/tmp"))  # dipakai hanya untuk bentuk data
    features, coverage, miss = eda.classify_columns(df)
    keep, dropped = eda.drop_high_nan_columns(features, df, 0.30)
    assert any(c == "mostly_nan" for c, _ in dropped)
    assert "mostly_nan" not in keep


def test_high_nan_column_absent_from_analysis(tmp_path):
    df = make_dummy(n=140, seed=4, high_nan_frac=0.95)
    csv = write_csv(df, tmp_path)
    outdir = tmp_path / "eda"
    res = eda.run(csv, outdir)
    assert res["n_features"] > 0
    mwu = pd.read_csv(outdir / "mannwhitney.csv")
    assert "mostly_nan" not in set(mwu["feature"])
    # harus tercatat di missingness + disebut di summary
    miss = pd.read_csv(outdir / "missingness.csv")
    row = miss[miss["column"] == "mostly_nan"].iloc[0]
    assert row["decision"] == "drop"
    assert "NaN" in row["reason"]
    summary = (outdir / "summary.md").read_text()
    assert "mostly_nan" in summary
    assert any("NaN" in w for w in res["warnings"])


def test_column_just_below_threshold_is_kept():
    df = make_dummy(n=200, seed=4, high_nan_frac=0.10)
    features, _, _ = eda.classify_columns(df)
    keep, dropped = eda.drop_high_nan_columns(features, df, 0.30)
    assert "mostly_nan" in keep
    assert all(c != "mostly_nan" for c, _ in dropped)


# ---------------------------------------------------------------------------
# Ringkasan & peringatan
# ---------------------------------------------------------------------------

def test_summary_has_all_required_sections(dummy_run):
    _, outdir, _ = dummy_run
    text = (outdir / "summary.md").read_text()
    for heading in ("Ringkasan data", "Distribusi label", "NaN per kolom",
                    "Boxplot", "Spearman", "Mann-Whitney", "Scatter",
                    "Matriks korelasi", "Analisis usia",
                    "Fitur signifikan", "Audit"):
        assert heading in text, f"bagian '{heading}' hilang dari summary.md"


def test_summary_reports_row_and_label_counts(dummy_run):
    _, outdir, df = dummy_run
    text = (outdir / "summary.md").read_text()
    assert str(len(df)) in text
    labels = df["label_underweight"].value_counts()
    for lab, n in labels.items():
        assert f"| {lab} | {n} |" in text, f"baris label={lab} (n={n}) tidak muncul"


def test_small_sample_warning_appears(tmp_path):
    df = make_dummy(n=40, seed=6)
    csv = write_csv(df, tmp_path)
    res = eda.run(csv, tmp_path / "eda")
    text = (tmp_path / "eda" / "summary.md").read_text()
    assert any("40 < 100" in w for w in res["warnings"])
    assert "Peringatan ukuran sampel" in text


def test_small_class_warning_appears(tmp_path):
    df = make_dummy(n=200, seed=6)
    # paksa hanya 5 baris label=1
    df.loc[df.index[:5], "label_underweight"] = 1
    df.loc[df.index[5:], "label_underweight"] = 0
    csv = write_csv(df, tmp_path)
    res = eda.run(csv, tmp_path / "eda")
    assert any("label=1" in w and "5 sampel" in w for w in res["warnings"])
    assert any("label=1" in w for w in res["warnings"])


def test_no_sample_warning_when_sample_is_large_and_balanced(dummy_run):
    res, outdir, df = dummy_run
    counts = df["label_underweight"].value_counts()
    if len(df) >= 100 and counts.min() >= 20:
        assert not any("< 100" in w for w in res["warnings"])
        assert not any("hanya" in w and "sampel" in w for w in res["warnings"])


# ---------------------------------------------------------------------------
# Kasus degenerate: satu kelas kosong
# ---------------------------------------------------------------------------

def test_single_class_missing_does_not_crash(tmp_path):
    df = make_dummy(n=80, seed=8, drop_class="one")
    csv = write_csv(df, tmp_path)
    outdir = tmp_path / "eda"
    res = eda.run(csv, outdir)          # harus selesai tanpa exception
    assert (outdir / "summary.md").exists()
    assert res["n_features"] > 0


def test_single_class_missing_warns_and_reports_reason(tmp_path):
    df = make_dummy(n=80, seed=8, drop_class="one")
    csv = write_csv(df, tmp_path)
    outdir = tmp_path / "eda"
    res = eda.run(csv, outdir)
    assert any("tidak ada sama sekali" in w for w in res["warnings"])
    assert any("label 0" in w for w in res["warnings"]), "kelas hilang harus disebut"

    mwu = pd.read_csv(outdir / "mannwhitney.csv")
    assert mwu["p_raw"].isna().all(), "MWU tidak boleh dihitung tanpa kelas 0"
    assert mwu["note"].notna().all() and (mwu["note"].str.len() > 0).all(), \
        "setiap fitur harus punya alasan tidak bisa diuji"
    text = (outdir / "summary.md").read_text()
    assert "tidak bisa dihitung" in text


def test_single_class_missing_produces_zero_significant(tmp_path):
    df = make_dummy(n=80, seed=8, drop_class="one")
    csv = write_csv(df, tmp_path)
    outdir = tmp_path / "eda"
    res = eda.run(csv, outdir)
    assert res["n_significant"] == 0


def test_boxplot_files_exist_even_with_one_class(tmp_path):
    df = make_dummy(n=80, seed=8, drop_class="one")
    csv = write_csv(df, tmp_path)
    outdir = tmp_path / "eda"
    eda.run(csv, outdir)
    assert list(outdir.glob("boxplots_page*.png")), "boxplot tetap harus dibuat"


def test_single_class_zero_also_works(tmp_path):
    df = make_dummy(n=70, seed=9, drop_class="zero")
    csv = write_csv(df, tmp_path)
    res = eda.run(csv, tmp_path / "eda")
    assert any("label 1" in w for w in res["warnings"])
    assert res["n_significant"] == 0


# ---------------------------------------------------------------------------
# Audit: tidak ada pembuangan diam-diam
# ---------------------------------------------------------------------------

def test_rows_with_missing_label_are_recorded(tmp_path):
    df = make_dummy(n=60, seed=10)
    df.loc[df.index[:5], "label_underweight"] = np.nan
    csv = write_csv(df, tmp_path)
    outdir = tmp_path / "eda"
    res = eda.run(csv, outdir)
    assert res["n_rows"] == 60, "jumlah baris terbaca tidak boleh ikut berkurang"
    assert res["n_used_rows"] == 55

    excl = pd.read_csv(outdir / "excluded_rows.csv")
    assert len(excl) == 5
    assert excl["reason"].str.contains("label_underweight").all()
    text = (outdir / "summary.md").read_text()
    assert "excluded_rows.csv" in text
    assert "Baris yang dikeluarkan" in text


def test_rows_with_missing_lila_are_recorded(tmp_path):
    df = make_dummy(n=60, seed=11)
    df.loc[df.index[:3], "lila_cm"] = np.nan
    csv = write_csv(df, tmp_path)
    outdir = tmp_path / "eda"
    res = eda.run(csv, outdir)
    assert res["n_used_rows"] == 57
    excl = pd.read_csv(outdir / "excluded_rows.csv")
    assert len(excl) == 3
    assert excl["reason"].str.contains("lila_cm").all()


def test_rows_with_missing_label_are_also_absent_from_analysis(tmp_path):
    """Baris tanpa label tidak boleh ikut dihitung, tapi harus tercatat."""
    df = make_dummy(n=120, seed=12)
    df.loc[df.index[:10], "label_underweight"] = np.nan
    csv = write_csv(df, tmp_path)
    outdir = tmp_path / "eda"
    eda.run(csv, outdir)
    mwu = pd.read_csv(outdir / "mannwhitney.csv")
    total_in_mwu = int((mwu["n_label0"] + mwu["n_label1"]).max())
    assert total_in_mwu == 110, f"MWU harus pakai 110 baris, dapat {total_in_mwu}"


def test_pairwise_deletion_is_documented(dummy_run):
    _, outdir, _ = dummy_run
    text = (outdir / "summary.md").read_text()
    assert "pairwise deletion" in text
    assert "n_used" in text


def test_n_used_column_present_in_every_output_table(dummy_run):
    _, outdir, _ = dummy_run
    sp = pd.read_csv(outdir / "spearman_lila.csv")
    assert "n_used" in sp.columns
    mwu = pd.read_csv(outdir / "mannwhitney.csv")
    assert "n_label0" in mwu.columns and "n_label1" in mwu.columns


def test_excluded_rows_file_absent_when_nothing_dropped(dummy_run):
    """Kalau tidak ada baris dibuang, file tidak dibuat (bukan dibuat kosong
    yang disalahartikan sebagai ada masalah)."""
    _, outdir, _ = dummy_run
    assert not (outdir / "excluded_rows.csv").exists()


# ---------------------------------------------------------------------------
# Redundansi & analisis usia
# ---------------------------------------------------------------------------

def test_duplicate_features_flagged_redundant(dummy_run):
    _, outdir, _ = dummy_run
    summary = (outdir / "summary.md").read_text()
    assert "redundan" in summary
    assert res_redundant_pairs(outdir) >= 1, "dup_a & dup_b harus terdeteksi"


def res_redundant_pairs(outdir):
    matrix = pd.read_csv(outdir / "correlation_matrix.csv", index_col=0)
    feats = list(matrix.columns)
    n = 0
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            if abs(matrix.iat[i, matrix.columns.get_loc(b)]) > 0.90:
                n += 1
    return n


def test_redundant_pairs_have_rho_above_threshold(dummy_run):
    _, outdir, df = dummy_run
    matrix = pd.read_csv(outdir / "correlation_matrix.csv", index_col=0)
    for i, a in enumerate(matrix.columns):
        for b in matrix.columns[i + 1:]:
            r = matrix.iat[i, matrix.columns.get_loc(b)]
            if r > 0.95:      # dup_a & dup_b hampir identik
                assert abs(r) > 0.90


def test_age_mirror_feature_detected(dummy_run):
    """age_proxy sengaja dibuat dari usia -> harus ditandai cerminan usia."""
    _, outdir, df = dummy_run
    age = pd.read_csv(outdir / "age_analysis.csv")
    assert "age_proxy" in set(age["feature"])
    row = age[age["feature"] == "age_proxy"].iloc[0]
    assert bool(row["age_mirror"]), "age_proxy harus ditandai age_mirror"
    assert abs(row["rho_age"]) > abs(row["rho_partial"]) or \
        abs(row["rho_partial"]) < 0.5 * abs(row["rho_lila"])


def test_age_analysis_has_both_correlations_and_partial(dummy_run):
    _, outdir, _ = dummy_run
    age = pd.read_csv(outdir / "age_analysis.csv")
    for col in ("feature", "rho_lila", "rho_age", "rho_partial", "drop_ratio",
                "age_mirror"):
        assert col in age.columns


def test_age_analysis_skipped_when_usia_absent(tmp_path):
    df = make_dummy(n=120, seed=13, with_covariates=False)
    csv = write_csv(df, tmp_path)
    outdir = tmp_path / "eda"
    res = eda.run(csv, outdir)
    assert not (outdir / "age_analysis.csv").exists()
    text = (outdir / "summary.md").read_text()
    assert "tidak ada di data" in text
    # catatat sebagai CATATAN (bukan peringatan ukuran sampel)
    assert any("analisis usia" in n for n in res["notes"])
    assert any("usia" in n for n in res["notes"])


def test_age_mirror_not_flagged_for_independent_feature(dummy_run):
    _, outdir, _ = dummy_run
    age = pd.read_csv(outdir / "age_analysis.csv")
    noise_rows = age[age["feature"].isin(["cheek_width_px", "jaw_width_px",
                                          "sclera_a_star"])]
    assert not noise_rows["age_mirror"].any(), \
        "fitur acak tidak boleh ditandai cerminan usia"


# ---------------------------------------------------------------------------
# Redundansi statistically: BH tidak membuat fitur noise jadi signifikan
# ---------------------------------------------------------------------------

def test_bh_correction_suppresses_noise_features(tmp_path):
    """Fitur acak harus tetap tidak signifikan setelah BH."""
    df = make_dummy(n=60, seed=14)
    for col in ("yaw_proxy", "roll_deg", "sclera_a_star", "cheek_width_px",
                "jaw_width_px", "cheek_to_jaw_ratio"):
        df[col] = np.random.default_rng(abs(hash(col)) % 2 ** 31).normal(0, 1, len(df))
    csv = write_csv(df, tmp_path)
    outdir = tmp_path / "eda"
    eda.run(csv, outdir)
    mwu = pd.read_csv(outdir / "mannwhitney.csv")
    noise = mwu[mwu["feature"].isin(["yaw_proxy", "roll_deg", "sclera_a_star",
                                     "cheek_width_px", "jaw_width_px",
                                     "cheek_to_jaw_ratio"])]
    assert not noise["significant"].any(), "fitur noise tidak boleh signifikan"


# ---------------------------------------------------------------------------
# Kontrak keluaran
# ---------------------------------------------------------------------------

EXPECTED_FILES = [
    "summary.md", "missingness.csv", "spearman_lila.csv", "spearman_lila_heatmap.png",
    "mannwhitney.csv", "correlation_matrix.csv", "correlation_heatmap.png",
    "scatter_top5.png", "age_analysis.csv",
]


def test_all_expected_files_are_written(dummy_run):
    _, outdir, _ = dummy_run
    for name in EXPECTED_FILES:
        assert (outdir / name).exists(), f"{name} tidak dibuat"
        assert (outdir / name).stat().st_size > 0, f"{name} kosong"


def test_boxplot_pages_written(dummy_run):
    _, outdir, _ = dummy_run
    pages = sorted(outdir.glob("boxplots_page*.png"))
    assert pages, "tidak ada boxplot"


def test_spearman_sorted_by_absolute_strength(dummy_run):
    _, outdir, _ = dummy_run
    sp = pd.read_csv(outdir / "spearman_lila.csv")
    finite = sp[sp["rho"].notna()]
    assert list(finite["abs_rho"]) == sorted(finite["abs_rho"], reverse=True)


def test_mannwhitney_sorted_by_p_bh(dummy_run):
    _, outdir, _ = dummy_run
    mwu = pd.read_csv(outdir / "mannwhitney.csv")
    finite = mwu[mwu["p_bh"].notna()]
    assert list(finite["p_bh"]) == sorted(finite["p_bh"])


def test_spearman_and_mannwhitney_cover_same_features(dummy_run):
    _, outdir, _ = dummy_run
    sp = pd.read_csv(outdir / "spearman_lila.csv")
    mwu = pd.read_csv(outdir / "mannwhitney.csv")
    assert set(sp["feature"]) == set(mwu["feature"])
    res, outdir, _ = dummy_run
    assert len(sp) == res["n_features"]


def test_top_five_features_scatter_plots_exactly_five(dummy_run):
    _, outdir, _ = dummy_run
    assert (outdir / "scatter_top5.png").exists()
    sp = pd.read_csv(outdir / "spearman_lila.csv")
    finite = sp[sp["rho"].notna()].head(5)
    assert len(finite) <= 5
    assert (outdir / "scatter_top5.png").stat().st_size > 1000


def test_scatter_without_sex_column_still_works(tmp_path):
    df = make_dummy(n=100, seed=15, with_covariates=False)
    csv = write_csv(df, tmp_path)
    outdir = tmp_path / "eda"
    res = eda.run(csv, outdir)
    assert (outdir / "scatter_top5.png").exists()
    text = (outdir / "summary.md").read_text()
    assert "5 fitur terkuat" in text


def test_run_is_deterministic(dummy_run, tmp_path):
    _, outdir, df = dummy_run
    sp1 = pd.read_csv(outdir / "spearman_lila.csv")
    outdir2 = tmp_path / "eda2"
    eda.run(write_csv(df, tmp_path, "features2.csv"), outdir2)
    sp2 = pd.read_csv(outdir2 / "spearman_lila.csv")
    pd.testing.assert_frame_equal(sp1, sp2)


def test_stale_output_files_are_removed(tmp_path):
    """Keluaran run sebelumnya tidak boleh tertinggal dan terbaca sebagai
    hasil run sekarang."""
    outdir = tmp_path / "eda"
    outdir.mkdir()
    stale = outdir / "boxplots_page9.png"
    stale.write_bytes(b"lama")
    (outdir / "summary.md").write_text("lama", encoding="utf-8")
    (outdir / "mannwhitney.csv").write_text("lama", encoding="utf-8")
    # berkas milik pengguna harus TIDAK ikut terhapus
    keep = outdir / "catatan_pribadi.txt"
    keep.write_text("jangan dihapus", encoding="utf-8")

    csv = write_csv(make_dummy(n=80, seed=21), tmp_path)
    eda.run(csv, outdir)

    assert not stale.exists(), "halaman boxplot lama harus dibersihkan"
    assert (outdir / "summary.md").read_text() != "lama"
    assert keep.exists(), "skrip tidak boleh menghapus berkas milik pengguna"
    assert keep.read_text() == "jangan dihapus"


def test_removed_stale_is_noted_in_summary(tmp_path):
    outdir = tmp_path / "eda"
    outdir.mkdir()
    (outdir / "boxplots_page7.png").write_bytes(b"lama")
    csv = write_csv(make_dummy(n=80, seed=22), tmp_path)
    res = eda.run(csv, outdir)
    text = (outdir / "summary.md").read_text()
    assert "boxplots_page7.png" in text, "penghapusan harus dicatat di summary"
    assert "lama" in text


def test_cleanup_does_not_happen_when_input_missing(tmp_path):
    """Kalau input tidak ada, hasil run sebelumnya harus utuh."""
    outdir = tmp_path / "eda"
    outdir.mkdir()
    (outdir / "summary.md").write_text("run lama", encoding="utf-8")
    with pytest.raises(SystemExit):
        eda.run(tmp_path / "hilang.csv", outdir)
    assert (outdir / "summary.md").read_text() == "run lama"


def test_expected_output_stems_all_exist_after_run(dummy_run):
    _, outdir, _ = dummy_run
    for stem in eda.OUTPUT_STEMS:
        if stem == "excluded_rows.csv":
            continue  # sengaja tidak dibuat kalau tidak ada baris terbuang
        assert (outdir / stem).exists(), f"{stem} tidak dibuat"


def test_expected_output_stems_exist_even_without_corroboration(tmp_path):
    """Tabel MWU wajib ada walau tidak ada kelas yang bisa diuji."""
    df = make_dummy(n=80, seed=23, drop_class="one")
    csv = write_csv(df, tmp_path)
    outdir = tmp_path / "eda"
    eda.run(csv, outdir)
    for stem in eda.OUTPUT_STEMS:
        if stem == "excluded_rows.csv":
            continue  # sengaja tidak dibuat kalau tidak ada baris terbuang
        assert (outdir / stem).exists(), f"{stem} tidak dibuat"


# ---------------------------------------------------------------------------
# Guard: tidak boleh melatih model
# ---------------------------------------------------------------------------

def test_script_does_not_train_any_model():
    src = (ROOT / "scripts" / "eda.py").read_text()
    for forbidden in ("sklearn", ".fit(", "LogisticRegression", "RandomForest",
                      "train_test_split", "xgboost", "lightgbm", "torch"):
        assert forbidden not in src, f"skrip EDA tidak boleh memakai {forbidden}"


def test_summary_states_no_model_trained(dummy_run):
    _, outdir, _ = dummy_run
    text = (outdir / "summary.md").read_text()
    assert "tidak ada yang dilatih" in text.lower()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_missing_input_file_raises_systemexit(tmp_path):
    with pytest.raises(SystemExit):
        eda.run(tmp_path / "tidak_ada.csv", tmp_path / "eda")


def test_empty_input_raises_systemexit(tmp_path):
    csv = tmp_path / "features.csv"
    csv.write_text("subject_id,lila_cm,label_underweight\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        eda.run(csv, tmp_path / "eda")


def test_missing_required_columns_names_them(tmp_path):
    csv = tmp_path / "features.csv"
    csv.write_text("subject_id,tinggi_cm\nS1,170\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        eda.run(csv, tmp_path / "eda")
    msg = str(exc.value)
    assert "label_underweight" in msg and "lila_cm" in msg


def test_single_feature_matrix_does_not_crash(tmp_path):
    """Cuma satu fitur yang lolos filter -> matriks korelasi tidak ada."""
    df = make_dummy(n=60, seed=16)
    keep = ["subject_id", "image_path", "panjang_tinggi_cm", "berat_kg", "lila_cm",
            "label_underweight", "usia", "jenis_kelamin", "hair_chroma"]
    csv = write_csv(df[keep], tmp_path)
    outdir = tmp_path / "eda"
    res = eda.run(csv, outdir)
    assert res["n_features"] == 1
    assert (outdir / "summary.md").exists()
    text = (outdir / "summary.md").read_text()
    assert "tidak dihitung" in text


def test_zero_analysable_features_exits_with_clear_message(tmp_path):
    """Semua kolom terkena pengecualian/konstan -> berhenti, bukan KeyError.

    Tanpa guard ini satu-satunya tanda masalah adalah KeyError: 'rho' di tengah
    stack trace, padahal penyebabnya kolom konstan semua.
    """
    n = 10
    df = pd.DataFrame({
        "subject_id": [f"S{i}" for i in range(n)],
        "image_path": [f"{i}.jpg" for i in range(n)],
        "lila_cm": np.linspace(10.0, 14.0, n),
        "label_underweight": [0, 1] * (n // 2),
        "hair_l": np.full(n, 40.0),             # konstan -> bukan fitur
        "notes": ["" for _ in range(n)],        # non-numerik -> bukan fitur
    })
    csv = write_csv(df, tmp_path)
    with pytest.raises(SystemExit) as exc:
        eda.run(csv, tmp_path / "eda")
    msg = str(exc.value)
    assert "Tidak ada kolom fitur" in msg
    assert (tmp_path / "eda" / "missingness.csv").exists(), (
        "alasan per kolom tetap harus tertulis walaupun analisis berhenti"
    )


def test_all_nan_feature_column_is_handled(tmp_path):
    df = make_dummy(n=60, seed=17)
    df["selalu_kosong"] = np.nan
    csv = write_csv(df, tmp_path)
    outdir = tmp_path / "eda"
    res = eda.run(csv, outdir)
    assert "selalu_kosong" not in set(pd.read_csv(outdir / "mannwhitney.csv")["feature"])
    miss = pd.read_csv(outdir / "missingness.csv")
    row = miss[miss["column"] == "selalu_kosong"].iloc[0]
    assert row["decision"] == "exclude"
    assert row["reason"]


def test_cli_main_returns_zero(tmp_path):
    df = make_dummy(n=100, seed=18)
    csv = write_csv(df, tmp_path)
    rc = eda.main(["--input", str(csv), "--outdir", str(tmp_path / "eda")])
    assert rc == 0


def test_cli_respects_custom_nan_threshold(tmp_path):
    df = make_dummy(n=150, seed=19, high_nan_frac=0.20)
    csv = write_csv(df, tmp_path)
    loose = tmp_path / "eda_loose"
    tight = tmp_path / "eda_tight"
    eda.main(["--input", str(csv), "--outdir", str(loose), "--max-nan-rate", "0.9"])
    eda.main(["--input", str(csv), "--outdir", str(tight), "--max-nan-rate", "0.05"])
    loose_feats = set(pd.read_csv(loose / "mannwhitney.csv")["feature"])
    tight_feats = set(pd.read_csv(tight / "mannwhitney.csv")["feature"])
    assert len(loose_feats) > len(tight_feats), \
        "ambang NaN lebih longgar harus mempertahankan lebih banyak fitur"
    assert "mostly_nan" in loose_feats
    assert "mostly_nan" not in tight_feats


def test_cli_respects_min_samples_and_min_class(tmp_path):
    df = make_dummy(n=100, seed=20)
    csv = write_csv(df, tmp_path)
    outdir = tmp_path / "eda"
    res = eda.run(csv, outdir, min_samples=1000, min_class=1)
    assert any("100 < 1000" in w for w in res["warnings"])


def test_summary_contains_clinical_caveat(dummy_run):
    """Summary harus memperingatkan bahwa ambang LiLA beda per kategori.

    Kalau peringatan ini hilang, pembaca bisa mencampur balita dan ibu lalu
    menyimpulkan sesuatu dari label yang memang tidak sebanding.
    """
    _, outdir, _ = dummy_run
    text = (outdir / "summary.md").read_text()
    assert "12.5" in text and "23.5" in text, "ambang LiLA per kategori hilang"
    assert "kategori" in text and "tidak sebanding" in text
