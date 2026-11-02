#!/usr/bin/env python
"""
eda.py
------
Analisis eksplorasi atas data/features.csv (hasil scripts/extract_dataset.py).

Skrip ini HANYA mendeskripsikan data. Tidak ada model yang dilatih, tidak ada
prediksi, tidak ada split train/test. Tujuannya mencari fitur yang
memang punya sinyal, fitur yang dobel, dan fitur yang sinyalnya cuma ngikut
usia.

Yang dilakukan:
  1. Ringkasan data: jumlah baris, distribusi label, NaN per kolom.
     Kolom dengan NaN > --max-nan-rate ditandai dan dikeluarkan dari analisis.
  2. Boxplot tiap fitur numerik, dikelompokkan menurut label.
  3. Korelasi Spearman tiap fitur dengan LiLA (tabel + heatmap, terurut).
  4. Mann-Whitney U per fitur antara label 0 vs 1 + koreksi Benjamini-Hochberg
     + effect size rank-biserial.
  5. Scatter 5 fitur terkuat vs LiLA, diwarnai jenis_kelamin bila tersedia.
  6. Matriks korelasi antar fitur; |rho| > --redundant-threshold = redundan.
  7. Analisis usia: korelasi fitur-usia + korelasi parsial fitur-LiLA yang
     dikontrol usia.
  8. reports/eda/summary.md berisi semuanya + catatan pembuangan.

Aturan yang dijaga:
  - Tidak ada baris/kolom yang dibuang diam-diam. Setiap kolom yang
    dikeluarkan dicatat di missingness.csv dengan alasannya; setiap baris yang
    tidak bisa dipakai dicatat di excluded_rows.csv.
  - Penghitungan memakai pairwise deletion, jadi jumlah baris yang benar-benar
    dipakai per fitur dicatat sebagai kolom `n_used` di setiap tabel.
  - Tidak ada model yang dilatih di sini.

Contoh:
    python scripts/eda.py
    python scripts/eda.py --input data/features.csv --outdir reports/eda/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: harus sebelum import pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Kontrak kolom features.csv (dari scripts/extract_dataset.py)
# ---------------------------------------------------------------------------

ID_COLUMNS = ["subject_id", "image_path", "image_abs_path"]
MEASURE_COLUMNS = ["berat_kg", "lila_cm", "panjang_tinggi_cm"]
TARGET_COLUMN = "label_underweight"
LILA_COLUMN = "lila_cm"
AGE_COLUMN = "usia"
# usia_bulan adalah kovariat yang sama dengan usia (untuk balita); kalau ikut
# dianalisis sebagai fitur ia akan meniru usia dan menggandakan sinyalnya.
AGE_COLUMNS = (AGE_COLUMN, "usia_bulan")
SEX_COLUMN = "jenis_kelamin"

# Flag "ketersediaan fitur" bukan fitur: isinya menandai apakah fitur itu bisa
# dihitung, bukan mengukur apa pun. Korelasi flag ini dengan LiLA tidak punya
# makna, jadi dikeluarkan dengan alasannya (bukan dibuang diam-diam).
COVERAGE_SUFFIXES = ("_available",)


# ---------------------------------------------------------------------------
# Util statistik
# ---------------------------------------------------------------------------

MIN_N_FOR_RANK = 3   # di bawah ini korelasi rank tidak bermakna
MIN_N_PER_CLASS = 2  # 1 anggota per kelas -> varians tidak terdefinisi


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int, str]:
    """Spearman (rank) korelasi. Kembalikan (rho, p, n_used, reason).

    reason == "" kalau normal. reason terisi kalau tidak bisa dihitung, dan
    pemanggil WAJIB menuliskannya ke laporan.
    """
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < MIN_N_FOR_RANK:
        return float("nan"), float("nan"), n, f"hanya {n} baris berpasangan (min {MIN_N_FOR_RANK})"

    xs, ys = x[mask], y[mask]
    if xs.nunique() < 2:
        return float("nan"), float("nan"), n, "kolom konstan (tanpa variasi)"
    if ys.nunique() < 2:
        return float("nan"), float("nan"), n, "kolom target konstan"

    res = stats.spearmanr(xs, ys)
    return float(res.statistic), float(res.pvalue), n, ""


def mannwhitney_with_effect(a: np.ndarray, b: np.ndarray) -> tuple:
    """Mann-Whitney U dua arah + rank-biserial correlation.

    rank-biserial r = 2*U1/(n1*n0) - 1, dengan U1 = U untuk grup `a`.
    r > 0 berarti grup `a` cenderung punya nilai lebih tinggi.
    """
    n1, n0 = len(a), len(b)
    if n1 < MIN_N_PER_CLASS or n0 < MIN_N_PER_CLASS:
        return None, None, None, n1, n0, (
            f"kelas terlalu kecil ({n1} vs {n0}), min {MIN_N_PER_CLASS} per kelas"
        )
    res = stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
    u1 = float(res.statistic)
    p = float(res.pvalue)
    r_rb = 2.0 * u1 / (n1 * n0) - 1.0
    return u1, p, r_rb, n1, n0, ""


def benjamini_hochberg(pvalues: list) -> np.ndarray:
    """Koreksi BH FDR (step-up). NaN diteruskan apa adanya.

    Implementasi eksplisit supaya tidak bergantung statsmodels, dan supaya
    bisa diuji langsung terhadap nilai yang diketahui benar.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out

    pv = p[valid]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]

    # adjusted = min over k>=i of (p_k * m / k), dari bawah ke atas
    scaled = ranked * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    restored = np.empty(m)
    restored[order] = adjusted_sorted
    out[valid] = restored
    return out


def fmt_label(value) -> object:
    """Label came dari CSV jadi bisa jadi 0.0/1.0. Tampilkan sebagai 0/1."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    return int(f) if f.is_integer() else f


def partial_corr_pvalue(r: float, n: int) -> float:
    """p-value korelasi parsial via transformasi Fisher z (n-3 derajat bebas)."""
    if not np.isfinite(r) or n <= 3 or abs(r) >= 1.0:
        return float("nan")
    t = r * np.sqrt(n - 3) / np.sqrt(1.0 - r ** 2)
    return float(2.0 * stats.norm.sf(abs(t)))


def partial_spearman(x: pd.Series, y: pd.Series, z: pd.Series) -> tuple[float, int, str]:
    """Korelasi parsial x~y yang dikontrol z, dihitung pada rank (Spearman).

    r_xy.z = (r_xy - r_xz*r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))
    """
    mask = x.notna() & y.notna() & z.notna()
    n = int(mask.sum())
    if n < MIN_N_FOR_RANK:
        return float("nan"), n, f"hanya {n} baris berpasangan"

    xs, ys, zs = x[mask], y[mask], z[mask]
    if xs.nunique() < 2 or ys.nunique() < 2 or zs.nunique() < 2:
        return float("nan"), n, "salah satu kolom konstan"

    rxz = float(stats.spearmanr(xs, zs).statistic)
    ryz = float(stats.spearmanr(ys, zs).statistic)
    rxy = float(stats.spearmanr(xs, ys).statistic)

    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    if not np.isfinite(denom) or abs(denom) < 1e-8:
        return float("nan"), n, "penyebut tidak stabil (z hampir perfectly explains)"
    return (rxy - rxz * ryz) / denom, n, ""


# ---------------------------------------------------------------------------
# Ringkasan data & pemilihan kolom
# ---------------------------------------------------------------------------

def classify_columns(df: pd.DataFrame) -> tuple[list, list, pd.DataFrame]:
    """Bagi kolom jadi (analysable, coverage_flag, missingness_report).

    Kolom dibuang SELALU punya alasan recorded di missingness.csv.
    """
    rows = []

    def record(col, decision, reason, kind, n_nan, nan_rate, nunique):
        rows.append({
            "column": col, "dtype": str(kind), "n_nan": int(n_nan),
            "nan_rate": round(nan_rate, 4), "n_unique": int(nunique),
            "decision": decision, "reason": reason,
        })

    analysable, coverage = [], []

    for col in df.columns:
        s = df[col]
        n_nan = int(s.isna().sum())
        nan_rate = n_nan / len(df) if len(df) else 1.0
        nunique = int(s.nunique(dropna=True))
        kind = s.dtype

        # kolom identitas & pengukuran: bukan fitur, dipakai sebagai konteks
        if col in ID_COLUMNS:
            record(col, "exclude", "kolom identitas, bukan fitur", kind, n_nan, nan_rate, nunique)
            continue
        if col in MEASURE_COLUMNS:
            record(col, "exclude", "kolom pengukuran/label, bukan fitur", kind, n_nan, nan_rate, nunique)
            continue
        if col == TARGET_COLUMN:
            record(col, "exclude", "kolom label (variabel dependent)", kind, n_nan, nan_rate, nunique)
            continue
        # usia & jenis kelamin adalah kovariat, bukan fitur. Kalau ikut dianalisis
        # sebagai fitur, mereka akan mendominasi (umur memang berkorelasi kuat
        # dengan LiLA pada dataset anak) dan menyesatkan pembaca summary.
        if col in AGE_COLUMNS:
            record(col, "exclude",
                   "kovariat (usia) - dipakai sebagai kontrol korelasi parsial, bukan fitur",
                   kind, n_nan, nan_rate, nunique)
            continue
        if col == SEX_COLUMN:
            record(col, "exclude",
                   "kovariat (jenis kelamin) - dipakai sebagai warna scatter, bukan fitur",
                   kind, n_nan, nan_rate, nunique)
            continue

        if col.endswith(COVERAGE_SUFFIXES):
            record(col, "exclude",
                   f"flag ketersediaan fitur ('{col}'), bukan fitur terukur",
                   kind, n_nan, nan_rate, nunique)
            coverage.append(col)
            continue

        if not pd.api.types.is_numeric_dtype(s):
            record(col, "exclude", f"kolom non-numerik (dtype {kind})", kind, n_nan, nan_rate, nunique)
            continue

        if n_nan == len(df) and len(df) > 0:
            record(col, "exclude", "seluruhnya kosong (NaN semua baris)", kind, n_nan, nan_rate, nunique)
            continue

        if nunique <= 1:
            record(col, "exclude",
                   f"kolom konstan ({nunique} nilai unik) -> korelasi tidak terdefinisi",
                   kind, n_nan, nan_rate, nunique)
            continue

        if nan_rate > 1.0:  # tidak akan terjadi; jaga agar aman
            record(col, "exclude", "nan_rate tidak valid", kind, n_nan, nan_rate, nunique)
            continue

        analysable.append(col)
        record(col, "keep", "", kind, n_nan, nan_rate, nunique)

    return analysable, coverage, pd.DataFrame(rows)


def drop_high_nan_columns(features: list, df: pd.DataFrame, max_rate: float):
    """Kembalikan (dipakai, dibuang) berdasarkan ambang NaN."""
    keep, dropped = [], []
    for col in features:
        rate = float(df[col].isna().mean())
        if rate > max_rate:
            dropped.append((col, rate))
        else:
            keep.append(col)
    return keep, dropped


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_boxplots(df, features, label_col, outdir, per_page=12, ncols=3):
    """Grid boxplot fitur per label. Satu file PNG per halaman."""
    paths = []
    if not features:
        return paths
    nrows = int(np.ceil(per_page / ncols))
    pages = int(np.ceil(len(features) / per_page))
    for page in range(pages):
        chunk = features[page * per_page:(page + 1) * per_page]
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, col in zip(axes, chunk):
            ax.axis("on")
            # Kelas yang kosong tetap dik_plot sebagai kotak kosong dengan
            # label "n=0", supaya kelihatannya kelas itu memang tidak ada
            # (bukan sekadar hilang tanpa jejak).
            classes = sorted(df[label_col].dropna().unique())
            groups, tick_labels = [], []
            for lab in classes:
                vals = df.loc[df[label_col] == lab, col].dropna().to_numpy()
                groups.append(vals)
                tick_labels.append(f"label={lab}\nn={len(vals)}")
            if not groups:
                ax.text(0.5, 0.5, "tidak ada kelas label", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
                ax.set_title(col, fontsize=9)
                continue
            nonempty = [g for g in groups if len(g) > 0]
            if not nonempty:
                ax.text(0.5, 0.5, "semua kelas kosong\n(kolom NaN total)",
                        ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.set_title(col, fontsize=9)
                continue
            # matplotlib 3.9+ memakai tick_labels; 3.11 menolak `labels`.
            bp = ax.boxplot(nonempty, tick_labels=tick_labels,
                            showfliers=True, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("#9ecae1")
            ax.set_title(f"{col}\nn={len(df[col].dropna())}", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"Boxplot fitur per label (halaman {page + 1}/{pages})", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        p = outdir / f"boxplots_page{page + 1}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_heatmap(values: np.ndarray, row_labels, col_labels, title, outpath,
                 vmin=-1, vmax=1, figsize=None):
    """Heatmap sederhana dengan annotate + colorbar."""
    if figsize is None:
        figsize = (max(5.0, 0.42 * len(col_labels) + 3), max(3.0, 0.32 * len(row_labels) + 2))
    fig, ax = plt.subplots(figsize=figsize)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title, fontsize=10)
    if values.size <= 400:
        thresh = np.nanmax(np.abs(values)) / 2 if np.isfinite(values).any() else 0
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(v) > thresh else "black")
    fig.colorbar(im, ax=ax, shrink=0.6, label="rho (Spearman)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def plot_scatter_top(df, feats, lila_col, sex_col, outpath):
    """Scatter fitur vs LiLA, diwarnai jenis_kelamin bila ada."""
    n = len(feats)
    ncols = 2 if n <= 4 else 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    sexes = sorted(df[sex_col].dropna().unique()) if sex_col else []
    colors = {s: c for s, c in zip(sexes, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"])}

    for ax, feat in zip(axes, feats):
        ax.axis("on")
        if sex_col:
            for s in sexes:
                sub = df[df[sex_col] == s]
                ax.scatter(sub[feat], sub[lila_col], s=18, alpha=0.75,
                           label=str(s), color=colors[s], edgecolors="none")
        else:
            ax.scatter(df[feat], df[lila_col], s=18, alpha=0.75,
                       color="#1f77b4", edgecolors="none")
        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel(lila_col, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax.set_title(f"{feat} vs {lila_col}", fontsize=9)
    if sex_col and axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=7, title=sex_col, title_fontsize=7)
    fig.suptitle("5 fitur terkuat vs LiLA" + (f" (warna = {sex_col})" if sex_col else ""),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def plot_correlation_matrix(df, feats, outpath):
    mat = df[feats].corr(method="spearman")
    values = mat.to_numpy(dtype=float)
    figsize = (max(6.0, 0.45 * len(feats) + 3), max(5.0, 0.42 * len(feats) + 2))
    fig, ax = plt.subplots(figsize=figsize)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=90, fontsize=7)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=7)
    ax.set_title("Matriks korelasi Spearman antar fitur", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.7, label="rho")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return mat


# ---------------------------------------------------------------------------
# Laporan
# ---------------------------------------------------------------------------

def _fmt_cell(value) -> str:
    """Format nilai untuk tabel markdown: float dibulatkan, p-value kecil jadi
    notasi ilmiah, NaN jadi '-'. Tanpa ini summary.md penuh angka 16 digit."""
    if value is None:
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return "ya" if value else "tidak"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "-"
        if value != 0 and (abs(value) < 1e-4 or abs(value) >= 1e5):
            return f"{value:.3e}"
        return f"{value:.4g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.4g}"
    s = str(value)
    if s in ("nan", "NaN", "<NA>", "None"):
        return "-"
    return s.replace("\n", " ").replace("|", "\\|")


def fmt_table(rows: list, cols: list, headers: list) -> str:
    if not rows:
        return "_Tidak ada data._\n"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_fmt_cell(r.get(c)) for c in cols) + " |")
    return "\n".join(out) + "\n"


def write_summary(path, ctx):
    L = []
    add = L.append
    df = ctx["df"]

    add("# Ringkasan EDA — features.csv\n")
    add(f"- Sumber data: `{ctx['input']}`")
    add(f"- Folder keluaran: `{ctx['outdir']}`")
    add(f"- Jumlah baris dibaca: **{ctx['n_rows']}**")
    add(f"- Jumlah kolom dibaca: **{ctx['n_cols']}**")
    add(f"- Fitur dianalisis: **{len(ctx['features'])}**")
    add("- Model: **tidak ada yang dilatih**. Skrip ini hanya deskriptif.\n")

    # ---------------- peringatan ukuran sampel ----------------
    add("## 0. Peringatan ukuran sampel\n")
    warns = ctx["warnings"]
    if warns:
        for w in warns:
            add(f"- {w}")
    else:
        add("- Tidak ada peringatan: jumlah sampel dan kedua kelas memenuhi "
            "minimum yang disyaratkan.")
    add("")
    if ctx.get("removed_stale"):
        add(f"Catatan: {len(ctx['removed_stale'])} berkas keluaran lama dibersihkan "
            "sebelum analisis ulang agar tidak tertukar dengan hasil sekarang "
            f"({', '.join('`' + x + '`' for x in ctx['removed_stale'])}).\n")
    if ctx["notes"]:
        add("### Catatan pembuangan kolom")
        for nt in ctx["notes"]:
            add(f"- {nt}")
        add("")

    # ---------------- 1. ringkasan data ----------------
    add("## 1. Ringkasan data\n")
    add(f"- Baris total: **{ctx['n_rows']}**")
    add(f"- Baris dipakai untuk analisis: **{ctx['n_used_rows']}** "
        f"(baris tanpa label atau tanpa LiLA dikeluarkan, dicatat di `excluded_rows.csv`)")
    add(f"- Rata-rata LiLA: **{ctx['lila_mean']:.2f}** (min {ctx['lila_min']:.2f}, "
        f"max {ctx['lila_max']:.2f})")
    if ctx.get("age_summary"):
        add(f"- Rentang usia: **{ctx['age_summary']}**")
    if ctx.get("sex_summary"):
        add(f"- Jenis kelamin: {ctx['sex_summary']}")
    add("")
    add("### Distribusi label\n")
    add(fmt_table(ctx["label_rows"], ["label", "n", "persen"], ["label", "jumlah", "persen"]))

    add("### NaN per kolom fitur\n")
    add(f"Kolom dengan NaN > {ctx['max_nan_rate']:.0%} **dikeluarkan dari "
        "analisis berikutnya** dan ditandai di tabel ini.\n")
    add(fmt_table(ctx["nan_rows"], ["column", "n_nan", "nan_rate", "decision", "reason"],
                  ["kolom", "jumlah NaN", "NaN rate", "keputusan", "alasan"]))

    # ---------------- 2. boxplot ----------------
    add("## 2. Boxplot fitur per label\n")
    add("Satu file per halaman:\n")
    for p in ctx["boxplot_files"]:
        add(f"- `{p.name}`")
    add("")
    add("Semua baris dipakai per kolom lewat pairwise deletion; jumlah Non-NaN "
        "setiap fitur tertera di judul plot dan di `missingness.csv`.\n")

    # ---------------- 3. spearman vs lila ----------------
    add("## 3. Korelasi Spearman fitur vs LiLA\n")
    add("Diurutkan dari korelasi terkuat (|rho| terbesar). Tabel lengkap: "
        "`spearman_lila.csv`, visual: `spearman_lila_heatmap.png`.\n")
    add(fmt_table(ctx["spearman_rows"][:20],
                  ["feature", "rho", "p_value", "n_used", "note"],
                  ["fitur", "rho", "p-value", "n dipakai", "catatan"]))

    # ---------------- 4. mann-whitney ----------------
    add("## 4. Mann-Whitney U: label 0 vs 1\n")
    add(f"Uji dua arah dengan koreksi Benjamini-Hochberg pada **{ctx['n_tests']}** uji. "
        f"Effect size = rank-biserial correlation "
        f"({ctx['effect_label']}).\n")
    n_sig = ctx["n_significant"]
    add(f"**{n_sig} fitur signifikan setelah koreksi BH** "
        f"(p_BH < {ctx['alpha']}).\n")
    add(fmt_table(ctx["mwu_rows"],
                  ["feature", "u_stat", "p_raw", "p_bh", "rank_biserial",
                   "n_label1", "n_label0", "significant", "note"],
                  ["fitur", "U", "p asli", "p BH", "rank-biserial",
                   "n label=1", "n label=0", "signif", "catatan"]))

    # ---------------- 5. scatter ----------------
    add("## 5. Scatter 5 fitur terkuat vs LiLA\n")
    if ctx["top_scatter"]:
        add("Top-5 berdasarkan |rho| terhadap LiLA, diwarnai per "
            f"`{SEX_COLUMN}` bila kolomnya tersedia.\n")
        add("- `scatter_top5.png`")
        add("")
        add("Fitur yang diplot: " + ", ".join(f"`{c}`" for c in ctx["top_scatter"]) + "\n")
    else:
        add("_Tidak ada fitur yang bisa diplot._\n")

    # ---------------- 6. korelasi antar fitur ----------------
    add("## 6. Matriks korelasi antar fitur\n")
    add(f"Pasangan dengan |rho| > {ctx['redundant_threshold']} ditandai **redundan** "
        "(satu pasangan tidak menambah informasi dari yang lain).\n")
    add("- `correlation_matrix.csv`, `correlation_heatmap.png`\n")
    if ctx["redundant_rows"]:
        add("### Pasangan redundan\n")
        add(fmt_table(ctx["redundant_rows"],
                      ["feature_a", "feature_b", "rho", "abs_rho", "note"],
                      ["fitur A", "fitur B", "rho", "abs rho", "catatan"]))
    else:
        add(f"_Tidak ada pasangan dengan |rho| > {ctx['redundant_threshold']}._\n")

    # ---------------- 7. analisis usia ----------------
    add("## 7. Analisis usia\n")
    if not ctx["age_available"]:
        add(f"_Kolom `{AGE_COLUMN}` tidak ada di data, sehingga korelasi parsial "
            "tidak dihitung._\n")
    else:
        add(f"Kolom `{AGE_COLUMN}` tersedia. Korelasi parsial = korelasi fitur-LiLA "
            f"setelah efek `{AGE_COLUMN}` dibuang (dirank), memakai "
            "r_xy.z = (r_xy - r_xz*r_yz)/sqrt((1-r_xz^2)(1-r_yz^2)).\n")
        add(f"**Fitur yang korelasinya dengan LiLA kemungkinan hanya cerminan "
          f"usia**: {ctx['age_mirror_count']} fitur.\n")
        if ctx["age_mirror_rows"]:
            add(fmt_table(ctx["age_mirror_rows"],
                          ["feature", "rho_lila", "rho_age", "rho_partial",
                           "drop_ratio", "rho_lila_partial_p", "note"],
                          ["fitur", "rho LiLA", "rho usia", "rho parsial",
                           "penurunan", "p parsial", "catatan"]))
        else:
            add("_Tidak ada fitur yang korelasinya dengan LiLA hilang begitu saja "
                "begitu usia dikendalikan._\n")
        add("Tabel lengkap: `age_analysis.csv`. Top-10 fitur:|rho| vs usia:\n")
        add(fmt_table(ctx["age_top_rows"],
                      ["feature", "rho_lila", "rho_age", "rho_partial", "note"],
                      ["fitur", "rho LiLA", "rho usia", "rho parsial", "catatan"]))

    # ---------------- 8. feature significance ----------------
    add("## 8. Fitur signifikan (urutan kekuatan)\n")
    if ctx["sig_rows"]:
        add(fmt_table(ctx["sig_rows"],
                      ["rank", "feature", "rho_lila", "p_bh", "rank_biserial",
                       "rho_age", "rho_partial", "note"],
                      ["#", "fitur", "rho LiLA", "p BH", "rank-biserial",
                       "rho usia", "rho parsial", "catatan"]))
    else:
        add("_Tidak ada fitur yang bertahan setelah koreksi BH._\n")

    # ---------------- audit pembuangan ----------------
    add("## Audit: apa yang dibuang dan kenapa\n")
    add("Tidak ada baris atau kolom yang dibuang tanpa catatan.\n")
    add("### Kolom yang dikeluarkan\n")
    excluded = ctx["missingness"][ctx["missingness"]["decision"] == "exclude"]
    add(fmt_table(excluded.to_dict("records"), ["column", "decision", "reason"],
                  ["kolom", "keputusan", "alasan"]))
    add("### Baris yang dikeluarkan\n")
    if ctx["excluded_rows"].empty:
        add("_Tidak ada baris yang dikeluarkan._\n")
    else:
        add(f"**{len(ctx['excluded_rows'])}** baris tidak dipakai, "
            "tercantum di `excluded_rows.csv`:\n")
        add(fmt_table(ctx["excluded_rows"].head(50).to_dict("records"),
                      ["subject_id", "reason"], ["subject_id", "alasan"]))

    add("### Baris yang tidak seragam per fitur\n")
    add("Penghitungan korelasi/uji memakai pairwise deletion: untuk tiap fitur, "
        "hanya baris dengan nilai non-NaN pada fitur DAN target yang dipakai. "
        "Jumlahnya dicatat sebagai kolom `n_used` di setiap tabel.\n")
    if ctx["pairwise_rows"]:
        add(fmt_table(ctx["pairwise_rows"],
                      ["feature", "n_total", "n_used", "n_dropped", "note"],
                      ["fitur", "n total", "n dipakai", "n dibuang", "catatan"]))
    else:
        add("_Semua fitur lengkap pada semua baris yang dipakai._\n")

    add("\n---\n")
    add("**Catatan penting.** Angka korelasi dan p-value di dokumen ini "
        "menggambarkan dataset ini saja, bukan bukti bahwa fitur tersebut "
        "berkorelasi kausal dengan status gizi. Label `label_underweight` "
        "ditetapkan dari LiLA dengan ambang yang BERBEDA per kategori "
        "(balita < 12.5 cm; ibu hamil/menyusui < 23.5 cm), jadi mencampur "
        "balita dan ibu dalam satu analisis membuat labelnya tidak sebanding. "
        "Periksa distribusi `kategori` sebelum memakai hasil ini.\n")

    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Nama berkas yang dihasilkan skrip ini. Dipakai untuk membersihkan sisa
# keluaran lama: tanpa ini, halaman boxplot dari run sebelumnya masih berdiri
# dan nanti dibaca seolah-olah itu hasil run sekarang.
OUTPUT_STEMS = (
    "summary.md", "missingness.csv", "excluded_rows.csv", "spearman_lila.csv",
    "spearman_lila_heatmap.png", "mannwhitney.csv", "correlation_matrix.csv",
    "correlation_heatmap.png", "scatter_top5.png", "age_analysis.csv",
)
OUTPUT_GLOBS = ("boxplots_page*.png",)


def clean_output_dir(outdir: Path) -> list:
    """Hapus HANYA berkas keluaran milik skrip ini di dalam outdir.

    Sengaja tidak memakai rmtree: berkas lain milik pengguna di folder itu
    tidak boleh ikut terhapus.
    """
    removed = []
    targets = [outdir / name for name in OUTPUT_STEMS]
    for pattern in OUTPUT_GLOBS:
        targets.extend(outdir.glob(pattern))
    for path in targets:
        if path.is_file():
            try:
                path.unlink()
                removed.append(path.name)
            except OSError:
                pass  # tidak bisa dihapus: bukan alasan gagalkan analisis
    return removed


def run(input_csv: Path, outdir: Path, max_nan_rate=0.30, alpha=0.05,
        min_samples=100, min_class=20, redundant_threshold=0.90,
        age_mirror_ratio=0.50, age_mirror_min=0.20, seed=0) -> dict:
    np.random.seed(seed)
    input_csv = Path(input_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise SystemExit(f"Input tidak ditemukan: {input_csv}")

    # Dibersihkan setelah input tervalidasi supaya run yang gagal tidak
    # menghapus hasil run sebelumnya tanpa alasan.
    removed_stale = clean_output_dir(outdir)
    if removed_stale:
        print(f"[eda]_membersihkan {len(removed_stale)} berkas keluaran lama: "
              + ", ".join(sorted(removed_stale)))

    df_raw = pd.read_csv(input_csv)
    if len(df_raw) == 0:
        raise SystemExit(f"Input kosong: {input_csv}")

    warnings = []
    notes = []  # catatan yang harus masuk summary

    # ---- validasi kolom wajib -------------------------------------------
    missing_required = [c for c in (TARGET_COLUMN, LILA_COLUMN) if c not in df_raw.columns]
    if missing_required:
        raise SystemExit(
            f"Kolom wajib tidak ada di {input_csv}: {', '.join(missing_required)}\n"
            f"Kolom yang tersedia: {', '.join(map(str, df_raw.columns))}"
        )

    # ---- baris yang tidak bisa dipakai ----------------------------------
    excl = []
    bad_label = df_raw[TARGET_COLUMN].isna()
    bad_lila = df_raw[LILA_COLUMN].isna()
    for idx, row in df_raw[bad_label].iterrows():
        excl.append({"row_index": idx, "subject_id": row.get("subject_id", ""),
                     "reason": f"kolom '{TARGET_COLUMN}' kosong"})
    for idx, row in df_raw[bad_lila].iterrows():
        excl.append({"row_index": idx, "subject_id": row.get("subject_id", ""),
                     "reason": f"kolom '{LILA_COLUMN}' kosong"})
    excluded_rows = pd.DataFrame(excl, columns=["row_index", "subject_id", "reason"])
    if not excluded_rows.empty:
        excluded_rows.to_csv(outdir / "excluded_rows.csv", index=False)

    df = df_raw[~bad_label & ~bad_lila].reset_index(drop=True)
    n_rows, n_cols = len(df_raw), len(df_raw.columns)
    n_used_rows = len(df)
    if len(excluded_rows):
        warnings.append(
            f"{len(excluded_rows)} baris dikeluarkan dari seluruh analisis "
            f"(label atau LiLA kosong) -> lihat `excluded_rows.csv`."
        )

    # ---- 1. klasifikasi kolom -------------------------------------------
    features, coverage, missingness = classify_columns(df)
    if coverage:
        notes.append(f"{len(coverage)} flag ketersediaan fitur dikeluarkan dari "
                     f"analisis: {', '.join(coverage)}.")

    label_series = df[TARGET_COLUMN].astype(float)
    lila_series = df[LILA_COLUMN].astype(float)
    age_series = df[AGE_COLUMN].astype(float) if AGE_COLUMN in df.columns else None
    sex_col = SEX_COLUMN if SEX_COLUMN in df.columns else None

    if age_series is None:
        notes.append(f"Kolom `{AGE_COLUMN}` tidak ada di features.csv; "
                     "analisis usia (langkah 7) dilewati.")
    elif age_series.isna().all():
        age_series = None
        notes.append(f"Kolom `{AGE_COLUMN}` ada tapi seluruhnya kosong; "
                     "analisis usia (langkah 7) dilewati.")
    if sex_col is None:
        notes.append(f"Kolom `{SEX_COLUMN}` tidak ada; scatter digambar dengan "
                     "satu warna.")

    # `usia` berisi BULAN untuk balita dan TAHUN untuk ibu, dan `usia_bulan`
    # hanya terisi untuk balita. Kalau dataset mencampur keduanya, kontrol
    # usia pada korelasi parsial tidak sebanding satuan.
    if "usia_bulan" in df.columns:
        ub = df["usia_bulan"]
        if ub.notna().any() and ub.isna().any():
            warnings.append(
                "`usia` memakai satuan berbeda per kategori (bulan untuk balita, "
                "tahun untuk ibu) dan `usia_bulan` hanya terisi untuk balita. "
                "Korelasi parsial yang dikontrol usia pada data campuran ini "
                "hanya indikatif kasar.")

    # ---- ambang NaN ------------------------------------------------------
    features, high_nan = drop_high_nan_columns(features, df, max_nan_rate)
    for col, rate in high_nan:
        missingness.loc[missingness["column"] == col, "decision"] = "drop"
        missingness.loc[missingness["column"] == col, "reason"] = (
            f"NaN {rate:.0%} > ambang {max_nan_rate:.0%}")
    missingness.to_csv(outdir / "missingness.csv", index=False)
    if high_nan:
        warnings.append(
            f"{len(high_nan)} kolom dikeluarkan karena NaN > {max_nan_rate:.0%}: "
            + ", ".join(f"{c} ({r:.0%})" for c, r in high_nan)
        )

    if not features:
        # Semua kolom terkena pengecualian/konstan/NaN. Berhenti di sini dengan
        # pesan yang jelas: melanjutkan hanya menghasilkan KeyError 'rho'.
        raise SystemExit(
            "Tidak ada kolom fitur yang tersisa untuk dianalisis.\n"
            f"Semua {n_cols} kolom di {input_csv} terkena pengecualian "
            f"(id/label/kovariat), konstan, atau NaN-nya melebihi "
            f"{max_nan_rate:.0%}.\n"
            f"Alasan per kolom ada di {outdir / 'missingness.csv'}"
        )

    # ---- peringatan ukuran sampel ---------------------------------------
    if n_used_rows < min_samples:
        warnings.append(
            f"**Total sampel {n_used_rows} < {min_samples}.** Korelasi dan "
            "p-value di sini tidak stabil; dipakai untuk eksplorasi saja."
        )
    counts = label_series.value_counts().to_dict()
    for lab, n in sorted(counts.items()):
        if n < min_class:
            warnings.append(
                f"**Kelas label={fmt_label(lab)} hanya punya {n} sampel "
                f"(< {min_class}).** Uji Mann-Whitney dan boxplot kelas ini "
                "tidak bisa diandalkan."
            )
    missing_classes = sorted({0, 1} - set(counts))
    if missing_classes:
        warnings.append(
            "**Kelas label " + ", ".join(map(str, missing_classes)) + " tidak ada "
            "sama sekali.** Perbandingan 0 vs 1 tidak bisa dihitung; tabel uji "
            "dan boxplot kelas kosong ditandai di tabel masing-masing."
        )

    # ---- 3. Spearman vs LiLA ---------------------------------------------
    sp_rows = []
    for feat in features:
        rho, p, n_used, reason = safe_spearman(df[feat], lila_series)
        sp_rows.append({"feature": feat, "rho": rho, "p_value": p,
                        "n_used": n_used, "note": reason})
    sp = pd.DataFrame(sp_rows)
    sp["abs_rho"] = sp["rho"].abs()
    sp = sp.sort_values("abs_rho", ascending=False, na_position="last").reset_index(drop=True)
    sp.insert(0, "rank", np.arange(1, len(sp) + 1))
    sp.to_csv(outdir / "spearman_lila.csv", index=False)

    plot_heatmap(sp["rho"].to_numpy(dtype=float).reshape(1, -1), [LILA_COLUMN],
                 list(sp["feature"]), "Korelasi Spearman fitur vs LiLA (terurut)",
                 outdir / "spearman_lila_heatmap.png", figsize=(0.42 * len(sp) + 3, 3.2))

    # ---- 4. Mann-Whitney U + BH -----------------------------------------
    mwu_rows = []
    for feat in features:
        g1 = df.loc[label_series == 1, feat].dropna().to_numpy()
        g0 = df.loc[label_series == 0, feat].dropna().to_numpy()
        u, p, r_rb, n1, n0, reason = mannwhitney_with_effect(g1, g0)
        mwu_rows.append({"feature": feat, "u_stat": u, "p_raw": p,
                         "rank_biserial": r_rb, "n_label1": n1, "n_label0": n0,
                         "note": reason})
    mwu = pd.DataFrame(mwu_rows)
    mwu["p_bh"] = benjamini_hochberg(list(mwu["p_raw"]))
    mwu["significant"] = (mwu["p_bh"] < alpha).fillna(False)
    mwu = mwu.sort_values("p_bh", ascending=True, na_position="last").reset_index(drop=True)
    mwu.insert(0, "rank", np.arange(1, len(mwu) + 1))
    mwu.to_csv(outdir / "mannwhitney.csv", index=False)

    effect_label = (f"positif = nilai fitur cenderung lebih tinggi pada label 1 "
                    f"(n label=1={int((label_series == 1).sum())}, "
                    f"n label=0={int((label_series == 0).sum())})")

    # ---- 6. korelasi antar fitur + redundansi ---------------------------
    if len(features) >= 2:
        corr_mat = plot_correlation_matrix(df, features, outdir / "correlation_heatmap.png")
    else:
        corr_mat = pd.DataFrame()
        notes.append("Hanya satu fitur yang tersisa; matriks korelasi antar fitur "
                     "tidak dihitung.")
    redundant_rows = []
    if not corr_mat.empty:
        corr_mat.to_csv(outdir / "correlation_matrix.csv")
        for i, a in enumerate(features):
            for j, b in enumerate(features):
                if j <= i:
                    continue
                r = corr_mat.iat[i, j]
                if np.isfinite(r) and abs(r) > redundant_threshold:
                    redundant_rows.append({
                        "feature_a": a, "feature_b": b, "rho": round(float(r), 4),
                        "abs_rho": round(abs(float(r)), 4),
                        "note": "saling menggantikan, pilih salah satu",
                    })

    # ---- 7. analisis usia -----------------------------------------------
    age_rows = []
    if age_series is not None:
        for feat in features:
            rho_lila, _, n_used, note1 = safe_spearman(df[feat], lila_series)
            rho_age, _, _, note2 = safe_spearman(df[feat], age_series)
            rho_par, n_par, note3 = partial_spearman(df[feat], lila_series, age_series)
            drop_ratio = (abs(rho_lila) - abs(rho_par)) / abs(rho_lila) if (
                np.isfinite(rho_lila) and abs(rho_lila) > 1e-12 and np.isfinite(rho_par)
            ) else float("nan")
            age_rows.append({
                "feature": feat, "rho_lila": rho_lila, "rho_age": rho_age,
                "rho_partial": rho_par, "drop_ratio": drop_ratio, "n_used": n_par,
                "rho_lila_partial_p": partial_corr_pvalue(rho_par, n_par),
                "note": "; ".join(x for x in (note1, note2, note3) if x),
            })
        age_df = pd.DataFrame(age_rows)
        # Perhatikan tanda kurung: di Python, `&` berikatan lebih longgar dari
        # perbandingan, jadi `a >= b & c` akan di-parse jadi `a >= (b & c)`.
        r_lila = age_df["rho_lila"].abs()
        r_par = age_df["rho_partial"].abs()
        r_age = age_df["rho_age"].abs()
        age_df["age_mirror"] = (
            (r_lila >= age_mirror_min)
            & (r_par <= age_mirror_ratio * r_lila)
            & (r_age >= age_mirror_min)
        )
        age_df = age_df.sort_values("rho_lila", key=lambda s: s.abs(),
                                    ascending=False, na_position="last")
        age_df.to_csv(outdir / "age_analysis.csv", index=False)
    else:
        age_df = pd.DataFrame()

    # ---- 2. boxplot + 5. scatter ----------------------------------------
    boxplot_files = plot_boxplots(df, features, TARGET_COLUMN, outdir)
    top_scatter = [r for r in sp["feature"].head(5).tolist() if r in features]
    if top_scatter:
        plot_scatter_top(df, top_scatter, LILA_COLUMN, sex_col, outdir / "scatter_top5.png")
    else:
        notes.append("Tidak ada fitur yang bisa diplot pada scatter.")

    # ---- rangkuman fitur signifikan --------------------------------------
    sig = mwu[mwu["significant"]].copy()
    if not sig.empty:
        sig = sig.merge(sp[["feature", "rho"]], on="feature", how="left")
        sig = sig.rename(columns={"rho": "rho_lila"})
        if age_series is not None and not age_df.empty:
            sig = sig.merge(age_df[["feature", "rho_age", "rho_partial", "age_mirror"]],
                            on="feature", how="left")
            sig["note"] = np.where(sig["age_mirror"], "perlu dicek: sinyal mungkin cerminan usia", "")
        else:
            sig["rho_age"] = np.nan
            sig["rho_partial"] = np.nan
        sig = sig.sort_values("p_bh").reset_index(drop=True)
        sig = sig.drop(columns=["rank"], errors="ignore")
        sig.insert(0, "rank", np.arange(1, len(sig) + 1))

    # ---- pairwise deletion audit ----------------------------------------
    pairwise_rows = []
    for feat in features:
        n_total = n_used_rows
        n_used = int((df[feat].notna() & lila_series.notna()).sum())
        if n_used < n_total:
            pairwise_rows.append({
                "feature": feat, "n_total": n_total, "n_used": n_used,
                "n_dropped": n_total - n_used,
                "note": "baris dengan NaN pada kolom ini tidak dipakai untuk kolom ini saja",
            })

    # ---- label distribution & NaN rows ----------------------------------
    label_rows = []
    for lab, n in sorted(counts.items()):
        pct = f"{n / n_used_rows:.1%}" if n_used_rows else "n/a"
        label_rows.append({"label": fmt_label(lab), "n": n, "persen": pct})
    nan_rows = []
    for feat in features:
        n_nan = int(df[feat].isna().sum())
        rate = n_nan / n_used_rows if n_used_rows else 0.0
        decision = "keep"
        reason = ""
        if rate > max_nan_rate:
            decision, reason = "drop", f"NaN {rate:.0%} > ambang {max_nan_rate:.0%}"
        else:
            reason = "lolos ambang NaN"
        nan_rows.append({"column": feat, "n_nan": n_nan,
                         "nan_rate": f"{rate:.1%}", "decision": decision, "reason": reason})

    age_summary = (f"{age_series.min():.0f}-{age_series.max():.0f} tahun "
                   f"(median {age_series.median():.0f})") if age_series is not None else None
    sex_summary = ", ".join(
        f"{k}={v}" for k, v in df[sex_col].value_counts().items()) if sex_col else None

    ctx = {
        "df": df, "input": input_csv, "outdir": outdir,
        "n_rows": n_rows, "n_cols": n_cols, "n_used_rows": n_used_rows,
        "features": features, "warnings": warnings, "notes": notes, "removed_stale": removed_stale,
        "lila_mean": float(lila_series.mean()), "lila_min": float(lila_series.min()),
        "lila_max": float(lila_series.max()),
        "age_summary": age_summary, "sex_summary": sex_summary,
        "label_rows": label_rows, "nan_rows": nan_rows,
        "max_nan_rate": max_nan_rate, "boxplot_files": boxplot_files,
        "spearman_rows": sp.head(20).to_dict("records"),
        "mwu_rows": mwu.to_dict("records"), "n_tests": len(features),
        "alpha": alpha, "n_significant": int(mwu["significant"].sum()),
        "effect_label": effect_label,
        "top_scatter": top_scatter,
        "redundant_threshold": redundant_threshold, "redundant_rows": redundant_rows,
        "age_available": age_series is not None,
        "age_mirror_rows": (age_df[age_df["age_mirror"]].to_dict("records")
                            if age_series is not None else []),
        "age_mirror_count": (int(age_df["age_mirror"].sum())
                             if age_series is not None else 0),
        "age_top_rows": (age_df.reindex(age_df["rho_age"].abs().sort_values(ascending=False).index)
                         .head(10).to_dict("records") if age_series is not None else []),
        "sig_rows": sig.to_dict("records") if not sig.empty else [],
        "missingness": missingness, "excluded_rows": excluded_rows,
        "pairwise_rows": pairwise_rows,
    }
    write_summary(outdir / "summary.md", ctx)

    return {
        "n_rows": n_rows, "n_used_rows": n_used_rows,
        "n_excluded_rows": len(excluded_rows), "n_features": len(features),
        "n_significant": int(mwu["significant"].sum()),
        "n_redundant": len(redundant_rows),
        "n_age_mirror": ctx["age_mirror_count"],
        "outdir": str(outdir), "warnings": warnings, "notes": notes,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="EDA features.csv (deskriptif saja, tanpa training model).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", type=Path, default=ROOT / "data" / "features.csv",
                    help="CSV hasil scripts/extract_dataset.py")
    ap.add_argument("--outdir", type=Path, default=ROOT / "reports" / "eda",
                    help="folder keluaran (summary.md + tabel + plot)")
    ap.add_argument("--max-nan-rate", type=float, default=0.30,
                    help="kolom dengan NaN di atas ambang ini dikeluarkan dari analisis")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="ambang signifikansi setelah koreksi BH")
    ap.add_argument("--min-samples", type=int, default=100,
                    help="peringatan jika total sampel di bawah ini")
    ap.add_argument("--min-class", type=int, default=20,
                    help="peringatan jika salah satu kelas label di bawah ini")
    ap.add_argument("--redundant-threshold", type=float, default=0.90,
                    help="|rho| di atas ini ditandai redundan")
    ap.add_argument("--age-mirror-ratio", type=float, default=0.50,
                    help="penurunan |rho| parsial >= rasio ini -> dicurigai cerminan usia")
    ap.add_argument("--age-mirror-min", type=float, default=0.20,
                    help="|rho| minimal agar fitur dianggap punya sinyal")
    ap.add_argument("--seed", type=int, default=0, help="seed (untuk reproduksibilitas)")
    args = ap.parse_args(argv)

    res = run(
        args.input, args.outdir,
        max_nan_rate=args.max_nan_rate, alpha=args.alpha,
        min_samples=args.min_samples, min_class=args.min_class,
        redundant_threshold=args.redundant_threshold,
        age_mirror_ratio=args.age_mirror_ratio,
        age_mirror_min=args.age_mirror_min, seed=args.seed,
    )

    print()
    print("=" * 70)
    print("RINGKASAN EDA")
    print("=" * 70)
    print(f"Baris dibaca        : {res['n_rows']}")
    print(f"Baris dianalisis    : {res['n_used_rows']} "
          f"(dibuang: {res['n_excluded_rows']})")
    print(f"Fitur dianalisis    : {res['n_features']}")
    print(f"Signifikan (BH)     : {res['n_significant']}")
    print(f"Pasangan redundan   : {res['n_redundant']}")
    print(f"Curiga cerminan usia : {res['n_age_mirror']}")
    print(f"Keluaran            : {res['outdir']}")
    if res["warnings"] or res["notes"]:
        print()
        if res["warnings"]:
            print("PERINGATAN:")
            for w in res["warnings"]:
                print(f"  - {w}")
        if res["notes"]:
            print("CATATAN:")
            for n in res["notes"]:
                print(f"  - {n}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())