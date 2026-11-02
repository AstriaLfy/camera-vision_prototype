"""
test_extract_dataset.py
-----------------------
Test untuk scripts/extract_dataset.py.

Yang diuji hanya bagian yang ARITMETIKA/ATURANNYA bisa salah diam-diam:
penetapan label dari LiLA, penanganan input-label rusak, dan syarat
"tidak ada baris yang hilang tanpa jejak". Feature extraction-nya sendiri sudah
diuji di test_end_to_end.py -- di sini kita hanya memastikan skrip memakai
fungsi pipeline yang benar, bukan implementasi ulang.

Tabel WHO BMI-for-age tetap dimuat dan diuji (loader + rumus z), tetapi TIDAK
dipakai menetapkan label: satu-satunya sumber label adalah LiLA.

Fungsi-fungsi ini sengaja ditulis tanpa menyentuh app/pipeline.py.
"""

import csv
import math
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import extract_dataset as ed  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture WHO
#
# Angka di bawah TIDAK ditulis dari ingatan: semuanya dibaca dari file WHO
# asli di data/who/ waktu test jalan. Kalau file itu hilang, fixture ini gagal
# lebih dulu -- bukan diam-diam pakai angka karangan.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def who_rows():
    """Baris mentah WHO termasuk kolom SD* (pembanding, bukan sumber label)."""
    rows = {}
    for sex, path in ed.WHO_FILES.items():
        assert path.exists(), (
            f"test butuh file WHO asli di {path}; jangan diganti angka karangan"
        )
        with path.open(newline="", encoding="utf-8-sig") as fh:
            raw = list(csv.DictReader(fh))
        rows[sex] = {int(r["Month"]): {k: float(v) for k, v in r.items()}
                     for r in raw}
    return rows


@pytest.fixture(scope="session")
def who_tables():
    """Tabel L/M/S lewat loader resmi skrip (ikut memvalidasi file asli)."""
    return ed.load_who_tables()


# Bulan yang dipakai lintas test: dua ujung rentang + dua di tengah.
FIXTURE_MONTHS = (61, 100, 150, 228)


# ---------------------------------------------------------------------------
# Penetapan label dari LiLA
# ---------------------------------------------------------------------------


def _labels_csv(tmp_path: Path, rows: list[dict], images_dir: Path | None = None) -> Path:
    images_dir = images_dir or (tmp_path / "images")
    images_dir.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "labels.csv"
    # gabung semua key yang muncul supaya kolom opsional ikut terbawa
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return p


def _row(**kw):
    """Baris labels.csv. subject_id/image_path/kategori/lila_cm/berat_kg wajib."""
    base = dict(subject_id="S1", image_path="a.jpg", kategori="balita",
                usia="12", jenis_kelamin="L", lila_cm="13.0", berat_kg="9.0")
    base.update(kw)
    return base


def _ibu_row(**kw):
    """Baris ibu: usia dalam TAHUN, jenis_kelamin boleh kosong."""
    base = dict(kategori="ibu_hamil", usia="28", jenis_kelamin="",
                lila_cm="24.0", berat_kg="55.0")
    base.update(kw)
    return _row(**base)


def _build(row, tmp_path):
    """build_label_row dengan folder gambar yang selalu sama."""
    return ed.build_label_row(row, tmp_path / "images")


@pytest.mark.parametrize("lila,expect", [
    ("12.4999", 1),
    ("12.5", 0),          # tepat di ambang -> bukan lagi gizi kurang
    ("13.0", 0),
    ("11.5", 1),          # kuning tetap label 1
    ("5.0", 1),
])
def test_balita_label_is_1_below_12_5_and_0_at_or_above(tmp_path, lila, expect):
    """Ambang balita LiLA 12.5 cm; label bukan BMI, jadi 12.5 harus teruji."""
    out = _build(_row(lila_cm=lila), tmp_path)
    assert out["label_underweight"] == expect, f"lila_cm={lila}"
    assert out["label_method"] == "lila_balita"
    assert out["kategori"] == "balita"


@pytest.mark.parametrize("lila,risk", [
    ("11.4999", "merah"),
    ("11.5", "kuning"),      # batas merah/kuning persis 11.5
    ("12.4999", "kuning"),
    ("12.5", "hijau"),       # batas kuning/hijau persis 12.5
    ("20.0", "hijau"),
])
def test_balita_kategori_risiko_boundaries(tmp_path, lila, risk):
    out = _build(_row(lila_cm=lila), tmp_path)
    assert out["kategori_risiko"] == risk, f"lila_cm={lila}"


@pytest.mark.parametrize("lila,expect,risk", [
    ("23.4999", 1, "merah"),
    ("23.5", 0, "hijau"),    # tepat di ambang -> hijau
    ("23.5001", 0, "hijau"),
    ("30.0", 0, "hijau"),
    ("20.0", 1, "merah"),
])
@pytest.mark.parametrize("kategori", ["ibu_hamil", "ibu_menyusui"])
def test_ibu_label_and_risk_use_23_5_cm(tmp_path, kategori, lila, expect, risk):
    """Ibu hamil/menyusui memakai satu ambang LiLA 23.5 cm."""
    out = _build(_ibu_row(kategori=kategori, lila_cm=lila), tmp_path)
    assert out["label_underweight"] == expect, f"{kategori} lila_cm={lila}"
    assert out["kategori_risiko"] == risk
    assert out["label_method"] == "lila_ibu"
    assert out["kategori"] == kategori


def test_ibu_has_no_kuning_category(tmp_path):
    """'kuning' itu aturan balita; ibu cuma merah/hijau."""
    for lila in ("11.0", "23.4999", "23.5", "30.0"):
        risk = _build(_ibu_row(lila_cm=lila), tmp_path)["kategori_risiko"]
        assert risk in ("merah", "hijau"), f"lila_cm={lila} -> {risk}"
    assert _build(_ibu_row(lila_cm="11.0"), tmp_path)["kategori_risiko"] == "merah"


@pytest.mark.parametrize("bulan,expect", [
    ("0", "reject"),
    ("5", "reject"),
    ("6", "ok"),        # batas bawah
    ("59", "ok"),       # batas atas
    ("60", "reject"),
])
def test_balita_age_boundaries_are_6_to_59_months(tmp_path, bulan, expect):
    """5 dan 60 bulan harus ditolak, bukan di-clip diam-diam ke batas terdekat."""
    row = _row(kategori="balita", usia="", usia_bulan=bulan)
    if expect == "reject":
        with pytest.raises(ed.Rejected) as exc:
            _build(row, tmp_path)
        assert exc.value.stage == "label"
        assert exc.value.reason == "LiLA tidak valid di luar 6-59 bulan"
        assert exc.value.reason == ed.ALASAN_BALITA_LUAR_RENTANG
    else:
        out = _build(row, tmp_path)
        assert out["usia_bulan"] == float(bulan)
        assert out["label_underweight"] in (0, 1)


def test_balita_usia_column_is_read_as_months(tmp_path):
    """Untuk balita kolom `usia` satuan BULAN, bukan tahun."""
    assert _build(_row(usia="6", usia_bulan=""), tmp_path)["usia_bulan"] == 6
    assert _build(_row(usia="59", usia_bulan=""), tmp_path)["usia_bulan"] == 59
    for usia in ("5", "60"):
        with pytest.raises(ed.Rejected) as exc:
            _build(_row(usia=usia, usia_bulan=""), tmp_path)
        assert exc.value.reason == ed.ALASAN_BALITA_LUAR_RENTANG, f"usia={usia}"


def test_usia_bulan_column_wins_over_usia(tmp_path):
    """Kolom usia_bulan didahulukan kalau keduanya ada."""
    out = _build(_row(usia="12", usia_bulan="59"), tmp_path)
    assert out["usia_bulan"] == 59, "usia_bulan harus menang atas usia"
    assert out["usia"] == "12", "nilai asli dari labels.csv dipertahankan"
    assert _build(_row(usia="60", usia_bulan="6"), tmp_path)["usia_bulan"] == 6


def test_ibu_usia_in_years_kept_but_not_used_as_months(tmp_path):
    """usia ibu dalam TAHUN disimpan apa adanya; usia_bulan dikosongkan."""
    out = _build(_ibu_row(usia="28"), tmp_path)
    assert out["usia"] == "28"
    assert out["usia_bulan"] == "", "tahun ibu tidak boleh dikalikan 12"
    # kolom usia_bulan yang terisi tetap dipakai apa adanya, bukan dikalikan
    out2 = _build(_ibu_row(usia="28", usia_bulan="300"), tmp_path)
    assert out2["usia_bulan"] == 300


@pytest.mark.parametrize("raw,expected", [
    ("BALITA", "balita"),
    ("  balita ", "balita"),
    ("ibu hamil", "ibu_hamil"),
    ("ibu-hamil", "ibu_hamil"),
    ("IBU_MENYUSUI", "ibu_menyusui"),
    ("Ibu Menyusui", "ibu_menyusui"),
])
def test_kategori_values_are_normalised(tmp_path, raw, expected):
    out = _build(_row(kategori=raw), tmp_path)
    assert out["kategori"] == expected, raw


def test_image_path_is_preserved_and_resolved(tmp_path):
    """image_path harus tetap sama dengan labels.csv supaya bisa di-join."""
    out = _build(_row(image_path="sub/dir/a.jpg"), tmp_path)
    assert out["image_path"] == "sub/dir/a.jpg"
    assert out["image_abs_path"] == str(tmp_path / "images" / "sub/dir/a.jpg")


def test_absolute_image_path_is_kept(tmp_path):
    abs_path = str(tmp_path / "abs.jpg")
    out = ed.build_label_row(_row(image_path=abs_path), tmp_path)
    assert out["image_abs_path"] == abs_path


@pytest.mark.parametrize("row,fragment", [
    (_row(subject_id=""), "subject_id kosong"),
    (_row(image_path=""), "image_path kosong"),
    (_row(kategori=""), "kategori kosong"),
    (_row(kategori="dewasa"), "kategori tidak dikenali"),
    (_row(kategori="ibu hamil bayi"), "kategori tidak dikenali"),
    (_row(lila_cm=""), "kosong"),
    (_row(lila_cm="abc"), "bukan angka"),
    (_row(lila_cm="NaN"), "bukan angka berhingga"),
    (_row(lila_cm="inf"), "bukan angka berhingga"),
    (_row(berat_kg=""), "kosong"),
    (_row(berat_kg="NaN"), "bukan angka berhingga"),
    (_row(berat_kg="abc"), "bukan angka"),
    (_row(usia=""), "usia kosong"),
    (_row(usia="thIRT"), "bukan angka"),
    (_row(usia="NaN"), "bukan angka berhingga"),
    (_row(usia_bulan="abc"), "usia_bulan"),
    (_row(usia_bulan="-1"), "tidak masuk akal"),
    (_row(jenis_kelamin=""), "jenis_kelamin"),
])
def test_invalid_label_rows_raise_with_useful_reason(tmp_path, row, fragment):
    """Setiap kegagalan label harus jatuh ke tahap 'label' dengan pesan jelas.

    Kalau baris ini diam-diam dilewati, dataset jadi lebih kecil tanpa jejak.
    """
    with pytest.raises(ed.Rejected) as exc:
        _build(row, tmp_path)
    assert exc.value.stage == "label"
    assert fragment in exc.value.reason, exc.value.reason


@pytest.mark.parametrize("lila,ok", [
    ("4.9", False), ("5.0", True), ("50.0", True), ("50.1", False),
    ("0", False), ("100", False),
])
def test_lila_must_be_within_5_to_50_cm(tmp_path, lila, ok):
    """Di luar rentang wajar berarti salah ketik; jangan dipakai diam-diam."""
    if ok:
        out = _build(_row(lila_cm=lila), tmp_path)
        assert out["lila_cm"] == pytest.approx(float(lila), abs=1e-9)
    else:
        with pytest.raises(ed.Rejected) as exc:
            _build(_row(lila_cm=lila), tmp_path)
        assert "rentang wajar" in exc.value.reason, exc.value.reason


@pytest.mark.parametrize("field,value,fragment", [
    ("lila_cm", "NaN", "bukan angka berhingga"),
    ("lila_cm", "-inf", "bukan angka berhingga"),
    ("berat_kg", "NaN", "bukan angka berhingga"),
    ("berat_kg", "inf", "bukan angka berhingga"),
    ("berat_kg", "0", "tidak masuk akal"),
    ("berat_kg", "-1", "tidak masuk akal"),
    ("panjang_tinggi_cm", "0", "tidak masuk akal"),
    ("panjang_tinggi_cm", "abc", "bukan angka"),
])
def test_non_finite_or_absurd_measurements_are_rejected(tmp_path, field, value, fragment):
    """NaN/inf/nol negatif tidak boleh menghasilkan label."""
    with pytest.raises(ed.Rejected) as exc:
        _build(_row(**{field: value}), tmp_path)
    assert exc.value.stage == "label"
    assert fragment in exc.value.reason, exc.value.reason
    assert field in exc.value.reason


def test_jenis_kelamin_is_mandatory_for_balita(tmp_path):
    with pytest.raises(ed.Rejected) as exc:
        _build(_row(jenis_kelamin=""), tmp_path)
    assert "jenis_kelamin" in exc.value.reason


@pytest.mark.parametrize("kategori", ["ibu_hamil", "ibu_menyusui"])
def test_jenis_kelamin_is_optional_for_ibu(tmp_path, kategori):
    out = _build(_ibu_row(kategori=kategori, jenis_kelamin=""), tmp_path)
    assert out["jenis_kelamin"] == "", f"{kategori} tidak butuh jenis kelamin"
    assert out["label_underweight"] in (0, 1)


def test_panjang_tinggi_cm_is_stored_without_z_score(tmp_path):
    """panjang_tinggi_cm disimpan apa adanya; z-score balita sengaja belum ada."""
    out = _build(_row(panjang_tinggi_cm="71.5"), tmp_path)
    assert out["panjang_tinggi_cm"] == pytest.approx(71.5)
    assert _build(_row(panjang_tinggi_cm=""), tmp_path)["panjang_tinggi_cm"] == ""


def test_no_bmi_column_left_in_output(tmp_path):
    """BMI sudah keluar dari alur label; tidak boleh muncul lagi di mana pun."""
    out = _build(_row(), tmp_path)
    for col in ("bmi", "bmi_z", "bmi_category", "tinggi_cm"):
        assert col not in out, f"{col} masih muncul di baris output"
        assert col not in ed.OUTPUT_COLUMNS, f"{col} masih ada di skema"
    assert ed.OUTPUT_COLUMNS[:4] == ["subject_id", "image_path",
                                     "image_abs_path", "kategori"]


# ---------------------------------------------------------------------------
# Referensi tabel WHO BMI-for-age (diuji, bukan sumber label)
#
# Skrip memuat tabel ini untuk audit, tetapi tidak memakainya menetapkan label.
# Semua angka di bawah diambil dari file WHO asli di data/who/ lewat fixture;
# tidak ada satu pun angka tabel yang diketik dari ingatan.
# ---------------------------------------------------------------------------

def test_who_real_files_load_and_cover_required_range(who_tables, who_rows):
    """Sanity: file WHO asli ada, valid, dan menutup 61-228 bulan."""
    assert set(who_tables) == {"boys", "girls"}
    for sex in ("boys", "girls"):
        assert sorted(who_tables[sex]) == list(range(61, 229))
        assert sorted(who_rows[sex]) == list(range(61, 229))
        # kolom SD* memang ada (dipakai pembanding, bukan sumber label)
        for col in ("L", "M", "S", "SD0", "SD2neg", "SD3neg"):
            assert col in who_rows[sex][61], f"kolom {col} hilang"


def test_z_score_matches_who_sd_columns(who_rows, who_tables):
    """Syarat utama: BMI = SD0 -> z ~ 0; SD2neg -> z ~ -2; SD3neg -> z ~ -3.

    Dipakai L/M/S dari tabel dan SD* sebagai pembanding, persis seperti
    pemakaian aslinya. Toleransi 0.01 karena kolom SD* di file hanya dibulatkan
    ke 4 signifikan -- deviasi terukur di seluruh file cuma ~0.0006.
    """
    for sex in ("boys", "girls"):
        for month in FIXTURE_MONTHS:
            src = who_rows[sex][month]
            L, M, S = who_tables[sex][month]
            assert ed.who_zscore(src["SD0"], L, M, S) == pytest.approx(0.0, abs=0.01), \
                f"{sex} bulan {month}: z(SD0) harus mendekati 0"
            assert ed.who_zscore(src["SD2neg"], L, M, S) == pytest.approx(-2.0, abs=0.01), \
                f"{sex} bulan {month}: z(SD2neg) harus mendekati -2"
            assert ed.who_zscore(src["SD3neg"], L, M, S) == pytest.approx(-3.0, abs=0.01), \
                f"{sex} bulan {month}: z(SD3neg) harus mendekati -3"


def test_z_score_is_monotonic_in_bmi(who_rows, who_tables):
    """z harus naik kalau BMI naik; kalau tidak, perhitungannya terbalik."""
    for sex in ("boys", "girls"):
        L, M, S = who_tables[sex][150]
        src = who_rows[sex][150]
        ordered = [src["SD3neg"], src["SD2neg"], src["SD1neg"], src["SD0"],
                   src["SD1"], src["SD2"]]
        zs = [ed.who_zscore(b, L, M, S) for b in ordered]
        assert zs == sorted(zs), f"{sex}: z tidak monoton terhadap BMI"


def test_z_score_formula_matches_lms_quantile_inverse(who_rows, who_tables):
    """Balik-balik: dari z -> BMI memakai rumus kuantil LMS, harus kembali sama.

    Rumus kuantilnya derivasi algebrainya dari rumus z, jadi kalau ada yang
    salah di cabang pangkat/ln, test ini menangkapnya.
    """
    for sex in ("boys", "girls"):
        for month in FIXTURE_MONTHS:
            L, M, S = who_tables[sex][month]
            bmi = who_rows[sex][month]["SD2neg"]
            z = ed.who_zscore(bmi, L, M, S)
            if L == 0:
                recovered = M * math.exp(S * z)
            else:
                recovered = M * (1.0 + L * S * z) ** (1.0 / L)
            assert recovered == pytest.approx(bmi, rel=1e-9)


def test_z_score_l_zero_branch():
    """Cabang L = 0: z = ln(BMI/M)/S.

    Tabel WHO yang kita pakai ber-L negatif semua, jadi cabang ini tidak
    pernah kena dari file aslinya. Tapi cabang ini wajib ada dan benar karena
    L=0 adalah batas rumus LMS, dan tabel lain bisa saja ber-L nol. Angka di
    sini dipakai untuk MENGUJI rumus, bukan sebagai tabel WHO.
    """
    L, M, S = 0.0, 20.0, 0.1

    assert ed.who_zscore(M, L, M, S) == pytest.approx(0.0)
    # ln(20.6/20)/0.1 = ln(1.03)*10 ~= 2.9559
    assert ed.who_zscore(20.6, L, M, S) == pytest.approx(math.log(1.03) * 10, rel=1e-12)
    # benang halus: kalau cabang ln lupa dan jatuh ke pangkat, hasilnya error
    with pytest.raises(ZeroDivisionError):
        ((20.6 / M) ** L - 1.0) / (L * S)

    # dan cabang L=0 harus setara dengan limit L->0 dari rumus utama
    for tiny in (1e-6, -1e-6):
        assert ed.who_zscore(20.6, tiny, M, S) == pytest.approx(
            math.log(20.6 / M) / S, rel=1e-4)


def test_categorize_z_matches_who_sd_columns(who_rows, who_tables):
    """Kategori z diuji terhadap kolom SD* file WHO asli (retensi, bukan label)."""
    for sex in ("boys", "girls"):
        for month in FIXTURE_MONTHS:
            src = who_rows[sex][month]
            L, M, S = who_tables[sex][month]
            where = f"{sex} bulan {month}"

            z = lambda v: ed.who_zscore(v, L, M, S)
            assert ed.categorize_z(z(src["SD0"])) == "normal", where
            # z ~ -2.5: jauh dari kedua ambang, jadi pembulatan file tak berpengaruh
            mid = (src["SD2neg"] + src["SD3neg"]) / 2
            assert ed.categorize_z(z(mid)) == "thinness", where
            assert ed.categorize_z(z(src["SD4neg"])) == "severe_thinness", where


def test_categorize_z_boundaries_are_strict():
    """-2 dan -3 persis bukan thinness/severe; batasnya `<`, bukan `<=`."""
    assert ed.categorize_z(0.0) == "normal"
    assert ed.categorize_z(-1.9999999) == "normal"
    assert ed.categorize_z(-2.0) == "normal"
    assert ed.categorize_z(-2.0000001) == "thinness"
    assert ed.categorize_z(-3.0) == "thinness"
    assert ed.categorize_z(-3.0000001) == "severe_thinness"


def test_lms_lookup_refuses_out_of_range_month():
    """lms_for_month tidak boleh meng-ekstrapolasi keluar tabel."""
    tables = ed.load_who_tables()
    with pytest.raises(ed.WhoTableError):
        ed.lms_for_month(tables["boys"], 60)
    with pytest.raises(ed.WhoTableError):
        ed.lms_for_month(tables["boys"], 229)


def test_fractional_month_inside_range_is_interpolated():
    """Bulan pecahan diinterpolasi, bukan dibulatkan ke bulan terdekat."""
    tables = ed.load_who_tables()
    lo = tables["boys"][100]
    hi = tables["boys"][101]
    got = ed.lms_for_month(tables["boys"], 100.5)
    assert got == pytest.approx(tuple((a + b) / 2 for a, b in zip(lo, hi)))


# --- jenis kelamin (pemetaan ke tabel WHO; retensi) ------------------------

@pytest.mark.parametrize("sex,gender", [
    ("boys", "L"), ("boys", "laki-laki"), ("boys", "laki"), ("boys", "male"),
    ("boys", "Laki_Laki"),
    ("girls", "P"), ("girls", "perempuan"), ("girls", "female"), ("girls", "pr"),
])
def test_accepted_jenis_kelamin_values(sex, gender):
    """Nilai yang konsisten diterima dan memetakan ke tabel WHO yang benar."""
    assert ed.parse_jenis_kelamin(gender) == sex


@pytest.mark.parametrize("gender", ["", "laki-laki/perempuan", "L/P", "biru",
                                    "wanita", "1"])
def test_unusable_jenis_kelamin_is_rejected(gender):
    """Selain nilai yang konsisten harus ditolak, bukan ditebak."""
    with pytest.raises(ed.Rejected) as exc:
        ed.parse_jenis_kelamin(gender)
    assert "jenis_kelamin" in exc.value.reason


def test_label_method_column_is_in_output_schema():
    """features.csv wajib punya kolom label LiLA (bukan z-score BMI)."""
    for col in ("label_method", "kategori", "kategori_risiko", "lila_cm",
                "usia_bulan", "label_underweight"):
        assert col in ed.OUTPUT_COLUMNS, f"kolom {col} hilang dari skema"


# --- validasi pemuatan tabel WHO ------------------------------------------

def _copy_who_boys(tmp_path: Path) -> Path:
    """Salin file WHO asli untuk dimutasi (pembanding validasi)."""
    src = ed.WHO_FILES["boys"]
    assert src.exists(), f"butuh file WHO asli: {src}"
    dst = tmp_path / "bmi_for_age_boys.csv"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def test_who_loader_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(ed.WhoTableError) as exc:
        ed.load_who_table(tmp_path / "tidak_ada.csv")
    msg = str(exc.value)
    assert "tidak ditemukan" in msg
    assert "TIDAK membuat tabel pengganti" in msg


def test_who_loader_rejects_missing_required_column(tmp_path):
    path = _copy_who_boys(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("L,", "", 1)          # buang kolom L dari header
    for i in range(1, len(lines)):
        lines[i] = lines[i].split(",", 1)[1]           # buang nilainya juga
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ed.WhoTableError) as exc:
        ed.load_who_table(path)
    assert "L" in str(exc.value)


def test_who_loader_rejects_nan_in_lms(tmp_path):
    path = _copy_who_boys(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    parts = lines[1].split(",")
    parts[1] = "NaN"                                    # kolom L baris pertama
    lines[1] = ",".join(parts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ed.WhoTableError) as exc:
        ed.load_who_table(path)
    assert "berhingga" in str(exc.value)


def test_who_loader_rejects_non_decimal_value(tmp_path):
    path = _copy_who_boys(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    parts = lines[1].split(",")
    parts[2] = "fifteen"                               # kolom M
    lines[1] = ",".join(parts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ed.WhoTableError) as exc:
        ed.load_who_table(path)
    assert "desimal" in str(exc.value)


def test_who_loader_rejects_gap_in_months(tmp_path):
    path = _copy_who_boys(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[10]                                       # hilangkan satu bulan
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ed.WhoTableError) as exc:
        ed.load_who_table(path)
    assert "celah" in str(exc.value)


def test_who_loader_rejects_incomplete_range(tmp_path):
    """Tabel yang berhenti sebelum 228 tidak cukup; jangan dipakai diam-diam."""
    path = _copy_who_boys(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-5]) + "\n", encoding="utf-8")
    with pytest.raises(ed.WhoTableError) as exc:
        ed.load_who_table(path)
    assert "228" in str(exc.value)


def test_who_loader_rejects_zero_s(tmp_path):
    path = _copy_who_boys(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    parts = lines[1].split(",")
    parts[3] = "0"                                      # kolom S
    lines[1] = ",".join(parts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ed.WhoTableError) as exc:
        ed.load_who_table(path)
    assert "S = 0" in str(exc.value)


# ---------------------------------------------------------------------------
# Akuntansi baris
# ---------------------------------------------------------------------------

def test_every_label_row_ends_up_in_features_or_rejected(tmp_path):
    """Syarat paling penting: tidak ada baris yang hilang tanpa jejak.

    Karena itu test ini memakai main() penuh dengan gambar yang sengaja dibuat
    gagal dengan berbagai alasan (tidak ada wajah, file rusak, file hilang,
    label rusak).
    """
    import cv2

    from tests.conftest import render_frontal_face

    images = tmp_path / "images"
    images.mkdir(parents=True)

    # Wajah synthesik (deterministik, tanpa file pihak ketiga) -> harus diterima
    cv2.imwrite(str(images / "good.jpg"), render_frontal_face())

    # sisanya sengaja rusak
    cv2.imwrite(str(images / "nowface.jpg"), np.zeros((480, 640, 3), np.uint8))
    (images / "broken.jpg").write_bytes(b"bukan gambar")

    rows = [
        _row(subject_id="OK",   image_path="good.jpg"),
        _row(subject_id="NF",   image_path="nowface.jpg"),
        _row(subject_id="BRK",  image_path="broken.jpg"),
        _row(subject_id="MISS", image_path="tidak_ada.jpg"),
        _row(subject_id="BAD",  image_path="good.jpg", lila_cm="abc"),
    ]
    labels = _labels_csv(tmp_path, rows, images)

    # Pipeline sungguhan dipakai (model asli), tanpa stub apa pun.
    out_features = tmp_path / "features.csv"
    out_rejected = tmp_path / "rejected.csv"
    rc = ed.main([
        "--images", str(images), "--labels", str(labels),
        "--out-features", str(out_features), "--rejected", str(out_rejected),
    ])
    assert rc == 0

    feats = list(csv.DictReader(out_features.open()))
    rej = list(csv.DictReader(out_rejected.open()))

    assert len(feats) + len(rej) == len(rows), (
        f"{len(feats)} diterima + {len(rej)} ditolak != {len(rows)} baris input"
    )
    assert {f["subject_id"] for f in feats} == {"OK"}
    stages = {r["subject_id"]: r["stage"] for r in rej}
    assert stages == {"NF": "no_face", "BRK": "decode",
                      "MISS": "missing_file", "BAD": "label"}, stages
    assert all(r["reason"] for r in rej), "setiap penolakan wajib punya alasan"


def test_missing_required_column_fails_loudly(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("subject_id,image_path,lila_cm\nS1,a.jpg,12.5\n",
                      encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        ed.read_labels(labels)
    msg = str(exc.value)
    assert "berat_kg" in msg and "kategori" in msg


def test_empty_labels_fails_loudly(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("subject_id,image_path,kategori,lila_cm,berat_kg\n",
                      encoding="utf-8")
    with pytest.raises(SystemExit):
        ed.read_labels(labels)


def test_optional_age_and_sex_columns_may_be_absent(tmp_path):
    """usia_bulan/jenis_kelamin/panjang_tinggi_cm bersifat opsional."""
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "subject_id,image_path,kategori,lila_cm,berat_kg\n"
        "S1,a.jpg,balita,13.0,9.0\n", encoding="utf-8")
    rows = ed.read_labels(labels)
    assert len(rows) == 1
    assert rows[0]["kategori"] == "balita"
    assert rows[0]["lila_cm"] == "13.0"


# ---------------------------------------------------------------------------
# Skrip memakai pipeline, bukan implementasi ulang
# ---------------------------------------------------------------------------

def test_script_calls_existing_pipeline_functions(monkeypatch):
    """Guard: kalau ada yang menulis ulang landmark/quality sendiri, test ini
    gagal. Feature extraction WAJIB lewat app/pipeline.py."""
    called = []

    lm = np.zeros((478, 3), np.float64)
    lm[:, 0] = np.linspace(200, 400, 478)
    lm[:, 1] = np.linspace(150, 450, 478)
    monkeypatch.setattr(ed.pipeline, "detect_landmarks",
                        lambda img: (called.append("detect_landmarks"), (lm, 1))[1])
    monkeypatch.setattr(ed.pipeline, "sample_skin_lab",
                        lambda img, l: called.append("sample_skin_lab")
                        or np.array([60.0, 14.0, 18.0]))

    # QualityReport yang lolos, supaya flow benar-benar sampai ke stage fitur
    def fake_quality(*a, **k):
        called.append("assess_quality")
        return ed.pipeline.QualityReport(ok=True, score=1.0,
                                         checks={"sharpness": 500.0})
    monkeypatch.setattr(ed.pipeline, "assess_quality", fake_quality)

    # Stub setelah stage fitur: setiap stub mencatat namanya lalu mengembalikan
    # nilai minimal, sehingga urutan panggilan bisa diperiksa penuh.
    # Hanya combine_status (terakhir) yang menghentikan flow.
    def make_spy(name, result):
        def spy(*a, **k):
            called.append(name)
            return result
        return spy

    for name in ("extract_hair_features", "extract_sclera_features",
                 "extract_geometry_features"):
        monkeypatch.setattr(ed.pipeline, name, make_spy(name, {"available": False}))

    def make_score_spy(name):
        def spy(*a, **k):
            called.append(name)
            return SimpleNamespace(name=name, normalized=0.5,
                                   flagged=None, reason="stub")
        return spy

    for name in ("score_hair", "score_sclera", "score_geometry"):
        monkeypatch.setattr(ed.pipeline, name, make_score_spy(name))

    def stop(*a, **k):
        called.append("combine_status")
        raise ed.Rejected("stop", "sudah sampai combine_status")
    monkeypatch.setattr(ed.pipeline, "combine_status", stop)

    noisy = np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    with pytest.raises(ed.Rejected):
        ed.analyze_image(noisy)

    expected_order = [
        "detect_landmarks", "sample_skin_lab", "assess_quality",
        "extract_hair_features", "extract_sclera_features",
        "extract_geometry_features", "score_hair", "score_sclera",
        "score_geometry", "combine_status",
    ]
    assert called == expected_order, f"urutan/daftar panggilan beda: {called}"


def test_analyze_image_returns_all_expected_columns():
    """Kontrak skema features.csv harus stabil untuk downstream."""
    assert ed.OUTPUT_COLUMNS[: len(ed.META_COLUMNS)] == ed.META_COLUMNS
    assert ed.META_COLUMNS[:4] == ["subject_id", "image_path",
                                   "image_abs_path", "kategori"]
    for col in ("lila_cm", "berat_kg", "kategori_risiko", "label_underweight",
                "usia", "jenis_kelamin", "hair_available", "sclera_available",
                "geometry_available", "status_cv", "cv_score"):
        assert col in ed.OUTPUT_COLUMNS, f"kolom {col} hilang dari skema"
    # tidak boleh ada kolom duplikat
    assert len(ed.OUTPUT_COLUMNS) == len(set(ed.OUTPUT_COLUMNS))
    assert len(ed.REJECTED_COLUMNS) == len(set(ed.REJECTED_COLUMNS))
