"""
config.py
---------
Semua angka "magic number" dikumpulkan di sini: index landmark wajah, geometry
ratio, dan threshold klasifikasi per indikator.

--------------------------------------------------------------------------
PENAMAAN LANDMARK: PENTING
--------------------------------------------------------------------------
MediaPipe memberi label "LEFT"/"RIGHT" dari sudut pandang SUBYEK (subyek's left
= sisi kanan gambar). File ini memakai konvensi KAMERA (CAM_LEFT = sisi kiri
gambar / x kecil) karena itu yang dipakai saat memotong ROI.

Verified empirically terhadap models/face_landmarker.task (lihat
tests/test_landmarks.py::test_iris_centers_pair_with_correct_eye_ring):

    ring 33..133  -> "kiri gambar"  -> iris center 468  (MediaPipe: RIGHT_EYE)
    ring 263..362 -> "kanan gambar" -> iris center 473  (MediaPipe: LEFT_EYE)

Jadi kalau kamu mencocokkan index ini ke dokumentasi Google dengan label
LEFT/RIGHT, hasilnya akan TERBALIK. Nama di bawah pakai CAM_* supaya tidak
bisa tertukar.

Referensi topologi: mediapipe/python/solutions/face_mesh_connections.py
(FACEMESH_FACE_OVAL, FACEMESH_RIGHT_EYE, FACEMESH_LEFT_EYE, FACEMESH_*_IRIS).
"""

from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "face_landmarker.task"

# ---------------------------------------------------------------------------
# Landmark acuan oval wajah (semua anggota FACEMESH_FACE_OVAL)
# Diverifikasi: 10 = puncak dahi (bukan puncak kepala), 152 = dagu,
# 234 = tepi kiri gambar, 454 = tepi kanan gambar.
# CATATAN PENTING: 10 ada di face oval, artinya di garis rambut dahi.
# Region di atas 10 adalah DAHI + rambut, bukan murni rambut.
# Karena itu segmentasi rambut wajib lewat perbandingan warna kulit
# (lihat pipeline.segment_hair_mask), bukan crop kotak mentah.
# ---------------------------------------------------------------------------
FOREHEAD_TOP = 10
CHIN = 152
FACE_EDGE_CAM_LEFT = 234
FACE_EDGE_CAM_RIGHT = 454

# ---------------------------------------------------------------------------
# Landmark pipi & rahang
#
# PILI: 93/323 (bukan 50/280). Landmark 50/280 cuma ~0.67-0.70 dari lebar
# wajah penuh -- titik itu ada di area dekat hidung/bibir, BUKAN tulang pipi.
# Menggunakannya membuat rasio pipi/rahang selalu di bawah ambang sehingga
# SETIAP wajah ter-flag. Diukur pada wajah hasil render dan foto asli:
#     234/454 (tepi wajah)  : 1.00 x lebar wajah  <- titik terlebar
#     93/323  (tulang pipi)  : 0.98 x lebar wajah  <- dipilih
#     172/397 (sudut rahang) : 0.80 x lebar wajah  <- bigonial
#     50/280  (lama)         : 0.67 x lebar wajah  <- salah, bukan pipi
# Jadi rasio pipi/rahang yang terukur ~1.22 untuk kedua wajah.
# ---------------------------------------------------------------------------
CHEEK_CAM_LEFT = 93
CHEEK_CAM_RIGHT = 323
JAW_CAM_LEFT = 172
JAW_CAM_RIGHT = 397

# Ujung rahang (untuk rasio "cheek-to-chin width" ala literatur)
JAW_ANGLE_CAM_LEFT = 172
JAW_ANGLE_CAM_RIGHT = 397
CHIN_WIDTH_CAM_LEFT = 148
CHIN_WIDTH_CAM_RIGHT = 377

# Bibir bawah -> batas "lower face"
LOWER_LIP_BOTTOM = 17
LOWER_LIP_BOTTOM_ALT = 16

# ---------------------------------------------------------------------------
# Area mata. Dua ring dipisah supaya bisa dianalisis individual dan supaya
# bisa/sdkler tidak tercampur bridge hidung.
# ---------------------------------------------------------------------------
NOSE_TIP = 1

# Ring mata sisi KIRI gambar (MediaPipe RIGHT_EYE), center iris 468
EYE_CAM_LEFT_RING = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173,
]
IRIS_CAM_LEFT_CENTER = 468

# Ring mata sisi KANAN gambar (MediaPipe LEFT_EYE), center iris 473
EYE_CAM_RIGHT_RING = [
    263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398,
]
IRIS_CAM_RIGHT_CENTER = 473

# Ring IRIS (4 titik melingkari pupil, TANPA titik pusat).
# Dipakai untuk menghitung radius iris -> geometri annulus sklera.
# BUKAN ring mata (16 titik) yang jauh lebih lebar -- kalau keliru, radius iris
# terbaca terlalu besar dan annulus sklera jadi menelan area yang bukan mata.
IRIS_CAM_LEFT_RING = [469, 470, 471, 472]
IRIS_CAM_RIGHT_RING = [474, 475, 476, 477]

# ---------------------------------------------------------------------------
# Polygon warna kulit untuk (a) normalisasi white balance dan (b) segmentasi
# rambut. Dipilih hanya titik yang benar-benar kulit: dahi & pipi.
# Sengaja TIDAK memakai seluruh face oval karena memuat rambut, mata, mulut.
# ---------------------------------------------------------------------------
# Titik sampel kulit untuk estimasi warna kulit.
#
# Dipilih dari forehead, pelipis, dan pipi. Sengaja DIHINDARI:
#   - titik 10 (garis rambut): warna di sini sudah tercampur rambut, dan index
#     ini sekaligus dipakai sebagai anchor FOREHEAD_TOP -- kalau masuk daftar ini
#     dan dipakai ulang oleh fixture/segmen lain, anchor-nya tertimpa.
#   - kelopak mata (mis. 103, 54) dan titik yang terlalu dekat mata.
#   - hidung/lips (mis. 21, 162, 50, 280, 330, 347, 411): bayangan dan lipstik
#     membuat median warna kulit bias.
#   - titik pinggir oval wajah: paling sering kena background.
#
# Catatan: ini titik sampling BUKAN segmentasi kulit. Jika ada tahi lalat, acne,
# atau sunscreen, median 16 titik masih cukup tahan, tapi tidak robus.
SKIN_SAMPLE_POINTS = [
    151, 108, 337, 195,                # forehead tengah
    127, 356, 352, 425,                # pelipis kiri & kanan gambar
    101, 118, 117, 205,                # pipi atas
    187, 97, 2, 326,                   # pipi bawah / rahang atas
]

# Radius (px) untuk sampling kulit -- dikali scale wajah saat runtime
SKIN_SAMPLE_RADIUS_RATIO = 0.012

# Seberapa tinggi kotak SEARCH rambut diambil, relatif terhadap tinggi wajah
# (dahi -> dagu). Area di atas dahi = dahi atas + rambut, bukan murni rambut.
MARGIN_HAIR_REGION_RATIO = 0.30

# ---------------------------------------------------------------------------
# Kualitas foto / gate. Protocol: subjek menghadap kamera lurus, pół nossa
# menghadap ke sumber cahaya utama, jarak cukup agar wajah >= min_px.
#
# ANGKA INI BELUM DIKALIBRASI ke foto dunia nyata (tidak ada dataset
# malnutrisi berlabel yang bisa diakses). Nilai dipilih konservatif supaya
# lebih banyak foto ditolak daripada menghasilkan hasil menyesatkan.
# Kalau hasil testmu menunjukkan terlalu banyak false-reject, longgarkan
# DI SINI -- jangan longgarkan di pipeline.
# ---------------------------------------------------------------------------
QUALITY = {
    "max_upload_bytes": 12 * 1024 * 1024,   # 12 MB
    "max_dimension_px": 2000,                # sisi terpanjang, di-downscale
    "min_face_height_px": 160,               # landmark 10 -> 152 minimal
    "min_laplacian_variance": 40.0,          # deteksi blur (variance of Laplacian)
    "max_faces": 1,                          # >1 wajah -> tolak, jangan nebak subjek

    # Pencahayaan: L* mean kulit harus di rentang ini. Terlalu gelap = warna
    # tidak bisa dipercaya; terlalu terang = kontras warna hilang.
    "skin_l_min": 38.0,
    "skin_l_max": 94.0,

    # Pose. yaw_proxy dimensionless: jarak centroid iris dari tengah face-edge,
    # dibagi setengah lebar wajah. 0 = frontal, ~1 = profile.
    "max_yaw_proxy": 0.30,
    # Roll dalam derajat, dari garis penghubung kedua iris.
    "max_roll_deg": 15.0,

    # Pitch (kepala menengadah/mendongak), diproxikan dengan rasio
    # (jarak garis rambut -> garis iris) : (jarak garis iris -> dagu).
    # Wajah frontal apa adanya memberi ~0.40-0.43. Menengadah/mendongak
    # membuat jarak rambut ke iris berubah jauh lebih cepat sehingga rasio bergeser.
    # TIDAK terkalibrasi -- rentang dibuat lebar supaya hanya menahan pose
    # yang benar-benar ekstrem.
    "pitch_ratio_min": 0.24,
    "pitch_ratio_max": 0.62,
}

# ---------------------------------------------------------------------------
# Rasio geometri wajah
#
# LITERATUR: "Identification of malnutrition and prediction of BMI from facial
# images", IET Image Processing (2022) mendaftarkan rasio yang dipakai:
#   - width of cheek to chin
#   - ratio of width to upper face height
#   - lower face to face height ratio
#
# PENTING: angka potong di bawah adalah STARTING POINT. Rasio wajah lintas
# populasi sangat sempit, jadi sensitivitasnya rendah -- indikator ini
# menangkap kasus yang jelas-jelas abnormal saja, bukan skrining menyeluruh.
# ---------------------------------------------------------------------------
GEOMETRY = {
    # PENTING: ambang di bawah BELUM TERKALIBRASI. Tidak ada dataset malnutrisi
    # berlabel di repo ini, jadi nilainya hanya "tidak menandai wajah yang jelas
    # normal", bukan "menandai kasus malnutrisi". Jangan dipakai untuk klaim
    # sensitivitas/spesifisitas apa pun tanpa validasi eksternal.
    #
    # cheek/jaw = lebar pipi (93<->323) / lebar rahang (172<->397).
    # Nilai terukur pada dua wajah acuan: 1.224 (wajah hasil render) dan 1.234
    # (foto asli). Ambang 1.10 memberi margin ~10% sebelum mulai ter-flag.
    "cheek_to_jaw_ratio_min": 1.10,
    #
    # lower-face height = (bibir bawah -> dagu) / (garis rambut -> dagu).
    # Band LAMA (0.32-0.46) ~2.5x terlalu tinggi: band itu tidak mungkin
    # terpenuhi wajah mana pun -- acupolicy face thirds menempatkan region's
    #NOSEL->dagu sebagai sepertiga tinggi wajah, dan bibir bawah berada di
    # tengahnya, jadi chin->lower-lip ~1/6 tinggi wajah (~0.17).
    # Nilai terukur: 0.217 (render) dan 0.131 (foto asli, anak perempuan 3/4
    # view sehingga bagian bawah wajah terproyeksi lebih pendek).
    "lower_face_height_ratio_min": 0.11,
    "lower_face_height_ratio_max": 0.24,
}

# ---------------------------------------------------------------------------
# Indikator warna
#
# Semua fitur dihitung dalam CIELAB dan dinormalisasi terhadap warna kulit
# subjek sendiri. Alasan: nilai HSV/grayscale ABSOLUT tidak bisa dipakai untuk
# lintas subjek -- anak berkulit gelap dengan pencahayaan berbeda akan menghasilkan
# angka yang sama sekali berbeda, sementara fitur klinisnya sama.
# Normalisasi terhadap kulit sendiri membuat fitur relatif & lintas-subjek-safe.
# ---------------------------------------------------------------------------

# --- 1. Rambut: hipopigmentasi (proxy defisiensi protein) ---
# Landasan klinis: kwashiorkor Penurunan melanin rambut membuat rambut
# "reddish yellow to white" (Medscape, Protein-Energy Malnutrition) dan
# "gray-blond" (foto CDC). Studies: "Childhood malnutrition is associated with
# a reduction in the total melanin content of scalp hair" (Br J Nutr).
#
# Jadi sinyal yang benar: L* rambut NAIK mendekati kulit DAN chroma rambut
# TURUN tajam. BUKAN "hue kemerahan" seperti di versi lama.
#
# BATAASAN KERAS yang harus kamu sebut di laporan:
#   (a) tool ini hanya bisa melihat "rambut pucat/low-melanin SAAT INI".
#       Tanda klinis aslinya adalah PERUBAHAN warna dari kondisi normal
#       subjek. Tanpa foto baseline, rambut pirang asli tidak bisa dibedakan
#       dari rambut yang hipopigmen karena malnutrisi.
#   (b) karena itu indikator ini memakai bobot TERENDAH dan TIDAK boleh
#       dipakai sendiri untuk menyimpulkan diagnosis.
HAIR = {
    # jarak L* kulit - L* rambut. Rambut hipopigmen -> selisih mengecil.
    # "l_delta_flag" adalah titik yang di-flag, "l_delta_max" adalah titik
    # severity 0 (fully normal, dipakai untuk rentang severity 0..1).
    # Harus l_delta_flag < l_delta_max.
    "l_delta_max": 24.0,
    "l_delta_flag": 12.0,

    # chroma rambut / chroma kulit. Rambut hitam sudah low-chroma ->
    # angka kecil bisa normal, dipakai hanya sebagai konfirmasi arah.
    "chroma_ratio_max": 0.60,   # severity 0
    "chroma_ratio_flag": 0.30,  # di-flag

    # haircut minimum: kalau rambut terlalu pendek/tidak ada di region atas
    # dahi, indikator dinonaktifkan (bukan dianggap normal).
    "min_foreground_ratio": 0.12,
    # toleransi klasifikasi kulit (jarak LAB) saat memisahkan rambut dari kulit
    "skin_distance_tolerance": 12.0,
}

# --- 2. Kepucatan konjungtiva/sklera (proxy anemia) ---
# Landasan: konjungtiva pucat adalah tanda anemia non-invasif yang murah.
# BUKAN sklera yang paling tepat -- hanya perlu membuka kelopak, tidak terlihat
# di foto frontal. Kita mengukur area mata yang terekspos sebagai proxy dan
# MENYATAKANNYA secara eksplisit di respons API.
# Literatur:-segmentasi CIELLAB (Williams Asare 2024; Wulandari 2024 IEEE).
#   Conjunktival pallor sensitivitas untuk anemia RINGAN <69%, untuk anemia
#   SEDANG/BERAT >=70% -> alat ini hanya masuk akal untuk moderate/severe.
SCLERA = {
    # a* sklera / a* kulit. Darah memberi a* positif pada kulit; pada sklera
    # sehat a* relatif lebih rendah. Pucat -> rasio naik mendekati kulit.
    #
    # a_star_ratio_max = ambang FLAG.
    # a_star_ratio_normal = titik "sehat" yang jadi baseline severity, supaya
    #   subyek yang mendekati abnormal tetap punya skor di atas 0 (bukan 0.0
    #   yang identik dengan subyek sehat). Tanpa ini, skor agregat kehilangan
    #   informasi "mendekati abnormal" dan dua kondisi berbeda bisa mendapat
    #   status yang sama.
    "a_star_ratio_max": 0.80,
    "a_star_ratio_normal": 0.50,
    # Guard: kalau a* kulit <= 0, sample warna tidak valid -> fail closed
    "min_skin_a_star": 1.0,
    # Penyebut rasio DIBATAS di bawah. a* kulit bisa mendekati 0 pada kulit
    # netral/keputihan, dan membagi dengan angka kecil membuat rasio meledak:
    # a*_sklera=-1 dengan a*_kulit=0.5 menghasilkan rasio -2, yang akan
    # dihitung sebagai kepucatan berat. Floor ini membuat penyebut minimal
    # cukup besar sehingga rasio tidak berayun tanpa batas.
    "skin_a_star_floor": 10.0,
    # minimal pixel terpakai, di bawah ini ROI tidak bisa dipercaya
    "min_pixels": 150,
    # NAHKAN kalau ada foto kacamata (refleksi lenses bikin a* sky-high)
    "warn_if_glass_likely": True,
}

# ---------------------------------------------------------------------------
# Bobot indikator untuk agregasi skor (lihat pipeline.combine_status)
# Satu indikator kuat tidak boleh "tertutupi" indikator lain yang normal,
# jadi ini dipakai sebagai kontinu, bukan sekadar hitungan flag.
#
# Bobot rambut sengaja paling kecil -- lihat catatan BATAASAN KERAS di HAIR.
# ---------------------------------------------------------------------------
WEIGHTS = {
    "rambut": 0.5,
    "kepucatan_mata": 1.0,
    "geometri_wajah": 1.0,
}

# Ambang skor agregat -> status (0..1). Semua TIDAK tervalidasi klinis.
STATUS_THRESHOLDS = {
    "sedang": 0.45,
    "tinggi": 0.70,
}

# ---------------------------------------------------------------------------
# NOTE: versi lama memakai hair_reddish_hue_max/min_wrap, hair_saturation_min,
# hair_value_min, cheek_width_ratio_min, sclera_brightness_min.
# Semuanya DIHAPUS karena tidak valid:
#  - *_hue_*: salah arah (miss "reddish yellow", FP pada rambut Asia normal)
#  - *_brightness_*: grayscale brightness absolute tidak lintas-subjek aman
#  - cheek_width_ratio_min: rasio antar-landmark tetap -> hampir konstan,
#    mengukur hampir tidak apa-apa
# Jangan dikembalikan tanpa landasan klinis yang jelas.
# ---------------------------------------------------------------------------