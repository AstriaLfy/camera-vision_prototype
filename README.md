# Prototype: Skrining Dini Malnutrisi & Anemia dari Foto Wajah

Prototype REST API yang menilai foto wajah berfrontal dan mengembalikan
**indicator skrining** untuk hipopigmentasi rambut, kepucatan, dan perubahan
proorsi wajah.

> **Alat bantu SKRINING, bukan alat diagnosis.**
> Output ini berbasis rule-based threshold, bukan model yang dilatih dari
> dataset malnutrisi. **Selalu konfirmasi dengan pengukuran MUAC dan/atau tes
> hemoglobin.** Foto wajah tidak dapat menggantikan pemeriksaan
> anthropometri. Lihat [Batasan klinis](#batasan-klinis-jujur) — bagian itu
> bukan formalitas.

## Status validasi

Semua ambang di `app/config.py` **belum terkalibrasi**. Repo ini tidak
memuat dataset foto malnutrisi/anemia berlabel, sehingga:

- Angka ambang hanya berfungsi sebagai "jangan menandai wajah yang jelas normal".
- **Tidak boleh** dipakai untuk klaim sensitivitas, spesifisitas, atau
  klasifikasi status malnutrisi.
- Sebelum dipakai untuk keputusan klinis apa pun, ambang wajib divalidasi
  ulang terhadap dataset berlabel.

Test suite (533 test: 530 lulus + 3 xfailed) memverifikasi bahwa logika berjalan benar pada landmark
yang dikembalikan model asli — **bukan** bahwa angka klinisnya akurat.

## Struktur

```
cv-prototype/
├── app/
│   ├── main.py        # buat app + CORS + handler error + daftarkan router
│   ├── routers/       # satu APIRouter per domain (lihat app/routers/)
│   │   ├── screening.py  # POST /api/v1/screening/analyze, /api/v1/debug-landmarks
│   │   └── health.py     # GET /api/v1/health, GET /
│   ├── content.py     # SATU-SATUNYA sumber teks yang tampil ke pengguna
│   ├── risk_status.py # status risiko LiLA + indikator terukur (Pydantic)
│   ├── indicators.py  # fitur mentah -> baris indikator visual deskriptif
│   ├── thresholds.py  # SATU-SATUNYA sumber angka ambang LiLA & usia balita
│   ├── settings.py    # mode CV, batas upload, CORS origin (env / .env)
│   ├── deps.py        # dependencies lintas endpoint (placeholder autentikasi)
│   ├── errors.py      # format error seragam {"error": {"code", "message"}}
│   ├── pipeline.py    # deteksi wajah -> landmark -> fitur -> scoring
│   └── config.py      # SEMUA index landmark & angka threshold ada di sini
├── tests/
│   ├── conftest.py            # renderer wajah frontal + landmark sintetis
│   ├── test_landmarks.py      # landmark dari model asli
│   ├── test_quality_gate.py   # gate protokol foto
│   ├── test_segmentation.py   # pemisahan rambut & ROI sklera
│   ├── test_color.py          # scoring & agregasi status
│   ├── test_api.py            # kontrak HTTP dasar + helper EXIF/downscale
│   ├── test_health.py         # GET /api/v1/health
│   ├── test_error_format.py   # format error seragam (404/415/422/500/503)
│   ├── test_cors.py           # CORS ketat secara default
│   ├── test_settings.py       # konfigurasi env / .env
│   ├── test_content.py        # teks konten + pelapor placeholder
│   ├── test_risk_status.py    # status risiko: batas, data tidak layak, CV
│   ├── test_indicators.py     # indikator visual: pemetaan, NaN, ambang
│   ├── test_screening.py      # POST /api/v1/screening/analyze: kontrak respons
│   ├── test_extract_dataset.py # skrip ekstraksi (label LiLA)
│   ├── test_eda.py             # skrip EDA
│   └── test_end_to_end.py     # pipeline penuh dengan model
├── scripts/
│   ├── extract_dataset.py     # batch ekstraksi fitur + label LiLA dari dataset
│   └── eda.py                 # analisis eksplorasi atas features.csv
├── data/                      # DATA ASLI - di-ignore git
│   ├── images/                # foto subjek
│   ├── labels.csv             # isi labels_template.csv (skema LiLA: kategori, lila_cm, ...)
│   ├── labels_template.csv    # template + 2 baris CONTOH (hapus sebelum pakai)
│   └── features.csv           # hasil ekstraksi (dihasilkan skrip)
├── reports/                   # DATA ASLI - di-ignore git
│   ├── rejected.csv           # foto yang tidak lolos + alasannya
│   └── eda/                   # keluaran scripts/eda.py (summary.md, tabel, plot)
├── data_synthetic/            # data BUATAN untuk menguji pipeline (lihat README-nya)
├── reports_synthetic/         # keluaran EDA atas data sintetis
├── models/
│   └── face_landmarker.task   # hasil download_model.py
├── download_model.py          # unduh model (sekali saja)
├── .env.example               # templat konfigurasi (salin jadi .env kalau perlu)
├── requirements.txt
└── README.md
```

## Cara menjalankan

Runtime yang sudah diverifikasi: **Python 3.14** (butuh `numpy>=2.3.2` untuk
wheel cp314).

### 1. Install dependencies

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
```

### 2. Unduh model (sekali saja, butuh internet)

```bash
./venv/bin/python download_model.py
```

Menaruh `face_landmarker.task` ke `models/`. Kalau model tidak ada, endpoint
mengembalikan **503** dengan pesan yang jelas — bukan crash, dan tanpa
membocorkan path server.

### 3. Jalankan test

```bash
./venv/bin/python -m pytest tests/ -q
```

Test yang butuh model otomatis `skip` (bukan `fail`) kalau model belum ada.

### 4. Jalankan server

```bash
./venv/bin/python -m uvicorn app.main:app --port 8000
```

Dokumentasi interaktif: <http://127.0.0.1:8000/docs>

Cek cepat layanan hidup (berisi status + versi app):

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Konfigurasi (mode CV, batas upload, origin CORS) dibaca dari environment
atau file `.env` — lihat `.env.example`. Tanpa `.env`, semuanya memakai
default yang ketat: **CORS tidak mengizinkan origin mana pun** (`ALLOWED_ORIGINS`
kosong), bukan `"*"`.

### 5. Ekstraksi dataset (opsional)

Untuk membuat CSV fitur dari sekumpulan foto, pakai `scripts/extract_dataset.py`.
Skrip ini **memakai** fungsi-fungsi di `app/pipeline.py` dengan urutan yang sama
dengan endpoint, bukan menyalin logikanya — jadi pipeline dan dataset tidak
bisa berbeda versi.

```bash
./venv/bin/python scripts/extract_dataset.py \
    --images data/images \
    --labels data/labels.csv \
    --out-features data/features.csv \
    --rejected reports/rejected.csv
```

`labels.csv` minimal berisi `subject_id, image_path, kategori, lila_cm,
berat_kg`. Kolom `kategori` (isi `balita`, `ibu_hamil`, atau `ibu_menyusui`)
yang menentukan aturan labelnya. Untuk balita, `usia` **wajib** dalam **bulan**
(atau pakai kolom `usia_bulan`) dan `jenis_kelamin` juga wajib; untuk ibu,
`usia` diisi dalam **tahun** dan `jenis_kelamin` boleh kosong. `image_path`
relatif terhadap `--images`.

#### Template CSV

Salin `data/labels_template.csv` jadi `data/labels.csv`, lalu isi.

Template itu berisi **2 baris contoh** dengan `subject_id` dan `image_path`
berawalan `CONTOH-` (`CONTOH-001.jpg`, `CONTOH-002.jpg`). **Hapus kedua baris
itu sebelum pakai** — kalau dibiarkan, dua subjek fiktif ikut masuk
`data/features.csv` dan merusak hitungan label. Awalan `CONTOH-` sengaja
dipakai supaya baris contoh selalu terlihat, bukan tercampur diam-diam dengan
data asli.

> Baris komentar diawali `#` sengaja **tidak** dipakai di template. `csv.DictReader`
> (yang dipakai `scripts/extract_dataset.py`) membaca baris pertama sebagai
> header, jadi satu baris `#` di atas header membuat skrip berhenti dengan pesan
> "kolom wajib tidak ada". Kalau ingin berkomentar, taruh di `README.md` seperti
> paragraf ini.

#### Tabel WHO (dipertahankan, bukan sumber label)

Label dihitung **hanya** dari LiLA — lihat bagian berikut. Tabel
**WHO 2007 BMI-for-age (5–19 tahun)** di bawah ini tetap dimuat dan diuji
skrip, tetapi tidak ikut menetapkan label apa pun:

```
data/who/bmi_for_age_boys.csv
data/who/bmi_for_age_girls.csv
```

Format yang dibutuhkan: kolom `Month, L, M, S, SD4neg, SD3neg, SD2neg, SD1neg,
SD0, SD1, SD2, SD3, SD4`, mencakup bulan `61`–`228`. Kalau file itu tidak ada
atau rusak (kolom wajib hilang, ada `NaN` di `L/M/S`, bulan berurutan dengan
celah, atau angka tidak terbaca sebagai desimal) `tests/test_extract_dataset.py`
gagal dengan pesan yang jelas — test sengaja membaca file aslinya, bukan angka
hafalan. Alur ekstraksi sendiri tetap jalan karena tidak memakai tabel ini.

#### Cara label ditetapkan

Satu-satunya sumber label adalah **LiLA (lingkar lengan atas)** dalam cm.
BMI, z-score WHO, dan ambang `--bmi-threshold` **tidak dipakai sama sekali**,
termasuk untuk ibu hamil.

Aturannya bergantung pada kolom `kategori`:

| `kategori` | `label_method` | `label_underweight = 1` bila | `kategori_risiko` |
| --- | --- | --- | --- |
| `balita` | `lila_balita` | LiLA < 12.5 cm | merah < 11.5 · kuning 11.5–<12.5 · hijau ≥ 12.5 |
| `ibu_hamil` | `lila_ibu` | LiLA < 23.5 cm | merah < 23.5 · hijau ≥ 23.5 |
| `ibu_menyusui` | `lila_ibu` | LiLA < 23.5 cm | merah < 23.5 · hijau ≥ 23.5 |

Lebih rincinya:

1. **`kategori` wajib** berisi `balita`, `ibu_hamil`, atau `ibu_menyusui`.
   Huruf besar/kecil serta tanda spasi dan hubung ditoleransi
   (`ibu hamil` = `ibu-hamil` = `ibu_hamil`); selain itu baris ditolak, bukan
   ditebak.
2. **`lila_cm` wajib** dan harus angka berhingga dalam rentang wajar
   **5–50 cm**. Nilai kosong, `NaN`, `inf`, atau di luar rentang ditolak —
   LiLA yang salah ketik tidak boleh diam-diam berubah jadi label.
   `berat_kg` juga wajib dan harus lebih besar dari nol.
3. **Balita**: `usia` dibaca sebagai **bulan** dan harus **6–59 bulan**.
   Di luar rentang itu baris ditolak dengan tahap `label` dan alasan persis
   `LiLA tidak valid di luar 6-59 bulan`. Usia 5 bulan dan 60 bulan **tidak**
   ditempel ke bulan terdekat. `jenis_kelamin` wajib terisi untuk balita.
4. **Ibu hamil / menyusui**: `usia` diisi dalam **tahun**, disimpan apa adanya
   (tidak dikalikan 12) dan tidak memengaruhi label. `jenis_kelamin` boleh
   kosong. LiLA ibu hamil dinilai langsung dari LiLA, tidak lewat BMI.
5. **`usia_bulan` didahulukan** kalau kolomnya terisi, jadi `usia_bulan = 60`
   tetap ditolak walaupun `usia` berisi angka lain. Sebaliknya, kalau
   `usia_bulan` kosong, `usia` balita langsung dipakai sebagai bulan — tidak
   ada perkalian `* 12`.
6. **`panjang_tinggi_cm`** (opsional) disimpan apa adanya sebagai kolom
   pendukung; z-score-nya sengaja belum dihitung.

Yang dilakukan per foto:

1. Baca `labels.csv`, validasi `kategori` + LiLA + usia, lalu tetapkan label
   memakai aturan di atas.
2. Jalankan pipeline: landmark → sampling warna kulit → quality gate →
   fitur rambut/sklera/geometri → scoring.
3. Simpan semua fitur + LiLA + kategori risiko + label + usia + jenis kelamin
   ke `data/features.csv`.

Kolom tambahan di `data/features.csv` selain kolom fitur: `kategori`, `usia`,
`usia_bulan`, `jenis_kelamin`, `lila_cm`, `berat_kg`, `panjang_tinggi_cm`,
`kategori_risiko`, `label_method`, `label_underweight`.

Tidak ada foto yang dibuang diam-diam. Yang gagal — label rusak, kategori
tidak dikenali, LiLA di luar rentang, usia balita di luar 6–59 bulan, file
hilang, gambar rusak, wajah tidak terdeteksi, atau gagal quality gate —
dicatat di `reports/rejected.csv` lengkap dengan tahap (`stage`) dan
alasannya. Ringkasan dicetak di akhir: jumlah diterima/ditolak, metode label,
distribusi label dan kategori risiko, ketersediaan tiap kelompok fitur, dan
alasan penolakan yang paling sering.

Opsi lain: `--limit N` untuk smoke test.

### 6. EDA (opsional)

Untuk memahami `features.csv` sebelum dipakai untuk apa pun:

```bash
./venv/bin/python scripts/eda.py \
    --input data/features.csv \
    --outdir reports/eda/
```

Ringkasnya: jumlah baris, distribusi label, NaN per kolom (kolom NaN > 30%
dikeluarkan dari analisis), boxplot per label, korelasi Spearman fitur vs LiLA,
Mann-Whitney U + koreksi Benjamini-Hochberg + effect size rank-biserial,
scatter 5 fitur terkuat vs LiLA, matriks korelasi antar fitur (|rho| > 0.9
ditandai redundan), dan analisis usia (korelasi parsial fitur-LiLA yang
dikontrol usia, untuk melihat fitur yang sinyalnya cuma ngikut umur).

Hasilnya ada di `reports/eda/summary.md`, plus tabel CSV dan plot PNG.

Yang penting dibaca dari sana: **langkah 7 sering membatalkan langkah 4.**
Kalau sebuah fitur terlihat "signifikan" antara label 0 vs 1 tapi korelasinya
dengan LiLA hilang begitu saja setelah usia dikontrol, fitur itu tidak sedang
mengukur status gizi — fitur itu cuma mengukur umur.

Satu peringatan yang selalu dicetak di `summary.md`: **ambang LiLA berbeda
per kategori** (balita < 12.5 cm, ibu hamil/menyusui < 23.5 cm). Kalau
`features.csv` mencampur balita dan ibu dalam satu analisis, labelnya tidak
sebanding — cek distribusi `kategori` dulu sebelum memakai angka-angka di sana.
Karena itu pula kolom `usia_bulan` (bulan untuk balita) dan `usia` (bulan
balita / tahun ibu) dikeluarkan dari daftar fitur: satuan campur tidak bisa
jadi kovariat yang adil.

Skrip ini **tidak melatih model apa pun**. Tidak ada split train/test, tidak ada
prediksi. Angka-angkanya menggambarkan dataset ini saja, bukan bukti kausal.

## Deploy dengan Docker

Stack: `api` (FastAPI/uvicorn) + `db` (PostgreSQL 16) + `migrate` (Alembic) +
`ngrok` (profil `tunnel`) + `db-test` (Postgres sekali pakai untuk test). Semua
berada di satu jaringan internal bernama (`cv-prototype-internal`), sehingga
service baru (auth, profil, subjek, sync) cukup ditambahkan sebagai blok baru di
`docker-compose.yml` tanpa mengubah service yang sudah ada.

### Prasyarat

- Docker Engine + Docker Compose v2 (teruji: Docker 29, Compose v5).
- `.env` (lihat langkah berikut). Tanpa `.env`, semua default aman tetap dipakai.
- Model FaceLandmarker **opsional**: kalau tidak ada, layanan tetap hidup dan
  `/api/v1/health` melaporkan `model_available=false` beserta `model_message`.
- Authtoken ngrok (hanya bila ingin menjalankan tunnel).

### 1. Siapkan `.env`

```bash
cp .env.example .env
```

Isi minimal yang perlu diganti: `POSTGRES_PASSWORD` (jangan biarkan
`ganti-ini`) dan `NGROK_AUTHTOKEN` bila memakai tunnel. `DATABASE_URL` default
sudah menunjuk ke service `db`. `.env` **jangan** di-commit (sudah di
`.gitignore`).

### 2. Build & jalankan

```bash
make up            # atau: docker compose up -d --build
```

Urutannya: `db` sehat -> `migrate` (`alembic upgrade head`) sukses -> `api`
start. Port API **hanya** dipetakan ke `127.0.0.1:8000` (host), bukan `0.0.0.0`,
supaya tidak terbuka ke LAN.

Cek status: `docker compose ps` dan `make logs`.

### 3. Menjalankan dengan tunnel ngrok

```bash
# NGROK_AUTHTOKEN sudah diisi di .env
make up-tunnel      # atau: docker compose --profile tunnel up -d --build
```

Profil `tunnel` **tidak** jalan kecuali diminta. Service `ngrok` memakai image
resmi `ngrok/ngrok` dan meneruskan ke `api:8000` dengan domain statis dari
`NGROK_DOMAIN`. **Flag yang dipakai: `--url`** (sintaks ngrok agent v3;
`--domain` sudah deprecated). Authtoken diambil dari env `NGROK_AUTHTOKEN`,
tidak pernah ditulis ke file yang di-commit.

Dashboard inspeksi ngrok (<http://127.0.0.1:4040>) hanya bind ke localhost.

### 4. Cek health (lokal & publik)

```bash
curl http://127.0.0.1:8000/api/v1/health
curl https://jaws-hurler-unranked.ngrok-free.dev/api/v1/health
```

Untuk klien non-browser yang menembus halaman peringatan ngrok Free, kirim
header `ngrok-skip-browser-warning` (nilainya bebas, mis. `1`):

```bash
curl -H "ngrok-skip-browser-warning: 1" \
  https://jaws-hurler-unranked.ngrok-free.dev/api/v1/health
```

### 5. Migrasi database

```bash
make migrate        # alembic upgrade head
```

`alembic/` masih **kerangka placeholder** (`alembic.ini`, `env.py`, folder
`versions/` kosong) -- repo ini belum punya skema tabel. Jangan mengarang
migrasi sebelum model fitur (auth, profil, subjek, sync) disepakati. URL
database dibaca `alembic/env.py` dari `DATABASE_URL`, jadi source-nya tunggal.

### 6. Backup & restore volume Postgres

```bash
make backup                       # -> backup_<tanggal>.sql (pg_dump)
make restore FILE=backup_20260101_120000.sql
```

Backup level volume (tanpa server berjalan):

```bash
docker run --rm -v cv-prototype-pgdata:/data -v "$PWD:/backup" alpine \
  tar czf /backup/pgdata.tgz -C /data .
```

### 7. Test

```bash
make test           # test suite terhadap service db-test (Postgres terpisah, tanpa volume)
```

Test menjalankan image aplikasi dengan `tests/` dan `app/` di-bind-mount, dan
`DATABASE_URL` diarahkan ke `db-test`.

### 8. Akses port Postgres untuk alat lokal

Default `db` tidak mempublikasikan port. Bila perlu `psql`/DBeaver dari host:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### Model MediaPipe

Dua cara menyediakan model (jangan mengarang URL):

- **Volume (disarankan untuk dev)**: taruh `face_landmarker.task` di `models/`
  (hasil `python download_model.py`); compose sudah me-mount `./models` ke
  `/app/models` (read-only).
- **Build arg**: `docker compose build --build-arg MODEL_URL=<url-resmi>`
  (default kosong). Isi hanya bila punya URL model resmi.

### Keamanan & catatan

- **API ini terbuka tanpa autentikasi.** Selama `app/deps.py` masih placeholder,
  siapa pun yang punya URL publik bisa memanggilnya. Pakai **data dummy** untuk
  demo/pengujian; jangan kirim data pasien nyata.
- Endpoint debug `POST /api/v1/debug-landmarks` **hanya terdaftar** bila
  `ENABLE_DEBUG_ENDPOINTS=true` (default `false`).
- Dokumentasi interaktif (`/docs`, `/redoc`, `/openapi.json`) aktif di
  `APP_ENV=dev` (default) dan mati di `APP_ENV=prod`; bisa dipaksa lewat
  `ENABLE_DOCS=true|false`. Saat mati, `/openapi.json` ikut mati.
- **CORS** mengizinkan origin dari `ALLOWED_ORIGINS` (default kosong = ketat).
  **Aplikasi mobile native tidak terikat CORS** -- daftar origin hanya relevan
  untuk klien berbasis browser, jadi tiap origin web baru harus didaftarkan.
  Header `Authorization` dan `ngrok-skip-browser-warning` sudah diizinkan.
- `uvicorn` dijalankan dengan `--proxy-headers`, tetapi
  `--forwarded-allow-ips` **dibatasi ke subnet jaringan compose**
  (`172.28.0.0/16`), bukan `*`, supaya header `X-Forwarded-*` dari sumber lain
  tidak dipercaya.
- Batas upload mengikuti `MAX_UPLOAD_MB` di settings (dicek aplikasi). uvicorn
  tidak punya batas body sendiri; kalau nanti ditambah reverse proxy, samakan
  batasnya.
- Log streaming ke console container (stdout/stderr, ditangkap Docker) tanpa
  data pribadi (nama, foto, nilai ukur, email, token).
- **`opencv-python-headless` tidak dipakai.** `requirements.txt` memakai
  `opencv-contrib-python`, yang ditarik oleh `mediapipe`; menambahkan
  `opencv-python-headless` membuat dua distribusi OpenCV terpasang bersamaan
  dan `import cv2` bisa gagal. Jangan menggantinya tanpa menguji ulang.

## Contoh pemakaian

`POST /api/v1/screening/analyze` (multipart/form-data): data antropometri +
foto wajah. Status risiko **murni dari LiLA**; foto hanya menambah data
pendukung visual dan **tidak pernah** mengubah status.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/screening/analyze \
  -F "kategori=ibu_hamil" \
  -F "lila_cm=22.0" \
  -F "berat_kg=55.0" \
  -F "foto=@foto_wajah.jpg"
```

Field form:

| Field | Wajib | Keterangan |
|-------|-------|------------|
| `foto` | ya | JPEG/PNG (batas tipe & ukuran dari settings) |
| `kategori` | ya | `balita` \| `ibu_hamil` \| `ibu_menyusui` |
| `lila_cm` | ya | Lingkar lengan atas (cm), > 0 |
| `berat_kg` | ya | Berat badan (kg), > 0 |
| `usia_bulan` | tidak | Usia balita (bulan) |
| `usia` | tidak | Usia ibu (tahun); **tidak dipakai** menghitung status |
| `jenis_kelamin` | tidak | data subjek |
| `panjang_tinggi_cm` | tidak | data subjek |

Respons 200 (urutan field persis seperti ini):

```json
{
  "ringkasan": {
    "status": "merah",
    "label": "Merah - perlu ditindaklanjuti",
    "deskripsi": "Perlu ditindaklanjuti: hasil skrining LiLA berada di bawah ambang aman."
  },
  "subjek": { "kategori": "ibu_hamil", "kelompok": "ibu" },
  "pengukuran": [
    {
      "kunci": "lila",
      "label": "LiLA - risiko KEK (Kekurangan Energi Kronis)",
      "nilai": 22.0,
      "satuan": "cm",
      "badge": "merah",
      "keterangan": "22.0 cm, di bawah 23.5 cm",
      "sumber": "terukur"
    },
    {
      "kunci": "berat_badan",
      "label": "Berat badan",
      "nilai": 55.0,
      "satuan": "kg",
      "badge": null,
      "keterangan": null,
      "sumber": "terukur"
    }
  ],
  "data_pendukung": [
    {
      "kunci": "sclera_a_star",
      "kelompok": "konjungtiva",
      "label": "Nilai a* sklera",
      "nilai": -1.06,
      "satuan": "a*",
      "keterangan": "Nilai a* sklera: -1.06",
      "sumber": "visual_eksperimental",
      "badge": null,
      "badge_label": "Eksperimental"
    }
  ],
  "kualitas_foto": { "ok": true, "pesan": [] },
  "rekomendasi": ["Konfirmasi dengan pengukuran LiLA dan/atau pemeriksaan lanjutan. ..."],
  "alasan": ["LiLA 22.0 cm < 23.5 cm"],
  "disclaimer": ["...bukan diagnosis...", "...indikator visual bersifat eksperimental..."],
  "meta": { "cv_mode": "indicator_only", "versi": "0.1.0" }
}
```

- `pengukuran[]` berasal dari `risk_status.indikator_terukur`; `sumber` selalu
  `"terukur"`. **Hanya** item inilah yang boleh punya `badge`.
- `data_pendukung[]` berasal dari `indicators.build_indicators`; semuanya
  `sumber="visual_eksperimental"`, `badge=null`, dan diberi label eksperimental.
  Kosong kalau `CV_MODE=off` atau foto tidak layak.
- `status_cv` dan skor pipeline **tidak pernah** dihitung maupun dikirim.

### Kode status

| Kode | Arti |
|------|------|
| 200 | Permintaan diproses. Foto tidak layak / tanpa wajah **tetap 200**, dengan `kualitas_foto.ok=false` dan `data_pendukung=[]`; status LiLA tetap dikembalikan |
| 400 | File bukan gambar valid |
| 413 | File melebihi batas ukuran |
| 415 | Content-Type tidak didukung |
| 422 | Input tidak valid (field hilang/salah tipe/kategori tak dikenal) |
| 500 | Error tak terduga di server (pesan generik, tanpa detail internal) |
| 501 | `CV_MODE=model` — mode model belum diimplementasikan |
| 503 | Model FaceLandmarker belum tersedia di server |

Foto tidak layak **bukan** lagi error HTTP: pesan yang bisa dibaca pengguna ada
di `kualitas_foto.pesan` (teks dari `app/content.py`), bukan di `reason`/`hint`.

### Privasi (stateless)

Endpoint memproses foto **hanya di memori**: tidak ada foto, nama, atau nilai
pengukuran yang ditulis ke disk, database, atau log. Test
(`tests/test_screening.py`) memverifikasi tidak ada file yang dibuka untuk
ditulis selama request dan tidak ada data subjek yang muncul di log.

### Teks yang tampil ke pengguna

Semua disclaimer, label status, rekomendasi, dan panduan foto ada di
**`app/content.py`** — satu-satunya tempat teks produk ditulis. Endpoint
memanggil `content.get_text(key)` / `content.get_recommendation(status, kelompok)`;
menulis literal teks di `app/routers/` dianggap pelanggaran dan dicover test.

Kunci hilang melempar `ContentKeyError` (pesan menyebut kunci +
alternatifnya), bukan mengembalikan string kosong.

Rekomendasi per status × kelompok (`balita` / `ibu`) masih **placeholder**
bertanda `[PERLU DIVERIFIKASI AHLI GIZI]` — belum diisi karena belum ada
pedoman Kemenkes/WHO di repo ini. Test `test_content.py` **melaporkan**
(bukan gagal) berapa banyak yang masih placeholder.

### Status risiko (murni LiLA)

**`app/risk_status.py`** mengubah input mentah (`kategori`, `lila_cm`,
`berat_kg`, `usia_bulan`) menjadi model Pydantic `RiskStatus`:
`status_antropometri`, `status_akhir`, `kelompok`, `rekomendasi`,
`label_status`, `alasan[]`, `disclaimer`, `indikator_terukur[]`.

- **`status_akhir` murni dari LiLA** — selalu sama dengan
  `status_antropometri`. Modul ini **tidak punya** `cv_flag`, tidak punya
  `peringatan_visual`, dan tidak punya status `hijau_pantau`.
  `status_cv` dari `app/pipeline.py` (`normal`/`sedang`/`tinggi`/
  `tidak_valid`) tidak dipetakan ke status risiko apa pun — ada test yang
  memeriksa modul ini bebas nama-nama itu di luar dokumentasinya.
- **Batas** — balita (6–59 bln): `<11.5` merah, `11.5–<12.5` kuning,
  `≥12.5` hijau. Ibu hamil/menyusui: `<23.5` merah, `≥23.5` hijau
  (tanpa kategori kuning). Angkanya **tidak diduplikasi**: ada di
  **`app/thresholds.py`** dan diimpor oleh `app/risk_status.py` **dan**
  `scripts/extract_dataset.py`, jadi label dataset dan status di layar
  mustahil memakai ambang berbeda (diuji: objectnya identik, bukan salinan).
- **Data tidak layak** (usia di luar 6–59, LiLA di luar 5–50, NaN/inf,
  kategori tak dikenal) → `status_antropometri = "tidak_dapat_ditentukan"`
  beserta alasan; **tidak pernah** menebak dan tidak melempar exception.
  `kelompok` / `rekomendasi` jadi `None` hanya bila kategori tak dikenal.
- **Istilah per kelompok** (di `app/content.py`): ibu → **"risiko KEK
  (Kekurangan Energi Kronis)"**, balita → **"status gizi balita menurut
  LiLA"** (mengikuti ambang 11.5/12.5 cm). Ibu tidak pernah disebut
  "gizi buruk akut".

#### `indikator_terukur`

Daftar item `{kunci, label, nilai, satuan, badge, keterangan, sumber}`.
`risk_status.indikator_terukur` sendiri tidak punya field `sumber`; endpoint
menambahkannya (`sumber="terukur"`) lewat model respons.

| `kunci` | `badge` | `keterangan` |
|---------|---------|--------------|
| `lila` | sama dengan `status_akhir` | membandingkan nilai dengan ambang, mis. `22.0 cm, di bawah 23.5 cm`; kalau status tidak dapat ditentukan, keterangannya menjelaskan mengapa LiLA tidak dibandingkan |
| `berat_badan` | selalu `null` | selalu `null` — **tidak ada penilaian** naik/turun, karena repo ini tidak punya data riwayat berat |

Kedua kunci, label, dan seluruh keterangannya berasal dari `app/content.py`.

### Indikator visual deskriptif

**`app/indicators.py`** mengubah fitur **mentah** pipeline
(`extract_hair_features`, `extract_sclera_features`,
`extract_geometry_features`) menjadi `IndicatorResult { items, tersedia }`.
Barisnya berbentuk
`{kunci, kelompok, label, nilai, satuan, keterangan, sumber, badge, badge_label}`.

- **Deskriptif, bukan penilaian.** Tiap baris hanya melaporkan nilai + label
  netral, mis. `Rasio lebar pipi terhadap lebar rahang: 0.82`. Tidak ada ambang
  normal/abnormal, tidak ada skor risiko, dan `badge` selalu `null`
  (`badge_label` = "Eksperimental", `sumber` = "visual_eksperimental").
  Kata `malnutrisi`, `anemia`, `berisiko`, `terdeteksi`, `pucat`, `cekung`
  dilarang — test memindai teks `indikator.*`/`satuan.*` **dan** semua string
  hasil.
- **`status_cv` serta `score`/`normalized` tidak dipakai** dan tidak bocor ke
  output; yang diteruskan cuma fitur mentah. Test memberi marker pada
  kunci-kunci itu lalu memastikan tidak ada yang nyangkut, dan memeriksa
  sumber modul bebas nama-nama itu.
- **Pemetaan ke 4 kelompok** (`konjungtiva`, `rambut`, `pipi`, `kulit`) ada di
  tabel konstan `PEMETAAN`, memakai nama fitur yang ADA persis di
  `app/pipeline.py` — test mem-parse ulang ketiga fungsi extractor, jadi nama
  fitur karangan langsung ketahuan. Fitur yang bukan pengukuran (`available`,
  `reason`, `foreground_ratio`, `n_pixels`, `n_eyes`) tidak ditampilkan;
  bila pipeline menambah fitur baru, test **melaporkan** dan meminta
  keputusan: petakan, atau daftarkan sebagai metadata.
- **NaN / hilang tidak disembunyikan**: baris tetap ada dengan `nilai: null`
  dan keterangan `"tidak tersedia"` (dari `app/content.py`), tidak
  dihilangkan diam-diam.
- **Pipeline gagal** (`features` `None` / bukan dict / kosong) → `items: []`,
  `tersedia: false`, tanpa exception.
- **Mekanisme ambang tersedia tapi kosong**: `app/settings.py::ambang_indikator`
  (env `AMBANG_INDIKATOR`, JSON array) dibaca lewat
  `app.indicators.muat_ambang()`. Setiap entri **wajib** punya `sumber`
  (asal angka); entri tanpa `sumber` diabaikan, default kosong. Ambang itu
  sengaja **tidak** dipakai untuk menilai nilai — memakainya akan menghasilkan
  klasifikasi.

Label, satuan, keterangan, dan `badge_label` semuanya dari `app/content.py`
(kunci `indikator.*` dan `satuan.*`); test memastikan modul ini tidak
menulis literal teks apa pun. Modul belum dipasang ke endpoint mana pun —
pemasangannya urusan terpisah, supaya kontrak endpoint lama tidak berubah.

### Format error

Setiap endpoint membalas error dengan bentuk yang sama:

```json
{
  "error": {
    "code": "unsupported_media_type",
    "message": "Content-Type 'text/plain' tidak didukung. Yang diterima: image/jpeg, ..."
  }
}
```

`error.code` adalah slug stabil yang diturunkan dari status code (daftarnya di
`app/errors.py`), jadi klien boleh bercabang di atasnya tanpa harus mem-parse
pesan. `error.message` selalu kalimat yang bisa langsung ditampilkan ke
pengguna.

Field `detail` bawaan FastAPI tetap ikut dikirim **apa adanya** di samping
`error`, supaya klien lama tidak ada yang rusak: `reason`/`hint` penolakan
quality gate masih ada di `detail.reason` / `detail.hint`.

Untuk **error tak terduga** (500) balasannya juga seragam, tapi pesannya
selalu generik: stack trace, pesan exception, dan path file TIDAK pernah
dikirim ke klien — semuanya hanya ditulis ke log server (`app/errors.py`).

Endpoint baru (auth, profile, subjek, sync) cukup melempar `HTTPException`
seperti biasa — handler global di `app/errors.py` yang membungkusnya.

## Protokol foto (quality gate)

Foto ditolak — **bukan diproses diam-diam** — kalau:

- Bukan tepat satu wajah (multi-face ditolak, bukan dipilih diam-diam).
- Wajah terlalu kecil di frame (< 160 px tinggi).
- Kepala tidak menghadap lurus (yaw proxy > 0.30).
- Kepala miring (roll > 15°).
- Kepala menengadah/mendongak (pitch ratio di luar 0.24–0.62).
- Foto buram (variance of Laplacian < 40).
- Pencahayaan tidak sesuai (L* kulit di luar 38–94).

`GET`/debug: `POST /api/v1/debug-landmarks` mengembalikan gambar JPEG dengan
landmark, kotak region, dan mask — untuk validasi visual, tidak untuk produksi.

## Cara kerja indikator

Semua angka ada di `app/config.py`.

| Indikator | Yang diukur | Dasar klinis |
|-----------|-------------|--------------|
| `rambut` | L\* rambut − L\* kulit, dan rasio chroma rambut/kulit | Kwashiorkor mengurangi melanin rambut → rambut "reddish yellow to white" (Medscape) / "gray-blond" (dokumentasi foto CDC) |
| `kepucatan_mata` | rasio a\* sklera / a\* kulit | a\* mengukur kemerahan (proporsi hemoglobin). Sklera pucat kehilangan kemerahan |
| `geometri_wajah` | rasio lebar pipi/rahang, dan tinggi lower-face | Wajah menyempit, rahang lebih menonjol pada malnutrition kronis |

### Beberapa keputusan yang perlu diketahui

**Arah indikator rambut dibalik dari versi lama.** Versi lama memakai
`hue <= 20 or hue >= 160`, yang terbukti salah arah: ia *miss* pada warna
rambut justru yang dideskripsikan literatur ("reddish yellow"), dan
*false-positive* pada rambut coklat Asia yang normal. Sekarang yang dipakai
adalah L\* dan chroma relatif terhadap kulit subjek itu sendiri.

**Croma rambut saja tidak menambah severity.** Kalau rambut jauh lebih gelap
dari kulit (`l_delta` di atas `l_delta_max`), tidak mungkin hipopigmen —
sehingga chroma yang rendah tidak boleh menyumbang skor. Tanpa aturan ini
setiap anak berambut hitam akan mendapat skor > 0.

**Sklera memakai ROI per mata, berbasis ring iris.** Dua ROI dianalisis
terpisah supaya satu mata tertutup/kacamata tidak menghapus informasi mata
yang lain. Radius iris dihitung dari ring iris (4 titik), **bukan** ring mata
(16 titik) — kalau keliru, radius terbaca terlalu besar dan ROI menapu
hidung serta kulit sehingga a\* yang diukur bukan sklera.

**Region rambut mengikuti bentuk wajah, bukan kotak.** Kotak persegi di atas
wajah yang bulat selalu menyorotKAN sudut-sudut ke dinding/langit, dan sudut
itu ikut masuk rata-rata warna rambut. Area pencarian sekarang berbentuk trapesium
mengikuti oval kepala.

**Semua LAB dikonversi ke CIELAB asli.** OpenCV menyimpan LAB sebagai uint8
(L di 0–255, a\*/b\* +128). Nilai itu dipakai apa adanya di versi lama, yang
merusak dua hal: quality gate exposure menolak hampir semua foto normal
(L=181 terbaca, bukan L\*=71), dan rasio a\* sklera jadi ~1.0 tidak peduli
apa pun warnanya sehingga indikator anemia praktis mati.

**Status tidak boleh "normal" bila ada indikator yang FLAG.** Skor agregat
kecil tidak boleh menutupi satu temuan positif; status dinaikkan minimal ke
`sedang`.

## Batasan klinis (jujur)

1. **Tidak terkalibrasi.** Tidak ada dataset berlabel di repo ini.
2. **Konjungtiva sejati tidak terlihat** di foto biasa — kelopak mata harus
   ditarik. Indikator `kepucatan_mata` hanya mengukur area mata yang terekspos,
   jadi **tidak setara** pemeriksaan konjungtiva klinis.
3. **Sensitivitas kepucatan konjungtiva terhadap anemia ringan rendah**
   (<69%); untuk anemia sedang/berat lebih tinggi (≥70%).
4. **Rambut dipengaruhi banyak hal non-klinis**: cat, bleaching, pemutih,
   penata rambut, dan cahaya. Indikator harus dibaca bersama keterangan
   apakah ada perlakuan tersebut.
5. **Proorsi wajah dipengaruhi etiologi.** Malnutrisi bukan satu-satunya
   penyebab perubahan proorsi; kondisi gigi dan ortodonti juga
   mengubahnya.
6. **Kondisi kulit memengaruhi pembacaan.** Berjerawat, berkeringat, atau
   kemerahan membuat L\* dan a\* bergeser dan bisa salah memicu flag.
7. **Tidak ada validasi pada anak Indonesia.** Ambang berasal dari
   literatur umum, bukan dari populasi target.

Untuk anemia, cutoff hemoglobin menurut WHO (2024): 6–23 bulan `<105 g/L`,
24–59 bulan `<110 g/L`, 5–11 tahun `<115 g/L`. Penilaian anemia pada
anak tidak boleh menyimpang dari pemeriksaan laboratorium ini.

### Literatur yang tidak boleh dijadikan dasar

Sebuah artikel *Scientific Reports* (2025) tentang klasifikasi malnutrisi
dari citra wajah menggunakan ResNet-50 **berstatus RETRACTED**. Jangan
discitasi sebagai pembenaran.

## Catatan implementasi

- **EXIF orientation diterapkan.** `cv2.imdecode()` mengabaikan EXIF, jadi
  foto portrait dari HP akan diproses miring. Semua 8 nilai EXIF ditangani
  lewat `Pillow.ImageOps.exif_transpose`.
- **Pipeline dipindah ke threadpool.** OpenCV + MediaPipe itu CPU-bound;
  memanggilnya langsung di dalam `async def` memblokir event loop FastAPI dan
  ikut macetkan seluruh request lain.
- **FaceLandmarker dimuat sekali** dengan lock, bukan per-request.
- **Batas ukuran upload dan dimensi gambar** untuk mencegah OOM.
- Wajah uji di-render secara prosedural (`tests/conftest.render_frontal_face`),
  bukan foto pihak ketiga — menghindari masalah lisensi, dan foto 3/4 view
  yang ada di internet memang harus ditolak gate sehingga tidak bisa dipakai
  sebagai end-to-end positif.