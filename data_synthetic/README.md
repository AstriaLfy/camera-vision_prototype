# DATA SINTETIS — ARSIP SKEMA LAMA

> **ARSIP. `labels.csv` di folder ini memakai skema lama dan tidak kompatibel
> dengan `scripts/extract_dataset.py` yang berjalan sekarang.**
>
> **DATA SINTETIS, hanya untuk menguji pipeline, tidak boleh dipakai sebagai
> hasil penelitian.**

## Kenapa diarsipkan, bukan dimigrasikan

Label di repo ini sekarang berbasis **LiLA per kategori** (`balita`,
`ibu_hamil`, `ibu_menyusui`), dihitung dari `lila_cm`. `extract_dataset.py`
membutuhkan kolom `kategori` dan `lila_cm`.

`labels.csv` di sini hanya berisi:

```
subject_id,image_path,tinggi_cm,berat_kg,usia,jenis_kelamin
```

Tidak ada `kategori`, tidak ada `lila_cm` — **label LiLA tidak bisa dihitung
ulang dari data ini**, dan menambahkan kolomnya butuh pengukuran yang memang
tidak ada. Karena itu file ini sengaja **tidak dimigrasikan**; ia tetap
tersimpan sebagai artefak fase awal (label berbasis BMI/tinggi) supaya riwayat
pipeline terlacak.

Karena itu `scripts/extract_dataset.py` dan `scripts/eda.py` **tidak bisa
dijalankan** terhadap isi folder ini. Perintahnya sudah dihapus dari README
ini, karena pasti gagal: `labels.csv` tidak punya kolom `kategori` dan
`lila_cm` yang sekarang wajib.

Kalau suatu saat butuh data sintetis lagi, skrip pembuatnya harus menulis
ulang `labels.csv` dengan skema LiLA (salin `data/labels_template.csv`), lalu
skrip ekstraksi dan EDA dipakai seperti pada data asli.

## Apa isi folder ini

Semuanya di folder ini **dibuat dengan program**, bukan diambil dari manusia
sama sekali. Isinya:

| Berkas | Apa itu |
|---|---|
| `labels.csv` | **Skema lama.** Metadata 170 subjek sintetis: `subject_id`, `image_path`, `tinggi_cm`, `berat_kg`, `usia`, `jenis_kelamin` |
| `images/` | 170 foto wajah 640×640 hasil render prosedural + variasi photometric/geometri |
| `features.csv` | Hasil `extract_dataset.py` **versi lama** atas folder di atas (145 baris diterima, 25 ditolak) — turunan dari skema BMI, bukan dari skema LiLA |
| `../reports_synthetic/` | Laporan quality gate dan EDA atas `features.csv` versi lama, juga arsip |
| `README.md` | Catatan arsip ini |

Karena `features.csv` punya kolom `bmi`/`bmi_z`/`bmi_category` yang sudah
dihapus dari pipeline, `scripts/eda.py` **tidak bisa lagi** membacanya.

## Kenapa ini harus diperlakukan berbeda

Wajahnya **digambar oleh `render_frontal_face()` di `tests/conftest.py`** —
berupa elips dan garis, bukan foto orang. Pipeline, landmark, dan segmentasi
berjalan di atasnya tanpa error, tapi angka fitur yang keluar tidak
merepresentasikan apa pun soal tubuh manusia.

Tinggi, berat, dan usia juga **diformulasikan sendiri oleh skrip pembuatnya**
(ada korelasi buatan antara usia dan BMI), bukan hasil pengukuran.

Akibatnya, kesimpulan apa pun yang kita ambil dari `features.csv` ini —
termasuk hasil EDA di `reports_synthetic/` — **hanya berlaku untuk data ini**.
Angka `rho`, `p-value`, dan daftar fitur "signifikan" di sana tidak boleh
dibutip sebagai temuan, tidak boleh dipakai untuk membenarkan keputusan
klinis, dan tidak boleh dipakai menilai apakah pipeline "berhasil".

Sebagai contoh nyata: di EDA sintetis ini, `hair_chroma` terlihat sangat
"signifikan" membedakan label 0 vs 1 (p ≈ 6e-07). Kenyataannya efek itu
dibuat-buat oleh skrip pembuat dataset, dan begitu usia dikontrol korelasinya
jatuh ke ~0.007. Kalau dataset sintetis ini dipakai sebagai bukti, kesimpulan
yang dihasilkan justru salah.

## Data asli

Data asli tidak ada di repo ini. Letakkan di:

```
data/
├── images/                  # foto subjek
└── labels.csv               # isi dari data/labels_template.csv (skema LiLA)
```

`data/` dan `reports/` sudah masuk `.gitignore`, jadi data pribadi tidak akan
ter-commit. Folder `data_synthetic/` dan `reports_synthetic/` juga diabaikan
(kecuali README ini) supaya 12 MB gambar sintetis tidak ikut membesarakan
repo.
