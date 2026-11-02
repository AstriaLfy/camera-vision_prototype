# Indeks PRD

Sumber kebenaran fitur. Baca indeks ini dulu, lalu hanya file fitur yang relevan. Jika PRD bertabrakan dengan `KONFLIK.md`, hentikan dan tanyakan, jangan memilih sendiri.

| No | Fitur | File | Tujuan (kutipan PRD) | Status implementasi |
|---|---|---|---|---|
| 01 | Autentikasi | `01-autentikasi.md` | Kader bisa daftar dan langsung siap melakukan screening | belum |
| 02 | Beranda | `02-beranda.md` | Kader harus langsung tahu siapa yang harus discreening hari ini tanpa ribet | belum |
| 03 | Notifikasi | `03-notifikasi.md` | Kader tidak melewatkan kasus flag merah yang butuh tindak lanjut segera | belum |
| 04 | Edukasi | `04-edukasi.md` | Kader bisa mengakses referensi gizi & kesehatan darurat dengan cepat, tanpa perlu koneksi internet | belum |
| 05 | Detail Edukasi | `05-detail-edukasi.md` | Kader bisa cepat memahami dan menerapkan panduan tanpa perlu membaca teks panjang | belum |
| 06 | Screening | `06-screening.md` | Petugas bisa melakukan screening malnutrisi secara lengkap dan terverifikasi dalam waktu kurang dari 5 menit per subjek | sebagian (backend: `app/risk_status.py`, `app/indicators.py`, `app/content.py`, `app/thresholds.py`, `app/routers/screening.py`) |
| 07 | Laporan | `07-laporan.md` | Petugas bisa akses riwayat screening semua subjek dengan cepat | belum |
| 08 | Profil | `08-profil.md` | Petugas bisa lihat dan edit informasi akun dengan cepat | belum |

Catatan: Detail Subjek (`[FEAT] Detail Subjek`) dirujuk dari Laporan tetapi belum ada PRD-nya.
