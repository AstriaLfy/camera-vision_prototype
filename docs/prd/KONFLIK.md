# Konflik PRD vs Keputusan Repo

Daftar ini hanya mencatat. Tidak ada yang diputuskan di sini.

**Keputusan repo:** status risiko murni dari LiLA (balita 6-59 bulan, ibu hamil/menyusui); foto hanya data pendukung eksperimental tanpa badge risiko dan tanpa kata klinis dari foto; teks rekomendasi hanya dari `content.py` dan diverifikasi ahli gizi; "Tanya AI" disembunyikan; tidak ada BMI untuk balita dan ibu hamil; endpoint screening stateless.

## A. Bertabrakan dengan keputusan repo

| # | File PRD | Kutipan singkat | Keputusan repo yang bertabrakan |
|---|---|---|---|
| 1 | 06-screening | "Hasil analisis CV + skor risiko + indikator terdeteksi" | Foto tidak menghasilkan skor risiko |
| 2 | 06-screening | "Inferensi TFLite jalan lokal di device" | Tidak ada model terlatih; hanya deteksi landmark pra-latih + fitur deskriptif |
| 3 | 06-screening | "Indikator visual yang terdeteksi (rambut, konjungtiva, pipi, berat badan)" | Data pendukung eksperimental, badge null, tanpa kata "terdeteksi" |
| 4 | 06-screening | "Rekomendasi tindakan berbasis protokol WHO" | Teks hanya dari `content.py`, diverifikasi ahli gizi |
| 5 | 06-screening (desain FE) | "Indikator gizi buruk akut terdeteksi" untuk ibu (LiLA 22.0) | Ibu: istilah "risiko KEK" |
| 6 | 06-screening (desain FE) | Badge Risiko Tinggi/Sedang untuk Konjungtiva, Rambut, Pipi | Hanya item terukur (LiLA) yang berbadge |
| 7 | 06-screening (desain FE) | "Pucat 0 kemungkinan anemia gizi", "Tampak cekung ringan" | Dilarang kata klinis dari foto |
| 8 | 06-screening (desain FE) | Berat Badan: "kenaikan tidak cukup" + badge | Tanpa riwayat BB, berat badan hanya nilai, badge null |
| 9 | 06-screening (desain FE) | "PMT terapi F75/F100 sesuai protokol WHO" | Protokol medis tidak dikarang AI; harus dari pedoman resmi/ahli gizi |
| 10 | 06-screening (desain FE) | Tombol "Tanya AI untuk Rekomendasi" | Disembunyikan dulu |
| 11 | 02-beranda, 03-notifikasi, 07-laporan | "Indikator yang terdeteksi -> Rambut kemerahan, konjungtiva pucat" | Tidak ada kata klinis dari foto |
| 12 | 07-laporan | "Indikator paling sering terdeteksi: Konjungtiva pucat, Berat badan kurang, Rambut kemerahan, Pipi cekung" | Idem; statistik indikator foto tidak boleh berupa kesimpulan klinis |
| 13 | 03-notifikasi | Jenis notif "insight pola" | Belum ada dasar validasi pola dari foto |
| 14 | 05-detail-edukasi | "Berdasarkan hasil screening budi" (nama subjek di badge) | Perlu keputusan privasi data subjek |
| 15 | 06-screening | "Auto sync ke Google Cloud", hasil dapat diakses supervisor real-time | Endpoint screening stateless; penyimpanan/sync belum dibuat |

## B. Ketidakjelasan / inkonsistensi di PRD

| # | File PRD | Isu |
|---|---|---|
| 16 | 06-screening | Ambang LiLA balita: "11.5-12.4 (kuning) / >12.5 (hijau)" tidak mencakup 12.5 tepat (dan 12.45); ibu "<23.5 / >23.5" tidak mencakup 23.5 tepat. Repo: <11.5 merah, 11.5 sampai <12.5 kuning, ≥12.5 hijau; ibu <23.5 merah, ≥23.5 hijau |
| 17 | 06-screening | Jenis kelamin dan PB/TB tidak ada di input, padahal dibutuhkan untuk BB/TB WHO |
| 18 | 06-screening | Status "Risiko Sedang" (kuning) tidak ada untuk ibu (hanya merah/hijau menurut ambang) |
| 19 | 01-autentikasi | "Register untuk pemilik" (kemungkinan maksudnya kader/pengguna) |
| 20 | 01-autentikasi vs 08-profil | Nama contoh berbeda: Siti Rahmawati vs Siti Rahayu |
| 21 | 03-notifikasi | UI Komponen identik dengan Beranda (kemungkinan salah salin); UI notifikasi sebenarnya belum didefinisikan |
| 22 | 02-beranda | Constraint "hanya hari ini" tertulis dua kali; "Maksimal 3 aksi utama" belum dirinci aksinya |
| 23 | 07-laporan | `[FEAT] Detail Subjek` dirujuk tapi tidak ada PRD-nya |
| 24 | 04-edukasi | "Penanganan Gizi Buruk Akut" dan "Protokol MPASI Darurat": isi medis harus dari sumber resmi dan diverifikasi |
