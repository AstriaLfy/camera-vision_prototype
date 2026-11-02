# Screening

## Description
Flow multi-step yang dipicu dari tombol 🔵 FAB di tengah navbar. Flow ini adalah titik masuk semua proses screening baru dari identifikasi subjek, foto wajah untuk analisis CV, hingga rekomendasi tindakan berbasis protokol WHO.

## Goal
- Petugas kesehatan bisa melakukan screening malnutrisi secara lengkap dan terverifikasi dalam waktu kurang dari 5 menit per subjek tanpa perlu pencatatan manual di kertas
- Hasil screening langsung tersimpan ke sistem dan bisa diakses supervisor secara real-time

## Constraint
- Flow dianggap final kalau sudah menyelesaikan semua stepnya
- Berat badan diinput manual dan LiLA
- Foto wajah harus diambil langsung via kamera, tidak bisa dari galeri
- Semua data tersimpan lokal dulu, auto sync ke Google Cloud saat koneksi tersedia

## Data
- Nama subjek, kategori, usia, nomor tenda/lokasi di posko
- Foto wajah subjek
- Hasil analisis CV + skor risiko + indikator terdeteksi
- Rekomendasi tindakan + catatan petugas

## UI Komponen

### Step 1 — Identitas

**State A: Subjek Baru**
- Input nama lengkap
- Pilih kategori: Anak Balita / Ibu Hamil / Ibu Menyusui
- Input usia (Balita dalam bulan)
- Input nomor tenda di posko
- Tap "Simpan & Lanjut" -> data otomatis tersimpan ke database

**State B: Subjek Lama**
- Scan QR code subjek atau ketik nama di search bar
- Data subjek auto-fill dari database
- Tap "Lanjut"

### Step 2 — Data Antropometri
- Input berat badan (kg)
- Input LiLA (cm)
  - Anak balita = <11.5 cm (merah) / 11.5-12.4 (kuning) / >12.5 (hijau)
  - Ibu hamil/menyusui = <23.5 (merah) / > 23.5 (hijau)
- Tap "Konfirmasi & lanjut"

### Step 3 — Foto Wajah
- Kamera terbuka otomatis dengan face overlay guide (oval frame)
- Real-time indicator muncul: pencahayaan cukup/kurang, jarak terlalu jauh/dekat, wajah terdeteksi/tidak
- Tap tombol capture
- App otomatis cek kualitas foto -> blur atau wajah tidak terdeteksi -> diminta foto ulang
- Inferensi TFLite jalan lokal di device setelah foto dikonfirmasi

### Step 4 — Hasil Analisis
Tampilkan hasil screening lengkap:
- Status risiko: 🟢 Normal / 🟡 Risiko Sedang / 🔴 Risiko Tinggi
- Indikator visual yang terdeteksi (rambut, konjungtiva, pipi, berat badan)
- Data berat badan & LiLA vs threshold WHO sesuai usia/kategori
- Rekomendasi tindakan berbasis protokol WHO
- Tap "Tandai Prioritas" -> subjek masuk antrian tindak lanjut mendesak
- Tap "Lanjut ke Catatan" -> ke step terakhir

### Step 5 — Catatan Tindak Lanjut
Pilih tindakan yang dilakukan:
- Edukasi gizi
- Pemberian PMT (Pemberian Makanan Tambahan)
- Rujuk ke tenaga medis
- Monitoring lanjut

Lainnya:
- Text field untuk catatan tambahan
- Toggle "Ingatkan follow-up" -> subjek muncul di reminder besok di Beranda
- Tap "Selesai" -> data tersimpan lokal -> auto sync ke cloud saat online -> kembali ke Beranda

## Desain FE (referensi gambar)
Layar "Hasil Analisis" (langkah 4/5) pada desain menampilkan: header merah "Risiko Tinggi" dengan teks "Indikator gizi buruk akut terdeteksi — tindakan segera diperlukan", kartu Berat Badan (46 kg) dan LiLA (22.0 cm), daftar "Indikator Terdeteksi" (Konjungtiva, LiLA, Rambut, Berat Badan, Pipi, masing-masing dengan badge Risiko Tinggi/Sedang), kotak "Rekomendasi WHO / Kemenkes" (rujuk segera, PMT terapi F75/F100, lapor koordinator dalam 2 jam), tombol "Tandai Prioritas — Tindak Lanjut Mendesak", dan tombol "Tanya AI untuk Rekomendasi".
