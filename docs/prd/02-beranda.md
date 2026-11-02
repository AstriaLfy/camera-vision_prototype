# Beranda

## Description
Page utama yang langsung dilihat oleh tenaga kesehatan untuk bisa memantau kondisi pasien dalam satu kali lihat

## Goal
- Kader harus langsung tahu siapa yang harus discreening hari ini tanpa ribet

## Constraint
- Informasi yang ditampilkan hanya hari ini, data historis ada di page lain
- Informasi yang ditampilkan hanya konteks hari ini — data historis ada di halaman Laporan
- Maksimal 3 aksi utama yang bisa dilakukan dari Beranda — lebih dari itu overwhelming untuk konteks darurat

## Data
- Total subjek terdaftar di posko: 47
- Sudah di-screening: 32 | Belum di-screening: 15
- Nama posko aktif: Posko Pengungsian Cianjur Sektor 3

## UI Komponen

### Header
- Greeting personal: "Selamat pagi, [Nama Petugas] 👋"
- Nama posko yang ditugaskan
- Indikator koneksi (online/offline) + status sync ("Semua data tersimpan")
- Notif

### Summary Card 3
- Total subjek
- Selesai
- Pending

### Alert Bar
- Banner flag merah -> "⚠️ 2 kasus perlu perhatian segera"

### Daftar List Subjek
Setiap card di list ini menampilkan:
- Nama subjek + kategori (anak balita / ibu hamil / ibu menyusui)
- Badge warna
- Indikator yang terdeteksi -> "Rambut kemerahan, konjungtiva pucat"
- Waktu screening terakhir -> "Di-screening 2 jam lalu"
