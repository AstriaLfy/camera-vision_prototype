# Profil

## Description
Halaman Profil adalah pusat pengaturan akun dan konfigurasi teknis aplikasi. Berbeda dari halaman lain yang task-oriented, Profil bersifat administratif — diakses sesekali, bukan setiap hari.

## Goal
- Petugas bisa lihat dan edit informasi akun dengan cepat
- Petugas bisa logout dengan aman tanpa risiko kehilangan data yang belum sync

## Constraint
- Perubahan informasi posko harus konfirmasi dulu sebelum tersimpan untuk mencegah perubahan tidak sengaja yang mempengaruhi data laporan
- Logout hanya bisa dilakukan kalau semua data sudah sync ke cloud — kalau belum, tampilkan warning

## Data
- Nama lengkap petugas
- Foto profil
- Nama posko yang ditugaskan
- Status sync data terakhir
- No HP: +62 812 xxxxxxx

## UI Komponen

### Header Profil
- [Foto Profil]
- Nama Lengkap Petugas
- Terakhir sync: X menit lalu
- Posko: Posko Pengungsian Cianjur Sektor 3

### Section 1 — Informasi Akun
- Nama Lengkap: Siti Rahayu
- Email: siti.kader@gmail.com
- Nomor HP: +62 812 xxxxxxx
- Ganti Password
- Tap edit ->

### Section 2 — Informasi Posko
- Posko Aktif: Posko Cianjur ›
- Lokasi: Cianjur, Jabar ›
- Tap edit -> muncul konfirmasi: "Mengubah posko akan mempengaruhi data laporan. Lanjutkan?"

### Logout
- [Keluar dari Akun] -> cek status sync -> warning kalau ada data belum sync
