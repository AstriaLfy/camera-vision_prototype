# Autentikasi

## Description
Halaman pertama yang dilihat pengguna saat membuka app untuk pertama kali atau setelah logout. Terdiri dari dua flow utama: Register untuk pemilik dan Login untuk yang sudah punya akun.

## Goal
- Kader bisa daftar dan langsung siap melakukan screening
- Sistem tetap bisa memverifikasi identitas kader secara longgar tanpa menghalangi kecepatan respon

## Constraint
- Session persist, pengguna tidak perlu login ulang setiap buka app
- Register hanya bisa dilakukan sekali per email
- Register wajib online minimal sekali, tapi setelahnya bisa offline
- Setelah register selesai, langsung diarahkan ke Beranda
- Sumber data subjek per wilayah bisa lewat dari dua flow:
  - sinkron otomatis dari sistem terpusat
  - import manual file kalau belum ada integrasi sistem

## Data
- Nama: Siti Rahmawati
- Email: siti.kader@gmail.com
- Password: ••••••••
- Posko Tugas: Posko Pengungsian Cianjur Sektor 3
- Sumber data: Sinkron otomatis (BPBD Cianjur) / File impor manual

## UI Flow

### Login
1. Input email
2. Input password
3. Tap "Masuk" -> masuk ke Beranda
4. Auto login jika session masih aktif

### Register
1. Input Nama Lengkap
2. Input email
3. Input password
4. Input confirm password
5. Pilih Wilayah
6. Tap -> Daftar
7. Input OTP
8. Direct ke Beranda
9. Posko tugas: Posko Pengungsian Cianjur Sektor 3

### Lupa Password
1. Input email yang terdaftar
2. Sistem kirim link reset ke email
3. Input password baru
4. Simpan -> redirect ke Login
