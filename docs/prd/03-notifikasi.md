# Notifikasi

## Description
Pusat notifikasi yang dapat diakses dari ikon sekaligus mekanisme push notifikasi yang muncul di level sistem. berfungsi sebagai jembatan untuk memastikan kader tidak melewatkan kasus yang penting dalam satu notifikasi aja

## Goal
- Kader tidak melewatkan kasus flag merah yang butuh tindak lanjut segera
- Kader diingatkan otomatis untuk jadwal screening ulang subjek berisiko, tanpa perlu mengecek manual satu-satu

## Constraint
- Notifikasi bisa diakses offline dari data lokal
- Diurutkan berdasarkan prioritas visual: notifikasi flag merah -> pengingat follow up -> status sync

## Data
- Jenis notif: flag merah / pengingat follow up / status sync / insight pola
- Timestamp notifikasi dibuat
- Status: belum dibaca / sudah dibaca

## UI Komponen

> Catatan: bagian UI Komponen di sumber PM identik dengan Beranda (kemungkinan salah salin). Disalin apa adanya, lihat KONFLIK.md.

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
