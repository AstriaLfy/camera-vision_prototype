# Laporan

## Description
Halaman Laporan adalah pusat dokumentasi dan pelaporan hasil screening yang sudah dilakukan. Berbeda dengan Beranda yang fokus ke "apa yang harus dilakukan sekarang," Laporan fokus ke "apa yang sudah terjadi dan bagaimana trennya." Halaman ini melayani dua kebutuhan berbeda — petugas yang butuh riwayat per subjek, dan supervisor yang butuh data agregat untuk dilaporkan ke dinas kesehatan atau NGO koordinator.

## Goal
- Petugas bisa akses riwayat screening semua subjek dengan cepat
- Supervisor bisa lihat tren kondisi posko dari waktu ke waktu
- Data bisa di-export dalam format yang siap dilaporkan ke dinas kesehatan atau NGO tanpa perlu rekap manual
- Semua data historis tetap accessible meskipun offline dari cache lokal

## Constraint
- Data yang ditampilkan mencakup seluruh riwayat, bukan hanya hari ini — ini yang membedakan dari Beranda
- Export hanya bisa dilakukan saat online
- Filter dan search harus cepat dan simpel — tidak boleh lebih dari 2 tap untuk menemukan subjek tertentu

## Data
- Riwayat screening per subjek lengkap dengan timestamp
- Tren distribusi risiko per hari/minggu
- Total screening per periode
- Data per kategori: anak balita / ibu hamil / ibu menyusui
- Catatan tindak lanjut per subjek
- Data petugas yang melakukan screening

## UI Komponen

### Header
- Tombol Export di pojok kanan atas — aktif saat online, greyed out saat offline dengan tooltip "Hubungkan internet untuk export"
- Filter chips horizontal scroll di bawah judul: Semua / Anak Balita / Ibu Hamil / Ibu Menyusui
- Search bar di bawah filter untuk cari nama subjek spesifik

### Tab
- Tab pertama: Subjek (riwayat per individu)
- Tab kedua: Statistik (tren agregat)

### Tab Subjek
- List semua subjek yang pernah di-screening, diurutkan berdasarkan screening terakhir
- Setiap card menampilkan:
  - Nama subjek + kategori (anak balita / ibu hamil / ibu menyusui)
  - Status risiko dengan badge warna merah / kuning / hijau
  - Usia subjek
  - Nomor tenda / lokasi di posko
  - Indikator yang terdeteksi — "Rambut kemerahan, konjungtiva pucat"
- Tap card -> masuk ke [FEAT] Detail Subjek
- Sorting dan filter yang tersedia:
  - Default terbaru di atas
  - Filter by status: Merah / Kuning / Hijau
  - Filter by kategori: Anak Balita / Ibu Hamil / Ibu Menyusui

### Tab Statistik
- Summary periode dengan selector dropdown: hari ini / minggu ini / bulan ini / custom range
- Menampilkan total screening, jumlah subjek unik, rata-rata screening per hari
- Tren risiko bar chart per hari menunjukkan proporsi merah, kuning, hijau dari hari ke hari
  - Kalau tren merah menurun -> badge positif "Kondisi posko membaik dalam 3 hari terakhir"
  - Kalau tren merah meningkat -> alert "Kasus risiko tinggi meningkat — segera laporkan ke koordinator"
- Distribusi per kategori dalam donut chart — proporsi anak balita, ibu hamil, ibu menyusui
- Indikator paling sering terdeteksi — list berurutan dari terbanyak:
  1. Konjungtiva pucat
  2. Berat badan kurang
  3. Rambut kemerahan
  4. Pipi cekung

### Empty State
- Belum ada data: teks "Belum ada riwayat screening" dengan tombol shortcut "Mulai Screening" yang trigger FAB flow

### Error State
- Offline dan cache tidak tersedia: "Data tidak dapat dimuat. Hubungkan ke internet untuk sinkronisasi pertama."
- Offline tapi cache tersedia: data tetap tampil normal dengan banner kecil di atas "Menampilkan data terakhir — offline mode"

### Export Flow
1. Tap tombol Export -> bottom sheet muncul
2. Pilihan format: PDF — siap cetak dan kirim / Excel — untuk analisis lanjut
3. Pilihan periode: Hari ini / Minggu ini / Bulan ini / Custom range
4. Pilihan isi laporan:
   - Data screening per subjek
   - Statistik agregat
   - Catatan tindak lanjut
   - Foto subjek (opsional — ukuran besar)
5. Tap "Export Sekarang" -> file langsung bisa di-share via WhatsApp, Email, atau disimpan ke storage lokal
