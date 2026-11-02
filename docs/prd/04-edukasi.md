# Edukasi

## Description
Halaman Edukasi adalah pusat referensi konten gizi dan kesehatan darurat, berbentuk library artikel yang terkurasi dari sumber resmi (WHO/UNICEF, Kemenkes). Berfungsi sebagai referensi cepat buat kader di lapangan maupun panduan mandiri buat ibu di fase recovery. Halaman ini terhubung dengan flow Screening untuk hasil screening subjek bisa mengarahkan kader ke artikel yang relevan, sehingga Edukasi bukan modul berdiri sendiri melainkan bagian dari alur tindak lanjut.

## Goal
- Kader bisa mengakses referensi gizi & kesehatan darurat dengan cepat, tanpa perlu koneksi internet
- Konten yang ditampilkan kredibel dan bisa dipertanggungjawabkan sumbernya
- Edukasi terasa terintegrasi dengan hasil screening, bukan fitur terpisah; kader diarahkan ke artikel relevan berdasarkan kondisi subjek yang baru discreening

## Constraint
- Seluruh konten harus tersedia offline (data statis tersimpan lokal di device, bukan fetch dari server tiap kali dibuka)
- Setiap artikel wajib mencantumkan sumber rujukan resmi (WHO/UNICEF/Kemenkes) di bagian akhir konten
- Artikel yang berkaitan dengan tindakan medis (misal gizi buruk akut) wajib menyertakan disclaimer bahwa ini bersifat edukatif dan kasus risiko tinggi wajib dirujuk ke tenaga medis

## Data
- Judul topik + kategori (Balita/Ibu Hamil/Ibu Menyusui/Umum)
- Ikon/ilustrasi per topik
- Ringkasan singkat per topik (subjudul di card)
- Mapping rekomendasi artikel berdasarkan kategori subjek & indikator hasil screening

## UI Komponen
- List card topik, tiap card menampilkan: ikon, judul topik, subjudul kategori
- Topik inti untuk MVP:
  1. Protokol MPASI Darurat (Edukasi Gizi Balita)
  2. Mengenali Tanda Bahaya Gizi Buruk (Edukasi Gizi Balita)
  3. Nutrisi Ibu Hamil Pasca Bencana (Edukasi Ibu Hamil)
  4. Menjaga Produksi ASI di Kondisi Darurat (Edukasi Ibu Menyusui)
  5. Penanganan Gizi Buruk Akut (Tindakan Darurat Medis — dengan disclaimer)
  6. Sanitasi & Air Bersih (Panduan Kebersihan Posko)
- Tap card → masuk ke [FEAT] Detail Artikel (lihat `05-detail-edukasi.md`)
