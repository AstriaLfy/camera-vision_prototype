# Detail Edukasi

## Description
Halaman detail yang muncul saat kader tap salah satu card topik di halaman Edukasi, atau saat diarahkan lewat rekomendasi otomatis dari hasil Screening. Menampilkan konten edukasi lengkap dengan format yang mudah di-scan cepat (numbered steps, callout box), bukan paragraf panjang, karena kader mengaksesnya di tengah kesibukan lapangan.

## Goal
- Kader bisa cepat memahami dan menerapkan panduan tanpa perlu membaca teks panjang
- Kader yang masuk dari hasil screening subjek tertentu langsung paham kenapa artikel ini direkomendasikan untuknya
- Konten tetap kredibel dengan rujukan sumber yang jelas

## Constraint
- Format konten wajib terstruktur (numbered steps/poin, bukan paragraf panjang) untuk kecepatan baca di lapangan
- Artikel bertema tindakan medis wajib menampilkan banner disclaimer di bagian atas konten, sebelum isi utama
- Sumber rujukan wajib dicantumkan di bagian akhir setiap artikel

## Data
- Judul artikel
- Ilustrasi/gambar pendukung
- Isi konten (poin-poin terstruktur)
- Quick tip/callout tambahan (opsional per artikel)
- Sumber rujukan
- Konteks asal akses: dari tab Edukasi langsung, atau dari deep link hasil screening subjek tertentu

## UI Komponen
- Tombol "Kembali" di header
- Ilustrasi/gambar topik di bagian atas
- Judul artikel
- Badge kontekstual (kondisional, muncul kalau diakses via deep link dari Screening), dengan 3 tingkatan tampilan tergantung jumlah kasus serupa yang terdeteksi:
  - Kasus tunggal: "Berdasarkan hasil screening budi"
  - Ada beberapa kasus serupa: "Berdasarkan hasil screening dari 10 subjek"
  - Kasus signifikan: "22 subjek menunjukkan indikasi serupa"
- Konten utama dalam format numbered steps, tiap poin punya judul singkat + deskripsi pendek
- Callout box "Quick Tip" untuk catatan tambahan yang relevan (misal soal sanitasi)
- Footer sumber rujukan, contoh: "Sumber: WHO/UNICEF/IFE Core Group — Infant and Young Child Feeding in Emergencies" atau "Sumber: Kemenkes RI — Pedoman Gizi Darurat Bencana"
