"""
thresholds.py
-------------
Satu-satunya sumber angka ambang LiLA dan rentang usia balita.

Dipakai BERSAMA oleh dua tempat yang dulu menulis angkanya sendiri-sendiri:

  * app/risk_status.py            -- status risiko yang tampil ke pengguna
  * scripts/extract_dataset.py    -- label dataset

Duplikasi angka itu berbahaya: satu sisi boleh bergerak tanpa ketahuan sisi
lain, padahal label dataset dan status di layar harus memakai ambang yang
sama persis. Test `tests/test_risk_status.py` memastikan kedua modul membaca
objek konstanta dari file ini (bukan salinan), plus tidak ada lagi definisi
ambang kedua di dalam skrip.

Kenapa bukan app/config.py? Karena app/config.py memuat ambang indikator wajah
dan tidak boleh diubah dalam pekerjaan ini. LiLA bukan bagian dari pipeline CV,
jadi angkanya dipisah ke modul tersendiri yang tidak menyentuh pipeline.

Urutan ambang balita (usia 6-59 bulan):
    < {LILA_BALITA_MERAH_CM}            -> merah
    {LILA_BALITA_MERAH_CM} s/d < {LILA_BALITA_AMAN_CM} -> kuning
    >= {LILA_BALITA_AMAN_CM}            -> hijau

Ibu hamil / menyusui:
    < {LILA_IBU_AMAN_CM}                -> merah
    >= {LILA_IBU_AMAN_CM}               -> hijau
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "LILA_MIN_CM",
    "LILA_MAX_CM",
    "LILA_BALITA_MERAH_CM",
    "LILA_BALITA_AMAN_CM",
    "LILA_IBU_AMAN_CM",
    "BALITA_MIN_MONTH",
    "BALITA_MAX_MONTH",
]

#: Rentang LiLA yang masih masuk akal sebagai pengukuran. Di luar ini
#: berarti salah ketik / salah satuan, jadi datanya ditolak, bukan dipakai.
LILA_MIN_CM: Final[float] = 5.0
LILA_MAX_CM: Final[float] = 50.0

#: Balita: < 11.5 merah, 11.5 s/d < 12.5 kuning, >= 12.5 hijau.
LILA_BALITA_MERAH_CM: Final[float] = 11.5
LILA_BALITA_AMAN_CM: Final[float] = 12.5

#: Ibu hamil / menyusui: satu ambang saja, tanpa kategori kuning.
LILA_IBU_AMAN_CM: Final[float] = 23.5

#: Balita harus berusia 6-59 bulan; di luar itu LiLA tidak dipakai.
BALITA_MIN_MONTH: Final[int] = 6
BALITA_MAX_MONTH: Final[int] = 59
