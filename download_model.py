"""
download_model.py
------------------
Download model FaceLandmarker resmi dari MediaPipe (sekali saja).
Jalankan dari direktori project:

    python download_model.py

Path di sini sengaja dibuat absolut relatif ke file ini, karena app/pipeline.py
membaca model dari app/config.py: MODEL_PATH yang juga relatif ke file.
Versi lama memakai Path("models") yang bergantung CWD, sehingga model bisa
terdownload ke satu direktori lalu dicari di direktori lain -> FileNotFoundError
padahal file-nya ada.
"""

import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
OUTPUT_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
        print(f"Model sudah ada di {OUTPUT_PATH} ({size_mb:.1f} MB), skip download.")
        return

    print(f"Downloading model dari {MODEL_URL} ...")
    tmp_path = OUTPUT_PATH.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        # hanya rename kalau ukuran wajar, biar file .part yang terpotong
        # tidak dipakai sebagai model yang rusak
        if tmp_path.stat().st_size < 1_000_000:
            raise RuntimeError(
                f"File yang didownload cuma {tmp_path.stat().st_size} byte -- "
                "kemungkinan terpotong oleh proxy. Coba lagi."
            )
        tmp_path.replace(OUTPUT_PATH)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print(f"Selesai. Model tersimpan di {OUTPUT_PATH}")


if __name__ == "__main__":
    main()