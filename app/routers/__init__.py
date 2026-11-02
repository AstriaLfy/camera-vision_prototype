"""
app/routers/
------------
Satu modul per domain, satu APIRouter per modul. Setiap router menentukan
prefix-nya sendiri (`/api/v1`) sehingga main.py cukup mendaftarkan router dan
tidak berisi logika endpoint.

Menambah endpoint baru (auth, profile, subjek, sync, dst):
    1. buat `app/routers/<domain>.py` berisi `router = APIRouter(prefix="/api/v1")`
    2. daftarkan di main.py: `app.include_router(<domain>.router)`

Domain yang sudah ada: screening (upload foto), health (status layanan).
"""
