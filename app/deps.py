"""
deps.py
-------
FastAPI dependencies yang dipakai lintas endpoint.

get_current_user() sekarang SELALU mengembalikan None: autentikasi belum ada.
Endpoint sudah memanggilnya lewat Depends(), jadi saat auth benar-benar
ditambahkan cukup mengganti isi fungsi ini (dan tipe di CurrentUser) -- tidak
perlu menyentuh endpoint mana pun, dan kontrak OpenAPI tidak berubah.

Pola pemakaian di endpoint:

    from app.deps import CurrentUser

    @router.post("/contoh")
    async def contoh(current_user: CurrentUser = None, ...):
        ...
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends


async def get_current_user() -> None:
    """Placeholder autentikasi.

    -> None selama belum ada auth.

    Nanti fungsi ini yang membaca token/API key, memvalidasinya, dan
    melempar HTTPException(401) bila gagal. Endpoint tidak berubah sama sekali.
    """
    return None


CurrentUser = Annotated[Any, Depends(get_current_user)]
"""Annotasi siap pakai di signature endpoint: `user: CurrentUser = None`.

Tipe `Any` supaya saat auth datang, mengubah isi dependency tidak memaksa
menulis ulang signature endpoint.
"""
