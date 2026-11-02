"""alembic/env.py
----------------
Konfigurasi runtime Alembic.

URL database diambil dari app.settings (env ``DATABASE_URL``) supaya aplikasi
dan migrasi tidak pernah memakai alamat database yang berbeda.

``target_metadata`` sengaja ``None``: belum ada model tabel di repo ini. Saat
fitur pertama yang butuh tabel ditambahkan, impor metadata-nya di sini lalu
buat revisi dengan ``alembic revision --autogenerate``.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context

# prepend_sys_path=. (alembic.ini) membuat `app` bisa diimpor dari root proyek.
from app.settings import get_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# URL tunggal: sama dengan yang dipakai aplikasi.
config.set_main_option("sqlalchemy.url", get_settings().database_url)

# Belum ada metadata tabel. Isi saat model pertama dibuat.
target_metadata = None


def run_migrations_offline() -> None:
    """Jalankan migrasi tanpa koneksi (menghasilkan SQL)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Jalankan migrasi melalui koneksi database."""
    from sqlalchemy import engine_from_config, pool

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
