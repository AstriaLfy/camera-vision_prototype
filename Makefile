# Perintah umum deploy. Semua lewat docker compose.
#
#   make up          # build + start api, db, migrate
#   make up-tunnel   # sama + ngrok (butuh NGROK_AUTHTOKEN di .env)
#   make down        # hentikan dan hapus container
#   make logs        # ikuti log semua service
#   make migrate     # jalankan alembic upgrade head
#   make test        # jalankan test suite terhadap db-test
#   make psql        # shell database service `db`
#   make backup      # pg_dump database ke file .sql
#   make restore FILE=backup.sql

COMPOSE := docker compose
SERVICE_API := api
TEST_DATABASE_URL := postgresql+psycopg://test:test@db-test:5432/test

.DEFAULT_GOAL := help

.PHONY: help up up-tunnel down logs migrate build test psql psql-test backup restore

help: ## Tampilkan daftar perintah
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build: ## Bangun image aplikasi
	$(COMPOSE) build

up: ## Build + start api, db, migrate
	$(COMPOSE) up -d --build

up-tunnel: ## Start dengan tunnel ngrok (profil "tunnel")
	$(COMPOSE) --profile tunnel up -d --build

down: ## Hentikan dan hapus container (volume tetap)
	$(COMPOSE) --profile tunnel --profile test down

logs: ## Ikuti log semua service
	$(COMPOSE) logs -f --tail=200

migrate: ## Jalankan migrasi database (alembic upgrade head)
	$(COMPOSE) up -d db
	$(COMPOSE) run --rm migrate

test: ## Jalankan test suite terhadap service db-test
	$(COMPOSE) --profile test up -d --wait db-test
	# Repo di-mount read-only ke /app supaya test memakai kode + scripts/ +
	# data/who/ yang ada saat ini (tests/ dan scripts/ sengaja tidak masuk
	# image produksi). -p no:cacheprovider: /app read-only.
	$(COMPOSE) run --rm --no-deps -T \
		-e DATABASE_URL=$(TEST_DATABASE_URL) \
		-v "$$(pwd):/app:ro" \
		$(SERVICE_API) python -m pytest tests/ -q -p no:cacheprovider

psql: ## Shell psql ke database service `db`
	$(COMPOSE) exec db sh -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'

psql-test: ## Shell psql ke database service `db-test`
	$(COMPOSE) --profile test exec db-test sh -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'

backup: ## Backup database `db` ke backup_<tanggal>.sql (pg_dump)
	$(COMPOSE) exec -T db sh -c 'pg_dump -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"' \
		> backup_$$(date +%Y%m%d_%H%M%S).sql
	@echo "Backup selesai."

restore: ## Restore dari FILE=backup.sql (psql)
	@test -n "$(FILE)" || (echo "Pakai: make restore FILE=backup.sql" && exit 1)
	$(COMPOSE) exec -T db sh -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"' < $(FILE)
	@echo "Restore selesai."
