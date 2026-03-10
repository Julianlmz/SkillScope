# ============================================================
# SkillScope — Makefile
# Uso: make <comando>
# ============================================================

.PHONY: help install run-full run-extract run-transform run-load verify clean

# Variables
PYTHON     = python
PIPELINE   = src/pipeline/main.py
RAW_DIR    = data/raw
DB_PATH    = data/jobs.duckdb

# ── Ayuda ────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  SkillScope — Comandos disponibles:"
	@echo ""
	@echo "  make install        Instala dependencias con uv"
	@echo "  make run-full       Ejecuta pipeline completo (ETL)"
	@echo "  make run-extract    Solo fase Extract"
	@echo "  make run-transform  Solo fase Transform"
	@echo "  make run-load       Solo fase Load"
	@echo "  make verify         Verifica registros en DuckDB"
	@echo "  make clean          Elimina la base de datos generada"
	@echo ""

# ── Entorno ──────────────────────────────────────────────────
install:
	uv pip install -r requirements.txt

# ── Pipeline ─────────────────────────────────────────────────
run-full:
	$(PYTHON) $(PIPELINE) --mode=full --raw-dir=$(RAW_DIR) --db-path=$(DB_PATH)

run-extract:
	$(PYTHON) $(PIPELINE) --mode=extract --raw-dir=$(RAW_DIR)

run-transform:
	$(PYTHON) $(PIPELINE) --mode=transform --raw-dir=$(RAW_DIR)

run-load:
	$(PYTHON) $(PIPELINE) --mode=load --raw-dir=$(RAW_DIR) --db-path=$(DB_PATH)

# ── Verificación ─────────────────────────────────────────────
verify:
	@echo "── Registros en DuckDB ──"
	$(PYTHON) -c "\
import duckdb; \
con = duckdb.connect('$(DB_PATH)'); \
print('jobs     :', con.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]); \
print('job_skills:', con.execute('SELECT COUNT(*) FROM job_skills').fetchone()[0]); \
con.close()"

# ── Limpieza ─────────────────────────────────────────────────
clean:
	rm -f $(DB_PATH)
	@echo "Base de datos eliminada."