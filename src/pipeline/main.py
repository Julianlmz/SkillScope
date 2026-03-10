"""
src/pipeline/main.py
Orquesta el pipeline ETL completo: Extract → Transform → Load.
Exporta un CSV resumen a data/processed/ para verificación de artefactos.
"""
import argparse
import logging
import sys
from pathlib import Path

# Permite importar src.* desde la raíz del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.extract import extract_jobs, extract_skills
from src.transformation.clean_jobs import transform_jobs, transform_skills
from src.utils.db import get_connection, create_schema, row_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_RAW       = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
DATA_DB        = Path(__file__).resolve().parents[2] / "data" / "jobs.duckdb"


def run_extract(raw_dir: Path):
    logger.info("=" * 40)
    logger.info("FASE 1 — EXTRACT")
    logger.info("=" * 40)
    df_jobs   = extract_jobs(raw_dir)
    df_skills = extract_skills(raw_dir)
    return df_jobs, df_skills


def run_transform(df_jobs, df_skills):
    logger.info("=" * 40)
    logger.info("FASE 2 — TRANSFORM")
    logger.info("=" * 40)
    df_jobs_clean   = transform_jobs(df_jobs)
    df_skills_clean = transform_skills(df_skills, set(df_jobs_clean["job_link"]))
    return df_jobs_clean, df_skills_clean


def run_load(df_jobs_clean, df_skills_clean, db_path: Path, processed_dir: Path):
    logger.info("=" * 40)
    logger.info("FASE 3 — LOAD")
    logger.info("=" * 40)

    # ── Cargar a DuckDB ──────────────────────────────────────
    con = get_connection(db_path)
    create_schema(con)

    con.execute("INSERT OR IGNORE INTO jobs SELECT * FROM df_jobs_clean")
    total_jobs = row_count(con, "jobs")
    logger.info(f"Jobs en DB: {total_jobs:,}")

    con.execute("INSERT OR IGNORE INTO job_skills SELECT * FROM df_skills_clean")
    total_skills = row_count(con, "job_skills")
    logger.info(f"Skills en DB: {total_skills:,}")

    con.close()
    logger.info(f"DuckDB guardado en: {db_path}")

    # ── Exportar CSV a data/processed/ ───────────────────────
    processed_dir.mkdir(parents=True, exist_ok=True)

    jobs_out   = processed_dir / "jobs_clean.csv"
    skills_out = processed_dir / "job_skills_clean.csv"

    df_jobs_clean.to_csv(jobs_out, index=False)
    logger.info(f"CSV exportado: {jobs_out} ({len(df_jobs_clean):,} filas)")

    df_skills_clean.to_csv(skills_out, index=False)
    logger.info(f"CSV exportado: {skills_out} ({len(df_skills_clean):,} filas)")


def main(mode: str, raw_dir: Path, db_path: Path, processed_dir: Path) -> None:
    logger.info(f"Iniciando pipeline — modo: {mode}")

    if mode in ("extract", "full"):
        df_jobs, df_skills = run_extract(raw_dir)

    if mode in ("transform", "full"):
        if mode == "transform":
            df_jobs, df_skills = run_extract(raw_dir)
        df_jobs_clean, df_skills_clean = run_transform(df_jobs, df_skills)

    if mode in ("load", "full"):
        if mode == "load":
            df_jobs, df_skills             = run_extract(raw_dir)
            df_jobs_clean, df_skills_clean = run_transform(df_jobs, df_skills)
        run_load(df_jobs_clean, df_skills_clean, db_path, processed_dir)

    logger.info("Pipeline finalizado exitosamente ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline ETL de SkillScope.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["extract", "transform", "load", "full"],
        default="full",
        help="Fase del pipeline a ejecutar (default: full)"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(DATA_RAW),
        help="Ruta a data/raw"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DATA_DB),
        help="Ruta al archivo jobs.duckdb"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(DATA_PROCESSED),
        help="Ruta a data/processed"
    )
    args = parser.parse_args()
    main(args.mode, Path(args.raw_dir), Path(args.db_path), Path(args.processed_dir))