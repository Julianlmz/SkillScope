"""
src/ingestion/extract.py
Carga los archivos CSV crudos del dataset de Kaggle.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

JOBS_COLS = [
    "job_link", "job_title", "company", "job_location",
    "first_seen", "search_city", "search_country",
    "search_position", "job_level", "job_type",
]


def extract_jobs(raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    """Carga linkedin_job_postings.csv y retorna un DataFrame."""
    path = raw_dir / "linkedin_job_postings.csv"
    logger.info(f"Leyendo jobs desde: {path}")
    df = pd.read_csv(path, usecols=JOBS_COLS, dtype=str)
    logger.info(f"Jobs cargados: {len(df):,} registros")
    return df


def extract_skills(raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    """Carga job_skills.csv y retorna un DataFrame."""
    path = raw_dir / "job_skills.csv"
    logger.info(f"Leyendo skills desde: {path}")
    df = pd.read_csv(path, dtype=str)
    logger.info(f"Skills cargadas: {len(df):,} registros")
    return df


def main(raw_dir: str = None) -> None:
    raw_dir = Path(raw_dir) if raw_dir else DATA_RAW
    df_jobs   = extract_jobs(raw_dir)
    df_skills = extract_skills(raw_dir)
    print(df_jobs.head(3).to_string())
    print(df_skills.head(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrae los CSV crudos del dataset.")
    parser.add_argument("--raw-dir", type=str, default=None,
                        help="Ruta a data/raw (opcional, usa la default del proyecto)")
    args = parser.parse_args()
    main(args.raw_dir)