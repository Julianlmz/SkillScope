"""
src/transformation/clean_jobs.py
Limpia y transforma los DataFrames de jobs y skills.
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


def transform_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas las transformaciones al DataFrame de jobs:
    - Filtra por United States
    - Convierte fechas
    - Agrega columna week
    - Limpia texto
    - Elimina duplicados
    """
    # 1. Filtrar US
    before = len(df)
    df = df[df["search_country"] == "United States"].copy()
    logger.info(f"Filtrado US: {before:,} → {len(df):,} registros")

    # 2. Convertir fechas
    df["first_seen"] = pd.to_datetime(df["first_seen"])

    # 3. Columna week (lunes de cada semana)
    df["week"] = (
        df["first_seen"] - pd.to_timedelta(df["first_seen"].dt.dayofweek, unit="D")
    ).dt.date

    # 4. Limpiar texto
    for col in ["job_title", "company", "job_level", "job_type"]:
        df[col] = df[col].str.strip()

    # 5. Eliminar duplicados
    before = len(df)
    df = df.drop_duplicates(subset="job_link")
    logger.info(f"Duplicados eliminados: {before - len(df)}")

    logger.info(f"Jobs limpios: {len(df):,} registros")
    return df


def transform_skills(df_skills: pd.DataFrame, jobs_us: set) -> pd.DataFrame:
    """
    Aplica todas las transformaciones al DataFrame de skills:
    - Elimina nulos
    - Filtra solo jobs de US
    - Explota skills (una fila por skill)
    - Limpia texto
    - Elimina duplicados
    """
    # 1. Eliminar nulos
    df = df_skills.dropna(subset=["job_skills"]).copy()

    # 2. Filtrar solo jobs de US
    df = df[df["job_link"].isin(jobs_us)]
    logger.info(f"Skills filtradas a US: {len(df):,} registros")

    # 3. Explotar skills
    df["job_skills"] = df["job_skills"].str.split(", ")
    df = df.explode("job_skills")
    df["job_skills"] = df["job_skills"].str.strip()
    df = df[df["job_skills"] != ""]

    # 4. Eliminar duplicados
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Duplicados en skills eliminados: {before - len(df)}")

    logger.info(f"Skills explotadas: {len(df):,} registros")
    return df


def main(raw_dir: str = None) -> None:
    from src.ingestion.extract import extract_jobs, extract_skills

    raw_path = Path(raw_dir) if raw_dir else None

    df_jobs   = extract_jobs(raw_path) if raw_path else extract_jobs()
    df_skills = extract_skills(raw_path) if raw_path else extract_skills()

    df_jobs_clean   = transform_jobs(df_jobs)
    df_skills_clean = transform_skills(df_skills, set(df_jobs_clean["job_link"]))

    print("\n── Jobs sample ──")
    print(df_jobs_clean.head(3).to_string())
    print("\n── Skills sample ──")
    print(df_skills_clean.head(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transforma y limpia jobs y skills.")
    parser.add_argument("--raw-dir", type=str, default=None,
                        help="Ruta a data/raw (opcional)")
    args = parser.parse_args()
    main(args.raw_dir)