import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

DB_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "jobs.duckdb"


def get_connection(db_path: Path = DB_DEFAULT) -> duckdb.DuckDBPyConnection:
    """Retorna una conexión activa a DuckDB."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Conectando a DuckDB: {db_path}")
    return duckdb.connect(str(db_path))


def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Crea las tablas del esquema si no existen."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_link        VARCHAR PRIMARY KEY,
            job_title       VARCHAR,
            company         VARCHAR,
            job_location    VARCHAR,
            first_seen      DATE,
            search_city     VARCHAR,
            search_country  VARCHAR,
            search_position VARCHAR,
            job_level       VARCHAR,
            job_type        VARCHAR,
            week            DATE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS job_skills (
            job_link    VARCHAR,
            skill       VARCHAR,
            PRIMARY KEY (job_link, skill)
        )
    """)
    logger.info("Esquema creado/verificado correctamente.")


def row_count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    """Retorna el número de filas de una tabla."""
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]