"""
models/kmeans.py
========================================================
Modelo 1 - Agrupador de habilidades por perfil (K-Means)
========================================================

Que hace?
    Agrupa las ofertas de trabajo en K clusteres segun las skills
    que requieren. Cada cluster representa un "perfil" del mercado
    laboral (ej: perfil Data Engineer, perfil BI Analyst, etc.).

Entradas:
    - data/jobs.duckdb  ->  tablas: jobs, job_skills

Salidas:
    - Tabla de metricas: inertia, silhouette score
    - CSV con top skills por cluster
    - JSON con metricas para el reporte consolidado

Uso:
    python models/kmeans.py
    python models/kmeans.py --clusters 8 --top-skills 10
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MultiLabelBinarizer

# Fix encoding para Windows
sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -- Rutas ------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[1]
DB_PATH   = ROOT / "data" / "jobs.duckdb"
OUT_DIR   = ROOT / "models" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Parametros por defecto ------------------------------------
DEFAULT_K          = 6
DEFAULT_TOP_SKILLS = 10
DEFAULT_SAMPLE     = 50_000   # filas de jobs para no saturar RAM
RANDOM_STATE       = 42


# ============================================================
# 1. CARGA DE DATOS
# ============================================================
def load_data(db_path: Path, sample: int) -> pd.DataFrame:
    """
    Carga desde DuckDB un pivot job x skill (matriz binaria).
    Cada fila = una oferta de trabajo.
    Cada columna = una skill (1 si la requiere, 0 si no).
    """
    logger.info(f"Conectando a DuckDB: {db_path}")
    con = duckdb.connect(str(db_path), read_only=True)

    logger.info(f"Cargando muestra de {sample:,} ofertas...")
    links_df = con.execute(f"""
        SELECT DISTINCT job_link
        FROM jobs
        USING SAMPLE {sample} ROWS
    """).df()

    links = links_df["job_link"].tolist()
    links_csv = "','".join(links)

    skills_df = con.execute(f"""
        SELECT job_link, skill
        FROM job_skills
        WHERE job_link IN ('{links_csv}')
    """).df()

    con.close()

    logger.info(f"Skills cargadas: {len(skills_df):,} relaciones")
    return skills_df


# ============================================================
# 2. PREPROCESAMIENTO
# ============================================================
def build_matrix(skills_df: pd.DataFrame, min_skill_freq: int = 100):
    """
    Construye la matriz binaria job x skill.

    Parametros:
        min_skill_freq: elimina skills muy raras para reducir ruido
                        (solo conserva skills que aparecen >= N veces)

    Retorna:
        X      : np.ndarray  (n_jobs x n_skills)
        mlb    : MultiLabelBinarizer entrenado
        job_ids: lista de job_links en el mismo orden que X
    """
    freq = skills_df["skill"].value_counts()
    skills_validas = freq[freq >= min_skill_freq].index
    skills_df = skills_df[skills_df["skill"].isin(skills_validas)]

    logger.info(f"Skills unicas (freq >= {min_skill_freq}): {len(skills_validas)}")

    agrupado = (
        skills_df.groupby("job_link")["skill"]
        .apply(list)
        .reset_index()
    )

    job_ids = agrupado["job_link"].tolist()

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(agrupado["skill"])

    logger.info(f"Matriz construida: {X.shape[0]} ofertas x {X.shape[1]} skills")
    return X, mlb, job_ids


# ============================================================
# 3. ENTRENAMIENTO
# ============================================================
def train_kmeans(X: np.ndarray, k: int) -> KMeans:
    """Entrena K-Means con k clusteres."""
    logger.info(f"Entrenando K-Means con k={k}...")
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=RANDOM_STATE,
    )
    model.fit(X)
    logger.info("Entrenamiento completado.")
    return model


# ============================================================
# 4. METRICAS
# ============================================================
def calcular_metricas(X: np.ndarray, model: KMeans, k: int) -> dict:
    """
    Calcula las metricas de desempeno del modelo.

    Metricas:
        inertia        : suma de distancias al centroide (menor = mejor)
        silhouette     : que tan bien separados estan los clusteres
                         [-1, 1] -- valores > 0.2 son aceptables
        distribucion   : cantidad de ofertas por cluster
    """
    logger.info("Calculando metricas...")

    inertia = model.inertia_

    sample_size = min(10_000, X.shape[0])
    idx = np.random.choice(X.shape[0], sample_size, replace=False)
    sil = silhouette_score(X[idx], model.labels_[idx], metric="euclidean")

    labels, counts = np.unique(model.labels_, return_counts=True)
    dist = {f"cluster_{int(l)}": int(c) for l, c in zip(labels, counts)}

    metricas = {
        "modelo":            "K-Means - Agrupador de habilidades por perfil",
        "k_clusters":        k,
        "inertia":           round(float(inertia), 2),
        "silhouette_score":  round(float(sil), 4),
        "distribucion":      dist,
        "total_ofertas":     int(X.shape[0]),
        "total_skills":      int(X.shape[1]),
    }

    logger.info(f"  Inertia        : {inertia:,.2f}")
    logger.info(f"  Silhouette     : {sil:.4f}")
    return metricas


# ============================================================
# 5. INTERPRETACION DE CLUSTERES
# ============================================================
def top_skills_por_cluster(
    model: KMeans, mlb: MultiLabelBinarizer, top_n: int
) -> pd.DataFrame:
    """
    Para cada cluster retorna las top_n skills mas representativas
    (las que tienen mayor peso en el centroide del cluster).
    """
    centroids   = model.cluster_centers_
    skill_names = mlb.classes_

    rows = []
    for cluster_id, centroid in enumerate(centroids):
        top_idx = np.argsort(centroid)[::-1][:top_n]
        for rank, idx in enumerate(top_idx, 1):
            rows.append({
                "cluster":   cluster_id,
                "rank":      rank,
                "skill":     skill_names[idx],
                "peso":      round(float(centroid[idx]), 4),
            })

    return pd.DataFrame(rows)


def nombrar_clusters(top_df: pd.DataFrame) -> dict:
    """
    Asigna un nombre descriptivo a cada cluster
    basandose en su skill #1 y #2.
    """
    nombres = {}
    for cluster_id in top_df["cluster"].unique():
        top2 = (
            top_df[top_df["cluster"] == cluster_id]
            .nsmallest(2, "rank")["skill"]
            .tolist()
        )
        nombres[cluster_id] = " + ".join(top2)
    return nombres


# ============================================================
# 6. GUARDADO DE RESULTADOS
# ============================================================
def guardar_resultados(metricas: dict, top_df: pd.DataFrame) -> None:
    metrics_path = OUT_DIR / "metrics_model1_kmeans.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    logger.info(f"Metricas guardadas: {metrics_path}")

    csv_path = OUT_DIR / "clusters_skills.csv"
    top_df.to_csv(csv_path, index=False)
    logger.info(f"Clusteres guardados: {csv_path}")


# ============================================================
# 7. REPORTE EN CONSOLA
# ============================================================
def imprimir_reporte(metricas: dict, top_df: pd.DataFrame, nombres: dict) -> None:
    sep = "-" * 55
    print(f"\n{sep}")
    print("  MODELO 1 - Agrupador de habilidades por perfil")
    print(f"{sep}")
    print(f"  K clusteres      : {metricas['k_clusters']}")
    print(f"  Total ofertas    : {metricas['total_ofertas']:,}")
    print(f"  Total skills     : {metricas['total_skills']:,}")
    print(f"{sep}")
    print("  METRICAS DE DESEMPENO")
    print(f"{sep}")
    print(f"  {'Metrica':<25} {'Valor':>15}")
    print(f"  {'-'*25} {'-'*15}")
    print(f"  {'Inertia':<25} {metricas['inertia']:>15,.2f}")
    print(f"  {'Silhouette Score':<25} {metricas['silhouette_score']:>15.4f}")
    print(f"{sep}")
    print("  TOP SKILLS POR CLUSTER")
    print(f"{sep}")
    for cid in sorted(top_df["cluster"].unique()):
        nombre = nombres.get(cid, f"Cluster {cid}")
        skills = (
            top_df[top_df["cluster"] == cid]
            .nsmallest(5, "rank")["skill"]
            .tolist()
        )
        print(f"  Cluster {cid} [{nombre}]")
        print(f"    {', '.join(skills)}")
    print(f"{sep}\n")


# ============================================================
# 8. MAIN
# ============================================================
def main(k: int = DEFAULT_K, top_skills: int = DEFAULT_TOP_SKILLS,
         sample: int = DEFAULT_SAMPLE) -> dict:

    # 1. Cargar datos
    skills_df = load_data(DB_PATH, sample)

    # 2. Construir matriz binaria
    X, mlb, job_ids = build_matrix(skills_df)

    # 3. Entrenar modelo
    model = train_kmeans(X, k)

    # 4. Metricas
    metricas = calcular_metricas(X, model, k)

    # 5. Interpretar clusteres
    top_df  = top_skills_por_cluster(model, mlb, top_skills)
    nombres = nombrar_clusters(top_df)

    # 6. Guardar
    guardar_resultados(metricas, top_df)

    # 7. Reporte
    imprimir_reporte(metricas, top_df, nombres)

    return metricas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modelo 1 - Agrupador de habilidades por perfil (K-Means)"
    )
    parser.add_argument("--clusters",   type=int, default=DEFAULT_K,
                        help=f"Numero de clusteres (default: {DEFAULT_K})")
    parser.add_argument("--top-skills", type=int, default=DEFAULT_TOP_SKILLS,
                        help=f"Top N skills por cluster (default: {DEFAULT_TOP_SKILLS})")
    parser.add_argument("--sample",     type=int, default=DEFAULT_SAMPLE,
                        help=f"Muestra de ofertas (default: {DEFAULT_SAMPLE:,})")
    args = parser.parse_args()

    main(k=args.clusters, top_skills=args.top_skills, sample=args.sample)