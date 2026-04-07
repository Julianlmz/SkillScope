"""
SkillScope - Dashboard Streamlit
Ejecutar: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import duckdb
import random
import numpy as np
from pathlib import Path
from datetime import date, timedelta

# ── Configuracion ─────────────────────────────────────────────
st.set_page_config(
    page_title="SkillScope",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #070d1a !important;
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] > div {
    background: #0d1526 !important;
    border-right: 1px solid #1e2d45 !important;
    padding-top: 1.5rem;
}
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1526, #111d35);
    border: 1px solid #1e2d45;
    border-radius: 14px;
    padding: 1.2rem 1.4rem !important;
    transition: all .25s ease;
    box-shadow: 0 4px 24px rgba(0,0,0,.3);
}
[data-testid="metric-container"]:hover {
    border-color: #3b82f6;
    box-shadow: 0 4px 24px rgba(59,130,246,.15);
    transform: translateY(-2px);
}
[data-testid="stMetricLabel"] p {
    font-size: .7rem !important;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #475569 !important;
    font-weight: 500;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.7rem !important;
    color: #e2e8f0 !important;
}
h1 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0 !important;
}
h2 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1rem !important;
    color: #94a3b8 !important;
    font-weight: 400 !important;
    letter-spacing: .04em;
}
hr { border-color: #1e2d45 !important; margin: 1.2rem 0 !important; }
.stTabs [data-baseweb="tab-list"] {
    background: #0d1526;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .75rem;
    letter-spacing: .05em;
    color: #475569;
    border-radius: 7px;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: #1d4ed8 !important;
    color: #fff !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }
.stSelectbox label, .stMultiSelect label, .stSlider label, .stDateInput label {
    font-size: .72rem !important;
    color: #475569 !important;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 600;
}
.stDownloadButton button {
    background: linear-gradient(135deg, #1d4ed8, #0369a1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .78rem !important;
    padding: .6rem 1.2rem !important;
}
[data-baseweb="tag"] { background: #1d4ed8 !important; border-radius: 6px !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Plotly layout base ────────────────────────────────────────
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8", size=11),
    xaxis=dict(gridcolor="#1e2d45", zerolinecolor="#1e2d45", showline=False),
    yaxis=dict(gridcolor="#1e2d45", zerolinecolor="#1e2d45", showline=False),
    margin=dict(l=4, r=4, t=36, b=4),
)
CMAP  = ["#0f2044", "#1d4ed8", "#3b82f6", "#93c5fd"]
CMAP2 = ["#0f2044", "#0891b2", "#06b6d4", "#a5f3fc"]
CMAP3 = ["#0f2044", "#7c3aed", "#8b5cf6", "#ddd6fe"]
CMAP4 = ["#0f2044", "#059669", "#10b981", "#6ee7b7"]
QUAL  = ["#3b82f6", "#06b6d4", "#8b5cf6", "#f59e0b",
         "#10b981", "#f43f5e", "#ec4899", "#84cc16"]

MAX_JOBS   = 150_000
MAX_SKILLS = 500_000


# ── Funcion para parsear fechas de DuckDB ─────────────────────
def parse_fecha(serie):
    """
    Convierte cualquier formato de fecha que devuelva DuckDB a
    datetime64[ns] tz-naive limpio, sin microsegundos ni tz.

    Casos manejados:
    - datetime64 con tz (UTC o cualquier otro)
    - datetime64 sin tz pero con precision alta (us, ns)
    - entero epoch en ms  (> 1e12)
    - entero epoch en s   (> 1e9)
    - string 'YYYY-MM-DD' o 'YYYY-MM-DD HH:MM:SS'
    """
    # 1) Ya es datetime → quitar tz y truncar a dia
    if pd.api.types.is_datetime64_any_dtype(serie):
        s = serie
        if s.dt.tz is not None:
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        # Forzar a datetime64[ns] para evitar problemas de precision
        s = s.astype("datetime64[ns]")
        return s

    # 2) Intentar como numero (epoch)
    num = pd.to_numeric(serie, errors="coerce")
    validos = num.dropna()
    if len(validos) > 0:
        mediana = validos.median()
        try:
            if mediana > 1e12:
                s = pd.to_datetime(num, unit="ms", errors="coerce")
            elif mediana > 1e9:
                s = pd.to_datetime(num, unit="s", errors="coerce")
            else:
                s = pd.to_datetime(serie, errors="coerce")
            if s.dt.tz is not None:
                s = s.dt.tz_convert("UTC").dt.tz_localize(None)
            return s.astype("datetime64[ns]")
        except Exception:
            pass

    # 3) Fallback: parsear como string
    s = pd.to_datetime(serie, errors="coerce")
    if s.dt.tz is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.astype("datetime64[ns]")


# ── Carga de datos ────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando datos...")
def load_data():
    db_candidates = [
        Path("data/jobs.duckdb"),
        Path("../data/jobs.duckdb"),
        Path(__file__).resolve().parent / "data" / "jobs.duckdb",
    ]
    db_path = next((p for p in db_candidates if p.exists()), None)

    if db_path:
        try:
            con = duckdb.connect(str(db_path), read_only=True)

            total_jobs   = con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            total_skills = con.execute("SELECT COUNT(*) FROM job_skills").fetchone()[0]

            # ── Detectar tipo real de first_seen ──────────────
            col_type_row = con.execute("""
                SELECT data_type FROM information_schema.columns
                WHERE table_name = 'jobs' AND column_name = 'first_seen'
            """).fetchone()
            col_type = col_type_row[0].upper() if col_type_row else "UNKNOWN"

            # Muestra de 3 valores raw para debug
            raw_sample = con.execute("SELECT first_seen FROM jobs LIMIT 3").fetchall()
            raw_vals   = [r[0] for r in raw_sample]

            # ── Elegir la expresion SQL correcta segun el tipo ─
            # TIMESTAMP / TIMESTAMPTZ / DATE  → castear directo a DATE
            # BIGINT / INTEGER que sea epoch en microsegundos (formato interno DuckDB)
            #   → epoch_us:  to_timestamp(first_seen / 1000000)::DATE
            # BIGINT epoch en milisegundos
            #   → epoch_ms:  to_timestamp(first_seen / 1000)::DATE
            # BIGINT epoch en segundos
            #   → epoch_s:   to_timestamp(first_seen)::DATE
            # VARCHAR  → strptime o TRY_CAST
            if any(t in col_type for t in ("TIMESTAMP", "DATE", "TIME")):
                date_expr = "CAST(first_seen AS DATE)"
            elif any(t in col_type for t in ("BIGINT", "HUGEINT", "INTEGER", "INT")):
                # Detectar escala por magnitud del primer valor
                first_num = raw_vals[0] if raw_vals else 0
                try:
                    first_num = int(first_num)
                except Exception:
                    first_num = 0
                if first_num > 1e15:
                    # microsegundos epoch (formato interno DuckDB TIMESTAMP guardado como BIGINT)
                    date_expr = "CAST(to_timestamp(first_seen / 1000000.0) AS DATE)"
                elif first_num > 1e12:
                    # milisegundos epoch
                    date_expr = "CAST(to_timestamp(first_seen / 1000.0) AS DATE)"
                elif first_num > 1e9:
                    # segundos epoch
                    date_expr = "CAST(to_timestamp(first_seen) AS DATE)"
                else:
                    # dias desde epoch (raro pero posible)
                    date_expr = "CAST(DATE '1970-01-01' + INTERVAL (first_seen) DAYS AS DATE)"
            elif "VARCHAR" in col_type or "TEXT" in col_type or "CHAR" in col_type:
                # Intentar parsear como fecha string
                date_expr = "TRY_CAST(first_seen AS DATE)"
            else:
                # Ultimo recurso: dejar que DuckDB decida
                date_expr = "TRY_CAST(first_seen AS DATE)"

            # Debug sidebar
            st.sidebar.caption(
                f"Tipo columna: `{col_type}`\n\n"
                f"Expr SQL: `{date_expr}`\n\n"
                f"Vals raw: `{raw_vals[:2]}`"
            )

            # ── Query principal ───────────────────────────────
            jobs = con.execute(f"""
                SELECT job_link, job_title, company, job_location,
                       {date_expr} AS first_seen,
                       search_city, search_country,
                       job_level, job_type
                FROM jobs
                USING SAMPLE {MAX_JOBS} ROWS
            """).df()

            # Solo skills de esos jobs
            links_csv = "','".join(jobs["job_link"].astype(str).tolist())
            skills = con.execute(f"""
                SELECT job_link, skill
                FROM job_skills
                WHERE job_link IN ('{links_csv}')
                LIMIT {MAX_SKILLS}
            """).df()

            con.close()

            # Confirmar fechas en sidebar
            if len(jobs) > 0 and "first_seen" in jobs.columns:
                parsed = pd.to_datetime(jobs["first_seen"], errors="coerce").dropna()
                if len(parsed):
                    st.sidebar.caption(
                        f"Fechas parseadas: `{parsed.min().date()}` → `{parsed.max().date()}`"
                    )

            return jobs, skills, True, total_jobs, total_skills

        except Exception as e:
            st.warning(f"Error leyendo DuckDB: {e}. Usando datos de muestra.")

    # ── Datos de muestra ─────────────────────────────────────
    random.seed(42)
    np.random.seed(42)
    n = 8_000

    companies = [
        "Google", "Amazon", "Microsoft", "Meta", "Apple", "Netflix",
        "Salesforce", "IBM", "Oracle", "Stripe", "Airbnb", "Uber",
        "Lyft", "Zoom", "Twilio", "Snowflake", "Databricks", "Palantir",
        "SpaceX", "Tesla", "Adobe", "Nvidia", "Intel", "Qualcomm", "Broadcom",
    ]
    titles = [
        "Data Engineer", "Data Scientist", "Software Engineer", "ML Engineer",
        "Data Analyst", "Backend Developer", "DevOps Engineer", "Cloud Architect",
        "BI Analyst", "Product Manager", "Frontend Developer",
        "Full Stack Developer", "Platform Engineer", "Analytics Engineer",
    ]
    levels = ["Mid-Senior level", "Associate", "Director", "Entry level", "Executive"]
    types_ = ["Full-time", "Contract", "Part-time"]
    cities = [
        "New York", "San Francisco", "Seattle", "Austin", "Chicago",
        "Boston", "Los Angeles", "Denver", "Atlanta", "Dallas", "Miami", "San Jose",
    ]

    base  = date(2024, 1, 1)
    dates = [base + timedelta(days=int(d)) for d in np.random.randint(0, 365, n)]

    jobs = pd.DataFrame({
        "job_link":       [f"https://linkedin.com/jobs/view/{i}" for i in range(n)],
        "job_title":      random.choices(titles,    k=n),
        "company":        random.choices(companies, k=n),
        "job_location":   [f"{c}, United States" for c in random.choices(cities, k=n)],
        "first_seen":     dates,
        "search_city":    random.choices(cities,    k=n),
        "search_country": ["United States"] * n,
        "job_level":      random.choices(levels, k=n, weights=[50, 20, 10, 15, 5]),
        "job_type":       random.choices(types_, k=n, weights=[78, 14, 8]),
    })

    skill_pool = (
        ["Communication", "Teamwork", "Leadership", "Problem Solving", "Management",
         "Collaboration", "Presentation", "Critical Thinking", "Adaptability", "Creativity"] +
        ["Python", "SQL", "Java", "JavaScript", "TypeScript", "Go", "Rust", "C++", "Scala", "R",
         "Data Analysis", "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
         "Excel", "Tableau", "Power BI", "Spark", "Kafka", "Airflow", "dbt", "Databricks",
         "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Git", "CI/CD",
         "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "Snowflake", "BigQuery",
         "React", "Angular", "Vue", "Node.js", "Django", "FastAPI", "Spring"]
    )
    w = [0.85] * 10 + [0.4] * (len(skill_pool) - 10)

    rows = []
    for link in jobs["job_link"]:
        chosen = list(set(random.choices(skill_pool, weights=w, k=random.randint(4, 9))))
        for s in chosen:
            rows.append({"job_link": link, "skill": s})

    skills = pd.DataFrame(rows)
    return jobs, skills, False, n, len(skills)


jobs_raw, skills_raw, is_real, TOTAL_JOBS, TOTAL_SKILLS = load_data()

# ── Preparacion de fechas ─────────────────────────────────────
jobs_raw = jobs_raw.copy()

# Paso 1: parsear a datetime64[ns] tz-naive
jobs_raw["first_seen"] = parse_fecha(jobs_raw["first_seen"])

# Paso 2: eliminar invalidos
jobs_raw = jobs_raw.dropna(subset=["first_seen"])

# Paso 3: convertir a fecha pura (solo año-mes-dia, sin hora)
# Usar .dt.floor("D") es mas seguro que .normalize() con precision alta
jobs_raw["first_seen"] = jobs_raw["first_seen"].dt.floor("D")

# Paso 4: semana (lunes de la semana correspondiente) como datetime
# Calcular offset de dias desde el lunes de esa semana
dow = jobs_raw["first_seen"].dt.dayofweek  # 0=lunes .. 6=domingo
jobs_raw["week"] = jobs_raw["first_seen"] - pd.to_timedelta(dow, unit="D")

# Paso 5: mes como string "YYYY-MM" para agrupar correctamente
jobs_raw["month"] = jobs_raw["first_seen"].dt.to_period("M").astype(str)

# Extraer ciudad limpia
jobs_raw["city_clean"] = (
    jobs_raw["job_location"].str.extract(r"^([^,]+)")[0].str.strip()
)

# ── Preparacion de skills ─────────────────────────────────────
skills_raw = skills_raw.copy()

if "skill" not in skills_raw.columns and "job_skills" in skills_raw.columns:
    skills_raw = skills_raw.rename(columns={"job_skills": "skill"})

# Normalizar texto
skills_raw["skill"] = skills_raw["skill"].str.strip().str.title()

# Clasificacion de skills
SOFT_KW = {
    "communication", "teamwork", "leadership", "problem solving", "management",
    "collaboration", "presentation", "critical thinking", "adaptability", "creativity",
    "customer service", "training", "sales", "nursing", "time management",
    "organization", "interpersonal", "negotiation", "mentoring", "coaching",
    "communication skills", "problemsolving", "problem-solving", "attention to detail",
}
TECH_KW = {
    "python", "sql", "java", "javascript", "typescript", "go", "rust", "c++", "scala", "r",
    "data analysis", "machine learning", "deep learning", "nlp", "computer vision",
    "excel", "tableau", "power bi", "spark", "kafka", "airflow", "dbt", "databricks",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "git", "ci/cd",
    "postgresql", "mongodb", "redis", "elasticsearch", "snowflake", "bigquery",
    "react", "angular", "vue", "node", "django", "fastapi", "spring", "flask",
    "microsoft office suite", "microsoft office",
}


def classify_skill(s):
    sl = s.lower()
    if any(k in sl for k in SOFT_KW):
        return "Blanda"
    if any(k in sl for k in TECH_KW):
        return "Tecnica"
    return "Blanda"


skills_raw["category"] = skills_raw["skill"].apply(classify_skill)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 SkillScope")
    st.markdown(
        '<span style="background:#1d4ed8;color:#fff;border-radius:6px;'
        'padding:2px 8px;font-size:.7rem;font-family:monospace">v1.0</span> '
        '<span style="background:#0d1526;color:#475569;border:1px solid #1e2d45;'
        'border-radius:6px;padding:2px 8px;font-size:.7rem;font-family:monospace">2024</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### Filtros")

    all_levels = sorted(jobs_raw["job_level"].dropna().unique())
    sel_levels = st.multiselect("Nivel de experiencia", all_levels, default=all_levels)

    all_types = sorted(jobs_raw["job_type"].dropna().unique())
    sel_types = st.multiselect("Tipo de empleo", all_types, default=all_types)

    all_titles = sorted(jobs_raw["job_title"].dropna().unique())
    sel_titles = st.multiselect(
        "Rol / Titulo (opcional)", all_titles, default=[],
        placeholder="Todos los roles"
    )

    # ── FIX: calcular fecha min/max de forma robusta ──────────
    valid_dates = jobs_raw["first_seen"].dropna()
    try:
        date_min = valid_dates.min().date()
        date_max = valid_dates.max().date()
        # Sanity check: fechas deben ser razonables (post-2000)
        if date_min.year < 2000 or date_max.year < 2000:
            raise ValueError("Fechas fuera de rango razonable")
    except Exception:
        date_min = date(2024, 1, 1)
        date_max = date(2024, 12, 31)

    date_range = st.date_input(
        "Rango de fechas",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

    st.markdown("---")
    top_n = st.slider("Top N skills", 5, 30, 15)

    if is_real:
        st.success(
            f"DuckDB conectado\n\n"
            f"{TOTAL_JOBS:,} ofertas reales\n\n"
            f"(muestra de {MAX_JOBS:,})"
        )
    else:
        st.info("Datos de muestra.\nEjecuta el pipeline ETL para datos reales.")

# ── Aplicar filtros ───────────────────────────────────────────
# FIX: manejar date_range que puede ser tuple o date unico,
#      y asegurar que los timestamps sean tz-naive para comparar
try:
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        d0 = pd.Timestamp(date_range[0]).normalize()
        d1 = pd.Timestamp(date_range[1]).normalize()
    else:
        # El usuario solo eligio una fecha (rango incompleto)
        single = pd.Timestamp(date_range[0] if isinstance(date_range, (list, tuple)) else date_range).normalize()
        d0 = single
        d1 = single
except Exception:
    d0 = jobs_raw["first_seen"].min()
    d1 = jobs_raw["first_seen"].max()

# Garantizar tz-naive en d0/d1
if hasattr(d0, "tz") and d0.tz is not None:
    d0 = d0.tz_localize(None)
if hasattr(d1, "tz") and d1.tz is not None:
    d1 = d1.tz_localize(None)

if not sel_levels:
    sel_levels = all_levels
if not sel_types:
    sel_types = all_types

mask = (
    jobs_raw["job_level"].isin(sel_levels) &
    jobs_raw["job_type"].isin(sel_types) &
    (jobs_raw["first_seen"] >= d0) &
    (jobs_raw["first_seen"] <= d1)
)
if sel_titles:
    mask &= jobs_raw["job_title"].isin(sel_titles)

jobs_f   = jobs_raw[mask]
skills_f = skills_raw[skills_raw["job_link"].isin(set(jobs_f["job_link"]))]

# ── Header ────────────────────────────────────────────────────
st.markdown("# SkillScope")
st.markdown("## Tendencias de habilidades · Mercado laboral EE.UU. · LinkedIn 2024")
st.markdown("---")

# ── KPIs ─────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total ofertas",    f"{TOTAL_JOBS:,}")
k2.metric("En muestra",       f"{len(jobs_f):,}")
k3.metric("Skills unicas",    f"{skills_f['skill'].nunique():,}")
k4.metric("Empresas",         f"{jobs_f['company'].nunique():,}")
k5.metric("Ciudades",         f"{jobs_f['city_clean'].nunique():,}")

st.markdown("---")

if len(jobs_f) == 0:
    st.warning("Sin datos con los filtros actuales. Ajusta los filtros del sidebar.")
    # Debug extra: mostrar rango de fechas en el dataset vs filtro aplicado
    if len(jobs_raw) > 0:
        st.info(
            f"Fechas en dataset: `{jobs_raw['first_seen'].min().date()}` → "
            f"`{jobs_raw['first_seen'].max().date()}`  |  "
            f"Filtro aplicado: `{d0.date()}` → `{d1.date()}`"
        )
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Skills",
    "📈  Tendencias",
    "🏢  Empresas",
    "🗺️  Geografia",
    "🔬  Detalles",
])

# ════════════════════════════════════════
# TAB 1 — SKILLS
# ════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("#### Top skills mas demandadas")
        top_sk = (
            skills_f.groupby("skill")["job_link"].nunique()
            .nlargest(top_n).reset_index(name="ofertas")
            .sort_values("ofertas")
        )
        fig = px.bar(
            top_sk, x="ofertas", y="skill", orientation="h",
            color="ofertas", color_continuous_scale=CMAP,
            labels={"skill": "", "ofertas": "Ofertas"},
        )
        fig.update_layout(**PL, coloraxis_showscale=False, height=max(380, top_n * 26))
        fig.update_traces(marker_line_width=0,
                          hovertemplate="<b>%{y}</b><br>%{x:,} ofertas<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Tecnicas vs Blandas")
        cat_c = skills_f.groupby("category")["job_link"].nunique().reset_index(name="n")
        fig2  = px.pie(
            cat_c, names="category", values="n",
            color_discrete_sequence=["#1d4ed8", "#06b6d4"], hole=0.6,
        )
        fig2.update_layout(**PL, height=220, showlegend=True,
                           legend=dict(orientation="h", y=-0.05, font=dict(size=11)))
        fig2.update_traces(textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Top tecnicas")
        top_t = (
            skills_f[skills_f["category"] == "Tecnica"]
            .groupby("skill")["job_link"].nunique()
            .nlargest(10).reset_index(name="n").sort_values("n")
        )
        fig3 = px.bar(
            top_t, x="n", y="skill", orientation="h",
            color_discrete_sequence=["#06b6d4"],
            labels={"skill": "", "n": "Ofertas"},
        )
        fig3.update_layout(**PL, height=300, showlegend=False)
        fig3.update_traces(marker_line_width=0,
                           hovertemplate="<b>%{y}</b><br>%{x:,}<extra></extra>")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Top skills por nivel de experiencia")
    lvl_opts = sorted(jobs_f["job_level"].dropna().unique())
    if lvl_opts:
        sel_lvl = st.selectbox("Nivel", lvl_opts, key="lvl1")
        lk_lvl  = set(jobs_f[jobs_f["job_level"] == sel_lvl]["job_link"])
        sk_lvl  = (
            skills_f[skills_f["job_link"].isin(lk_lvl)]
            .groupby("skill")["job_link"].nunique()
            .nlargest(10).reset_index(name="n").sort_values("n")
        )
        fig4 = px.bar(
            sk_lvl, x="n", y="skill", orientation="h",
            color="n", color_continuous_scale=CMAP3,
            labels={"skill": "", "n": "Ofertas"},
        )
        fig4.update_layout(**PL, coloraxis_showscale=False, height=340)
        fig4.update_traces(marker_line_width=0,
                           hovertemplate="<b>%{y}</b><br>%{x:,}<extra></extra>")
        st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════
# TAB 2 — TENDENCIAS
# ════════════════════════════════════════
with tab2:
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("#### Publicaciones por semana")
        weekly = (
            jobs_f.groupby("week").size()
            .reset_index(name="ofertas")
            .sort_values("week")
        )
        # Asegurar que week sea datetime para plotly
        weekly["week"] = pd.to_datetime(weekly["week"])
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=weekly["week"],
            y=weekly["ofertas"],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#3b82f6", width=2),
            fillcolor="rgba(59,130,246,.12)",
            hovertemplate="%{x|%d %b %Y}<br><b>%{y:,}</b> ofertas<extra></extra>",
        ))
        fig5.update_layout(**PL, height=300)
        fig5.update_xaxes(
            tickformat="%b %Y",
            dtick="M1",
            tickangle=-30,
            gridcolor="#1e2d45",
            type="date",
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.markdown("#### Por mes")
        monthly = (
            jobs_f.groupby("month").size()
            .reset_index(name="n")
            .sort_values("month")  # string YYYY-MM ordena lexicograficamente = cronologico
        )
        fig6 = px.bar(
            monthly, x="month", y="n",
            color_discrete_sequence=["#06b6d4"],
            labels={"month": "Mes", "n": "Ofertas"},
        )
        fig6.update_layout(**PL, height=300)
        fig6.update_xaxes(tickangle=-45, tickfont=dict(size=10), type="category")
        fig6.update_traces(marker_line_width=0,
                           hovertemplate="%{x}<br><b>%{y:,}</b> ofertas<extra></extra>")
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Evolucion semanal de skills")
    top12 = (
        skills_f.groupby("skill")["job_link"].nunique()
        .nlargest(12).index.tolist()
    )
    sel_evo = st.multiselect("Skills a comparar", top12, default=top12[:5])
    if sel_evo:
        evo = (
            skills_f[skills_f["skill"].isin(sel_evo)]
            .merge(jobs_f[["job_link", "week"]], on="job_link")
            .groupby(["week", "skill"])["job_link"].nunique()
            .reset_index(name="n")
            .sort_values("week")
        )
        evo["week"] = pd.to_datetime(evo["week"])
        fig7 = px.line(
            evo, x="week", y="n", color="skill",
            color_discrete_sequence=QUAL,
            labels={"week": "Semana", "n": "Ofertas", "skill": "Skill"},
        )
        fig7.update_layout(
            **PL, height=380,
            legend=dict(orientation="h", y=1.08, font=dict(size=10)),
        )
        fig7.update_xaxes(tickformat="%b %Y", dtick="M1", tickangle=-30, type="date")
        fig7.update_traces(
            hovertemplate="%{x|%d %b}<br><b>%{y:,}</b><extra>%{fullData.name}</extra>"
        )
        st.plotly_chart(fig7, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Top skills por tipo de empleo")
    jtype_opts = sorted(jobs_f["job_type"].dropna().unique())
    if jtype_opts:
        jt_sel = st.selectbox("Tipo", jtype_opts, key="jtype2")
        lk_jt  = set(jobs_f[jobs_f["job_type"] == jt_sel]["job_link"])
        sk_jt  = (
            skills_f[skills_f["job_link"].isin(lk_jt)]
            .groupby("skill")["job_link"].nunique()
            .nlargest(12).reset_index(name="n").sort_values("n")
        )
        fig8 = px.bar(
            sk_jt, x="n", y="skill", orientation="h",
            color_discrete_sequence=["#f59e0b"],
            labels={"skill": "", "n": "Ofertas"},
        )
        fig8.update_layout(**PL, height=360)
        fig8.update_traces(marker_line_width=0,
                           hovertemplate="<b>%{y}</b><br>%{x:,}<extra></extra>")
        st.plotly_chart(fig8, use_container_width=True)

# ════════════════════════════════════════
# TAB 3 — EMPRESAS
# ════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### Top 20 empresas")
        top_cmp = (
            jobs_f.groupby("company")["job_link"].nunique()
            .nlargest(20).reset_index(name="ofertas").sort_values("ofertas")
        )
        fig9 = px.bar(
            top_cmp, x="ofertas", y="company", orientation="h",
            color="ofertas", color_continuous_scale=CMAP,
            labels={"company": "", "ofertas": "Ofertas"},
        )
        fig9.update_layout(**PL, coloraxis_showscale=False, height=530)
        fig9.update_traces(marker_line_width=0,
                           hovertemplate="<b>%{y}</b><br>%{x:,}<extra></extra>")
        st.plotly_chart(fig9, use_container_width=True)

    with col2:
        st.markdown("#### Skills por empresa")
        cmp_sel = st.selectbox(
            "Empresa", sorted(jobs_f["company"].dropna().unique()), key="cmp1"
        )
        lk_cmp = set(jobs_f[jobs_f["company"] == cmp_sel]["job_link"])
        sk_cmp = (
            skills_f[skills_f["job_link"].isin(lk_cmp)]
            .groupby("skill")["job_link"].nunique()
            .nlargest(12).reset_index(name="n").sort_values("n")
        )
        fig10 = px.bar(
            sk_cmp, x="n", y="skill", orientation="h",
            color="n", color_continuous_scale=CMAP4,
            labels={"skill": "", "n": "Ofertas"},
        )
        fig10.update_layout(**PL, coloraxis_showscale=False, height=360)
        fig10.update_traces(marker_line_width=0,
                            hovertemplate="<b>%{y}</b><br>%{x:,}<extra></extra>")
        st.plotly_chart(fig10, use_container_width=True)

        jt_cmp = (
            jobs_f[jobs_f["company"] == cmp_sel]
            .groupby("job_type").size().reset_index(name="n")
        )
        fig11 = px.pie(
            jt_cmp, names="job_type", values="n",
            color_discrete_sequence=QUAL, hole=0.55, title="Modalidad",
        )
        fig11.update_layout(
            **PL, height=220,
            legend=dict(orientation="h", y=-0.05, font=dict(size=10)),
        )
        fig11.update_traces(textinfo="percent+label")
        st.plotly_chart(fig11, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Heatmap: empresas x nivel (top 15)")
    top15 = jobs_f.groupby("company")["job_link"].nunique().nlargest(15).index
    heat  = (
        jobs_f[jobs_f["company"].isin(top15)]
        .groupby(["company", "job_level"]).size()
        .reset_index(name="n")
        .pivot(index="company", columns="job_level", values="n")
        .fillna(0)
    )
    heat.columns.name = None
    fig12 = px.imshow(
        heat,
        color_continuous_scale=CMAP,
        aspect="auto",
        text_auto=True,
        labels=dict(x="Nivel", y="Empresa", color="Ofertas"),
    )
    fig12.update_layout(**PL, height=420, coloraxis_showscale=False)
    fig12.update_xaxes(tickangle=0)
    st.plotly_chart(fig12, use_container_width=True)

# ════════════════════════════════════════
# TAB 4 — GEOGRAFIA
# ════════════════════════════════════════
with tab4:
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("#### Ofertas por ciudad (top 20)")
        city_c = (
            jobs_f.groupby("city_clean")["job_link"].nunique()
            .nlargest(20).reset_index(name="ofertas").sort_values("ofertas")
        )
        fig13 = px.bar(
            city_c, x="ofertas", y="city_clean", orientation="h",
            color="ofertas", color_continuous_scale=CMAP3,
            labels={"city_clean": "Ciudad", "ofertas": "Ofertas"},
        )
        fig13.update_layout(**PL, coloraxis_showscale=False, height=520)
        fig13.update_traces(marker_line_width=0,
                            hovertemplate="<b>%{y}</b><br>%{x:,}<extra></extra>")
        st.plotly_chart(fig13, use_container_width=True)

    with col2:
        st.markdown("#### Skills por ciudad")
        city_opts = city_c["city_clean"].tolist()
        city_sel  = st.selectbox("Ciudad", city_opts, key="city1")
        lk_city   = set(jobs_f[jobs_f["city_clean"] == city_sel]["job_link"])
        sk_city   = (
            skills_f[skills_f["job_link"].isin(lk_city)]
            .groupby("skill")["job_link"].nunique()
            .nlargest(10).reset_index(name="n").sort_values("n")
        )
        fig14 = px.bar(
            sk_city, x="n", y="skill", orientation="h",
            color_discrete_sequence=["#8b5cf6"],
            labels={"skill": "", "n": "Ofertas"},
        )
        fig14.update_layout(**PL, height=340)
        fig14.update_traces(marker_line_width=0,
                            hovertemplate="<b>%{y}</b><br>%{x:,}<extra></extra>")
        st.plotly_chart(fig14, use_container_width=True)

        lvl_city = (
            jobs_f[jobs_f["city_clean"] == city_sel]
            .groupby("job_level").size().reset_index(name="n")
        )
        fig15 = px.pie(
            lvl_city, names="job_level", values="n",
            color_discrete_sequence=QUAL, hole=0.55,
            title="Nivel de experiencia",
        )
        fig15.update_layout(
            **PL, height=230,
            legend=dict(orientation="h", y=-0.05, font=dict(size=10)),
        )
        fig15.update_traces(textinfo="percent+label")
        st.plotly_chart(fig15, use_container_width=True)

    st.markdown("---")
    st.markdown("#### % de skill por ciudad")
    cities_comp = st.multiselect("Ciudades", city_opts, default=city_opts[:6])
    skill_comp  = st.selectbox(
        "Skill",
        skills_f["skill"].value_counts().nlargest(25).index.tolist(),
        key="sk_comp",
    )
    if cities_comp:
        rows = []
        for c in cities_comp:
            lk    = set(jobs_f[jobs_f["city_clean"] == c]["job_link"])
            total = len(lk)
            w_sk  = skills_f[
                (skills_f["job_link"].isin(lk)) &
                (skills_f["skill"] == skill_comp)
            ]["job_link"].nunique()
            rows.append({
                "Ciudad": c,
                "% con skill": round(w_sk / total * 100, 1) if total else 0,
            })
        comp_df = pd.DataFrame(rows).sort_values("% con skill", ascending=False)
        fig16   = px.bar(
            comp_df, x="Ciudad", y="% con skill",
            color="% con skill", text_auto=True,
            color_continuous_scale=CMAP2,
        )
        fig16.update_layout(**PL, coloraxis_showscale=False, height=320)
        fig16.update_traces(marker_line_width=0,
                            hovertemplate="<b>%{x}</b><br>%{y}%<extra></extra>")
        st.plotly_chart(fig16, use_container_width=True)

# ════════════════════════════════════════
# TAB 5 — DETALLES
# ════════════════════════════════════════
with tab5:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### Por nivel de experiencia")
        lvl_d = (
            jobs_f.groupby("job_level").size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )
        fig17 = px.bar(
            lvl_d, x="job_level", y="n",
            color="n", color_continuous_scale=CMAP,
            text_auto=True,
            labels={"job_level": "Nivel", "n": "Ofertas"},
        )
        fig17.update_layout(**PL, coloraxis_showscale=False, height=300,
                            xaxis_title="", yaxis_title="Ofertas")
        fig17.update_traces(marker_line_width=0)
        st.plotly_chart(fig17, use_container_width=True)

    with col2:
        st.markdown("#### Por tipo de empleo")
        type_d = jobs_f.groupby("job_type").size().reset_index(name="n")
        fig18  = px.pie(
            type_d, names="job_type", values="n",
            color_discrete_sequence=QUAL, hole=0.55,
            labels={"job_type": "Tipo"},
        )
        fig18.update_layout(
            **PL, height=300,
            legend=dict(orientation="h", y=-0.05),
        )
        fig18.update_traces(textinfo="percent+label")
        st.plotly_chart(fig18, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Roles con mayor densidad de skills tecnicas")
    tech_role = (
        skills_f[skills_f["category"] == "Tecnica"]
        .merge(jobs_f[["job_link", "job_title"]], on="job_link")
        .groupby("job_title")["skill"].nunique()
        .reset_index(name="Skills unicas")
        .sort_values("Skills unicas", ascending=False)
        .head(14)
    )
    fig19 = px.bar(
        tech_role.sort_values("Skills unicas"),
        x="Skills unicas", y="job_title", orientation="h",
        color="Skills unicas", color_continuous_scale=CMAP4,
        text_auto=True,
        labels={"job_title": "Rol", "Skills unicas": "Skills unicas"},
    )
    fig19.update_layout(**PL, coloraxis_showscale=False, height=420)
    fig19.update_traces(marker_line_width=0,
                        hovertemplate="<b>%{y}</b><br>%{x} skills<extra></extra>")
    st.plotly_chart(fig19, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Datos filtrados")
    show_df = jobs_f[[
        "job_title", "company", "city_clean",
        "job_level", "job_type", "first_seen"
    ]].copy()
    show_df["first_seen"] = show_df["first_seen"].dt.strftime("%d %b %Y")
    show_df.columns = ["Titulo", "Empresa", "Ciudad", "Nivel", "Tipo", "Fecha"]
    st.dataframe(
        show_df.head(300).reset_index(drop=True),
        use_container_width=True,
        height=340,
        column_config={
            "Titulo":  st.column_config.TextColumn("Titulo",  width="medium"),
            "Empresa": st.column_config.TextColumn("Empresa", width="medium"),
            "Ciudad":  st.column_config.TextColumn("Ciudad",  width="small"),
            "Nivel":   st.column_config.TextColumn("Nivel",   width="medium"),
            "Tipo":    st.column_config.TextColumn("Tipo",    width="small"),
            "Fecha":   st.column_config.TextColumn("Fecha",   width="small"),
        },
    )

    csv_bytes = jobs_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar CSV filtrado",
        csv_bytes,
        "skillscope_filtered.csv",
        "text/csv",
    )

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#334155;font-size:.72rem;font-family:monospace'>"
    "SkillScope · Julian · Ingenieria de Datos · "
    "<a href='https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024'"
    " style='color:#1d4ed8'>Dataset Kaggle</a></p>",
    unsafe_allow_html=True,
)