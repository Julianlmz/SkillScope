# SkillScope 🔍

> Pipeline de ingeniería de datos para análisis de tendencias de habilidades tecnológicas en el mercado laboral.

---

## Descripción

SkillScope procesa más de **1.3 millones de ofertas de trabajo** de LinkedIn (2024) para identificar qué habilidades técnicas y blandas son más demandadas en el mercado laboral de Estados Unidos. El proyecto implementa un pipeline ETL completo con análisis exploratorio de datos, sentando las bases para modelos predictivos en fases posteriores.

---

## Estructura del proyecto

```
SkillScope/
├── data/
│   ├── raw/             # Datos originales de Kaggle (excluidos del repo)
│   ├── processed/       # Datos transformados (excluidos del repo)
│   └── external/        # Listas de skills de referencia
├── notebooks/
│   └── skillscope_documentation.ipynb   # Notebook de documentación (ETL + EDA)
├── src/
│   ├── ingestion/       # Scrapers (LinkedIn, Indeed)
│   ├── pipeline/        # Orquestación del pipeline
│   ├── transformation/  # Limpieza y extracción de skills
│   └── utils/           # Base de datos y helpers
├── tests/               # Pruebas unitarias
├── requirements.txt
└── README.md
```

---

## Dataset

**Fuente:** [1.3M LinkedIn Jobs & Skills (2024)](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) — Kaggle

| Archivo | Registros | Tamaño |
|---|---|---|
| `linkedin_job_postings.csv` | 1,348,454 | 415 MB |
| `job_skills.csv` | 1,296,381 | 672 MB |
| `job_summary.csv` | — | 5.1 GB |

**Alcance:** Filtrado a **Estados Unidos → 1,149,342 ofertas** válidas.

---

## Pipeline ETL

### Extract
- Carga de `linkedin_job_postings.csv` y `job_skills.csv` con pandas
- 1,348,454 registros totales de entrada

### Transform
- Filtrado por país: `job_location` contiene "United States"
- Conversión de fechas (`original_listed_time` → datetime)
- Creación de columna `week` para análisis temporal
- Limpieza de texto en títulos y empresas
- Explosión de skills: 1 fila por par (job, skill) → **23,180,623 relaciones**
- Eliminación de duplicados: 0 encontrados

### Load
- Base de datos **DuckDB** (`data/jobs.duckdb`)
- Tabla `jobs`: 1,149,342 registros — `job_link` como PRIMARY KEY
- Tabla `job_skills`: 23,180,623 registros — PRIMARY KEY compuesta `(job_link, skill)`

---

## Análisis Exploratorio (EDA)

### Hallazgos principales

**Top 5 skills generales:**
| Skill | Frecuencia |
|---|---|
| Communication | 308,000 |
| Teamwork | 191,000 |
| Leadership | 153,000 |
| Problem Solving | ~120,000 |
| Management | ~100,000 |

**Top 5 skills técnicas (normalizadas):**
| Skill | Frecuencia |
|---|---|
| Data Analysis | 70,000 |
| Excel | 35,000 |
| Python | 22,000 |
| SQL | 18,000 |
| Java | 12,000 |

**Distribución del mercado:**
- **Nivel de experiencia:** Mid-Senior 89% / Associate 11%
- **Modalidad:** Onsite 99% (el dataset tiene un sesgo marcado hacia trabajo presencial)
- Excel supera a Python en el mercado general — Python domina en roles especializados
- Herramientas modernas (dbt, Airflow, Databricks) tienen baja presencia en el dataset

---

## Tecnologías

| Categoría | Tecnología |
|---|---|
| Lenguaje | Python 3.12+ |
| Gestión de entorno | uv |
| Base de datos | DuckDB |
| Procesamiento | pandas |
| Visualización | matplotlib, seaborn |
| Notebooks | Jupyter |
| Descarga de datos | Kaggle API |

---

## Instalación

### Prerrequisitos
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) instalado
- Cuenta en Kaggle con API key configurada en `~/.kaggle/kaggle.json`

### Pasos

```powershell
# 1. Clonar el repositorio
git clone <url-del-repo>
cd SkillScope

# 2. Crear entorno virtual
uv venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Instalar dependencias
uv pip install -r requirements.txt

# 4. Descargar el dataset
kaggle datasets download -d asaniczka/1-3m-linkedin-jobs-and-skills-2024 -p data/raw --unzip
```

---

## Uso

Abrir y ejecutar el notebook de documentación:

```powershell
jupyter notebook notebooks/skillscope_documentation.ipynb
```

El notebook ejecuta el pipeline completo en orden: Extract → Transform → Load → EDA.

---

## Roadmap

- [x] **Corte 1** — ETL + EDA
  - [x] Pipeline de extracción y carga a DuckDB
  - [x] Análisis exploratorio con 8 visualizaciones
- [ ] **Corte 2** — Modelos
  - [ ] Clustering de skills por rol
  - [ ] Recomendación de habilidades por perfil
  - [ ] Predicción de demanda temporal
  - [ ] Dashboard con Streamlit
  - [ ] Servidor MCP para consultas en lenguaje natural

---

## Autores

**Julian** — Ingeniería de Datos, Séptimo Semestre