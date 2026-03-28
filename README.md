# Chicago Crash Severity Prediction (Machine Learning Final Project)

Binary machine learning product that predicts if a crash is **severe** (`1`) or **non-severe** (`0`) using crash context data from Chicago Open Data.

## Business Problem
Road safety teams need to proactively identify high-risk crash conditions (weather, lighting, road surface, speed, crash type) so they can prioritize interventions, reduce severe injuries, and allocate resources better.

## Data Source (Free Public API)
- Crashes: `85ca-t3if`
- Vehicles: `68nd-jvt3`
- People: `u6pd-qa9d`
- Platform: Chicago Open Data (Socrata API)
- Time filter: `2021-01-01` to present

## Project Workflow
1. `ingest` mode: pull data from API with pagination and save immutable raw snapshots.
2. `prepare` mode: load raw CSVs into PostgreSQL and build curated `crash_features` table via SQL `SELECT`, `JOIN`, and `INSERT`.
3. `train` mode: train and tune Logistic Regression, Random Forest, and XGBoost with time-aware split; save best model and metrics.
4. `streamlit_app.py`: serve online predictions with risk band and feature contributions.

## Label Definition
`severe = 1` if `injuries_fatal > 0` or `injuries_incapacitating > 0`, else `0`.

## CLI Interface
Run from project root:

```bash
python src/app.py --mode ingest --start-date 2021-01-01
python src/app.py --mode prepare --start-date 2021-01-01
python src/app.py --mode train
```

Optional quick-test ingestion:

```bash
python src/app.py --mode ingest --start-date 2021-01-01 --max-rows-per-dataset 5000
```

## Streamlit App
After training:

```bash
streamlit run src/streamlit_app.py
```

## Environment Variables
Create `.env` in project root:

```bash
DATABASE_URL=postgresql://<USER>:<PASSWORD>@<HOST>:<PORT>/<DB_NAME>
SOCRATA_APP_TOKEN=<optional_but_recommended>
```

## Generated Artifacts
- Raw snapshots: `data/raw/chicago_<dataset>_raw_<timestamp>.csv`
- Curated features: `data/processed/crash_features.csv`
- SQL summary: `data/processed/sql_analysis_summary.csv`
- Quality report: `data/processed/data_quality_report.json`
- Trained model: `models/crash_severity_model.joblib`
- Metrics: `models/model_metrics.json`
- App metadata: `models/model_metadata.json`
- SQL script artifact: `sql/crash_feature_queries.sql`

## Deployment (Render)
`render.yaml` is included. Start command:

```bash
streamlit run src/streamlit_app.py --server.address 0.0.0.0 --server.port $PORT
```

## Presentation Support
Use `docs/presentation_outline.md` for the 5-minute final presentation structure.
