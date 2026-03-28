from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.types import Text

from pipeline_config import DATASETS, PROCESSED_DATA_DIR, RAW_DATA_DIR, SQL_DIR


RAW_TABLES = {
    "crashes": "crashes_raw",
    "vehicles": "vehicles_raw",
    "people": "people_raw",
}


def _latest_snapshot(dataset_name: str) -> Path:
    pattern = f"chicago_{dataset_name}_raw_*.csv"
    candidates = sorted(RAW_DATA_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]

    sample_path = RAW_DATA_DIR / f"chicago_{dataset_name}_sample.csv"
    if sample_path.exists():
        return sample_path

    raise FileNotFoundError(
        f"No snapshot found for '{dataset_name}'. Run ingest mode first."
    )


def _load_csv_in_chunks(
    csv_path: Path, engine: Engine, table_name: str, chunk_size: int = 30000
) -> int:
    total_rows = 0
    first_chunk = True

    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        low_memory=False,
        dtype=str,
        keep_default_na=False,
    ):
        chunk.columns = [col.strip().lower() for col in chunk.columns]
        # Raw layer should preserve source values as text to avoid mixed-type failures.
        chunk = chunk.replace({"": None, "nan": None, "NaN": None, "NULL": None})
        dtype_map = {col: Text() for col in chunk.columns} if first_chunk else None
        if_exists = "replace" if first_chunk else "append"
        chunk.to_sql(
            table_name,
            engine,
            if_exists=if_exists,
            index=False,
            method=None,
            chunksize=2000,
            dtype=dtype_map,
        )
        first_chunk = False
        total_rows += len(chunk)
        print(f"[prepare] inserted chunk {len(chunk)} rows into '{table_name}' (total={total_rows})")

    return total_rows


def _write_sql_artifact() -> Path:
    SQL_DIR.mkdir(parents=True, exist_ok=True)
    sql_path = SQL_DIR / "crash_feature_queries.sql"
    sql_content = """-- SELECT example: raw volume by dataset
SELECT COUNT(*) AS crashes_rows FROM crashes_raw;
SELECT COUNT(*) AS vehicles_rows FROM vehicles_raw;
SELECT COUNT(*) AS people_rows FROM people_raw;

-- INSERT + JOIN example: build crash-level modeling table
DROP TABLE IF EXISTS crash_features;
CREATE TABLE crash_features (
    crash_record_id TEXT PRIMARY KEY,
    crash_date TIMESTAMP,
    posted_speed_limit DOUBLE PRECISION,
    traffic_control_device TEXT,
    intersection_related_i TEXT,
    hit_and_run_i TEXT,
    work_zone_i TEXT,
    weather_condition TEXT,
    lighting_condition TEXT,
    first_crash_type TEXT,
    trafficway_type TEXT,
    alignment TEXT,
    roadway_surface_cond TEXT,
    road_defect TEXT,
    prim_contributory_cause TEXT,
    sec_contributory_cause TEXT,
    crash_hour DOUBLE PRECISION,
    crash_day_of_week DOUBLE PRECISION,
    crash_month DOUBLE PRECISION,
    num_units DOUBLE PRECISION,
    injuries_fatal DOUBLE PRECISION,
    injuries_incapacitating DOUBLE PRECISION,
    injuries_non_incapacitating DOUBLE PRECISION,
    injuries_total DOUBLE PRECISION,
    vehicle_count DOUBLE PRECISION,
    towed_vehicle_count DOUBLE PRECISION,
    people_count DOUBLE PRECISION,
    driver_count DOUBLE PRECISION
);

INSERT INTO crash_features (
    crash_record_id, crash_date, posted_speed_limit, traffic_control_device, intersection_related_i,
    hit_and_run_i, work_zone_i, weather_condition,
    lighting_condition, first_crash_type, trafficway_type, alignment, roadway_surface_cond,
    road_defect, prim_contributory_cause, sec_contributory_cause, crash_hour, crash_day_of_week,
    crash_month, num_units, injuries_fatal, injuries_incapacitating, injuries_non_incapacitating,
    injuries_total, vehicle_count, towed_vehicle_count, people_count, driver_count
)
SELECT
    c.crash_record_id,
    NULLIF(c.crash_date::TEXT, '')::TIMESTAMP AS crash_date,
    NULLIF(c.posted_speed_limit::TEXT, '')::DOUBLE PRECISION AS posted_speed_limit,
    c.traffic_control_device,
    c.intersection_related_i,
    c.hit_and_run_i,
    c.work_zone_i,
    c.weather_condition,
    c.lighting_condition,
    c.first_crash_type,
    c.trafficway_type,
    c.alignment,
    c.roadway_surface_cond,
    c.road_defect,
    c.prim_contributory_cause,
    c.sec_contributory_cause,
    NULLIF(c.crash_hour::TEXT, '')::DOUBLE PRECISION AS crash_hour,
    NULLIF(c.crash_day_of_week::TEXT, '')::DOUBLE PRECISION AS crash_day_of_week,
    NULLIF(c.crash_month::TEXT, '')::DOUBLE PRECISION AS crash_month,
    NULLIF(c.num_units::TEXT, '')::DOUBLE PRECISION AS num_units,
    NULLIF(c.injuries_fatal::TEXT, '')::DOUBLE PRECISION AS injuries_fatal,
    NULLIF(c.injuries_incapacitating::TEXT, '')::DOUBLE PRECISION AS injuries_incapacitating,
    NULLIF(c.injuries_non_incapacitating::TEXT, '')::DOUBLE PRECISION AS injuries_non_incapacitating,
    NULLIF(c.injuries_total::TEXT, '')::DOUBLE PRECISION AS injuries_total,
    COALESCE(v.vehicle_count, 0) AS vehicle_count,
    COALESCE(v.towed_vehicle_count, 0) AS towed_vehicle_count,
    COALESCE(p.people_count, 0) AS people_count,
    COALESCE(p.driver_count, 0) AS driver_count
FROM crashes_raw c
LEFT JOIN (
    SELECT
        crash_record_id,
        COUNT(*)::DOUBLE PRECISION AS vehicle_count,
        SUM(CASE WHEN towed_i = 'Y' THEN 1 ELSE 0 END)::DOUBLE PRECISION AS towed_vehicle_count
    FROM vehicles_raw
    GROUP BY crash_record_id
) v ON c.crash_record_id = v.crash_record_id
LEFT JOIN (
    SELECT
        crash_record_id,
        COUNT(*)::DOUBLE PRECISION AS people_count,
        SUM(CASE WHEN person_type = 'DRIVER' THEN 1 ELSE 0 END)::DOUBLE PRECISION AS driver_count
    FROM people_raw
    GROUP BY crash_record_id
) p ON c.crash_record_id = p.crash_record_id
WHERE NULLIF(c.crash_date::TEXT, '')::TIMESTAMP >= :start_date;
"""
    sql_path.write_text(sql_content, encoding="utf-8")
    return sql_path


def _execute_feature_sql(engine: Engine, start_date: str) -> None:
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS crash_features"))
        connection.execute(
            text(
                """
                CREATE TABLE crash_features (
                    crash_record_id TEXT PRIMARY KEY,
                    crash_date TIMESTAMP,
                    posted_speed_limit DOUBLE PRECISION,
                    traffic_control_device TEXT,
                    intersection_related_i TEXT,
                    hit_and_run_i TEXT,
                    work_zone_i TEXT,
                    weather_condition TEXT,
                    lighting_condition TEXT,
                    first_crash_type TEXT,
                    trafficway_type TEXT,
                    alignment TEXT,
                    roadway_surface_cond TEXT,
                    road_defect TEXT,
                    prim_contributory_cause TEXT,
                    sec_contributory_cause TEXT,
                    crash_hour DOUBLE PRECISION,
                    crash_day_of_week DOUBLE PRECISION,
                    crash_month DOUBLE PRECISION,
                    num_units DOUBLE PRECISION,
                    injuries_fatal DOUBLE PRECISION,
                    injuries_incapacitating DOUBLE PRECISION,
                    injuries_non_incapacitating DOUBLE PRECISION,
                    injuries_total DOUBLE PRECISION,
                    vehicle_count DOUBLE PRECISION,
                    towed_vehicle_count DOUBLE PRECISION,
                    people_count DOUBLE PRECISION,
                    driver_count DOUBLE PRECISION
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO crash_features (
                    crash_record_id, crash_date, posted_speed_limit, traffic_control_device, intersection_related_i,
                    hit_and_run_i, work_zone_i, weather_condition,
                    lighting_condition, first_crash_type, trafficway_type, alignment, roadway_surface_cond,
                    road_defect, prim_contributory_cause, sec_contributory_cause, crash_hour, crash_day_of_week,
                    crash_month, num_units, injuries_fatal, injuries_incapacitating, injuries_non_incapacitating,
                    injuries_total, vehicle_count, towed_vehicle_count, people_count, driver_count
                )
                SELECT
                    c.crash_record_id,
                    NULLIF(c.crash_date::TEXT, '')::TIMESTAMP AS crash_date,
                    NULLIF(c.posted_speed_limit::TEXT, '')::DOUBLE PRECISION AS posted_speed_limit,
                    c.traffic_control_device,
                    c.intersection_related_i,
                    c.hit_and_run_i,
                    c.work_zone_i,
                    c.weather_condition,
                    c.lighting_condition,
                    c.first_crash_type,
                    c.trafficway_type,
                    c.alignment,
                    c.roadway_surface_cond,
                    c.road_defect,
                    c.prim_contributory_cause,
                    c.sec_contributory_cause,
                    NULLIF(c.crash_hour::TEXT, '')::DOUBLE PRECISION AS crash_hour,
                    NULLIF(c.crash_day_of_week::TEXT, '')::DOUBLE PRECISION AS crash_day_of_week,
                    NULLIF(c.crash_month::TEXT, '')::DOUBLE PRECISION AS crash_month,
                    NULLIF(c.num_units::TEXT, '')::DOUBLE PRECISION AS num_units,
                    NULLIF(c.injuries_fatal::TEXT, '')::DOUBLE PRECISION AS injuries_fatal,
                    NULLIF(c.injuries_incapacitating::TEXT, '')::DOUBLE PRECISION AS injuries_incapacitating,
                    NULLIF(c.injuries_non_incapacitating::TEXT, '')::DOUBLE PRECISION AS injuries_non_incapacitating,
                    NULLIF(c.injuries_total::TEXT, '')::DOUBLE PRECISION AS injuries_total,
                    COALESCE(v.vehicle_count, 0) AS vehicle_count,
                    COALESCE(v.towed_vehicle_count, 0) AS towed_vehicle_count,
                    COALESCE(p.people_count, 0) AS people_count,
                    COALESCE(p.driver_count, 0) AS driver_count
                FROM crashes_raw c
                LEFT JOIN (
                    SELECT
                        crash_record_id,
                        COUNT(*)::DOUBLE PRECISION AS vehicle_count,
                        SUM(CASE WHEN towed_i = 'Y' THEN 1 ELSE 0 END)::DOUBLE PRECISION AS towed_vehicle_count
                    FROM vehicles_raw
                    GROUP BY crash_record_id
                ) v ON c.crash_record_id = v.crash_record_id
                LEFT JOIN (
                    SELECT
                        crash_record_id,
                        COUNT(*)::DOUBLE PRECISION AS people_count,
                        SUM(CASE WHEN person_type = 'DRIVER' THEN 1 ELSE 0 END)::DOUBLE PRECISION AS driver_count
                    FROM people_raw
                    GROUP BY crash_record_id
                ) p ON c.crash_record_id = p.crash_record_id
                WHERE NULLIF(c.crash_date::TEXT, '')::TIMESTAMP >= :start_date
                """
            ),
            {"start_date": start_date},
        )


def _save_sql_summary(engine: Engine) -> Path:
    summary_query = text(
        """
        SELECT
            COUNT(*) AS rows_total,
            AVG(CASE WHEN injuries_fatal > 0 OR injuries_incapacitating > 0 THEN 1 ELSE 0 END)
                AS severe_rate,
            AVG(posted_speed_limit) AS avg_speed_limit,
            AVG(vehicle_count) AS avg_vehicle_count
        FROM crash_features
        """
    )
    summary_df = pd.read_sql(summary_query, engine)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "sql_analysis_summary.csv"
    summary_df.to_csv(output_path, index=False)
    return output_path


def _to_quality_report(engine: Engine, start_date: str) -> Path:
    coverage_query = text(
        """
        SELECT
            (SELECT COUNT(*) FROM crash_features) AS feature_rows,
            (
                SELECT COUNT(*)
                FROM crashes_raw
                WHERE NULLIF(crash_date::TEXT, '')::TIMESTAMP >= :start_date
            ) AS crash_rows
        """
    )
    coverage_df = pd.read_sql(coverage_query, engine, params={"start_date": start_date})
    feature_rows = int(coverage_df.loc[0, "feature_rows"])
    crash_rows = int(coverage_df.loc[0, "crash_rows"])
    coverage = round(feature_rows / crash_rows, 4) if crash_rows else 0.0

    report = {
        "feature_rows": feature_rows,
        "crash_rows": crash_rows,
        "join_coverage": coverage,
        "coverage_target_met": coverage >= 0.9,
    }

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PROCESSED_DATA_DIR / "data_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def _export_curated_table(engine: Engine) -> Path:
    curated_df = pd.read_sql(text("SELECT * FROM crash_features"), engine)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "crash_features.csv"
    curated_df.to_csv(output_path, index=False)
    return output_path


def prepare_training_data(engine: Engine, start_date: str) -> dict[str, Path]:
    outputs: dict[str, Path] = {}

    for dataset_name in DATASETS.keys():
        csv_path = _latest_snapshot(dataset_name)
        table_name = RAW_TABLES[dataset_name]
        loaded_rows = _load_csv_in_chunks(csv_path=csv_path, engine=engine, table_name=table_name)
        print(f"[prepare] loaded {loaded_rows} rows into table '{table_name}' from {csv_path.name}")

    outputs["sql_artifact"] = _write_sql_artifact()
    _execute_feature_sql(engine=engine, start_date=start_date)
    outputs["curated_csv"] = _export_curated_table(engine=engine)
    outputs["sql_summary"] = _save_sql_summary(engine=engine)
    outputs["quality_report"] = _to_quality_report(engine=engine, start_date=start_date)
    return outputs
