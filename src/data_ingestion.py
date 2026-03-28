import os
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from pipeline_config import API_BASE_URL, DATASETS, DEFAULT_PAGE_SIZE, RAW_DATA_DIR


def _request_batch(
    dataset_id: str,
    start_date: str,
    limit: int,
    offset: int,
    app_token: Optional[str],
) -> list[dict]:
    params = {
        "$limit": limit,
        "$offset": offset,
        "$where": f"crash_date >= '{start_date}T00:00:00'",
        "$order": "crash_date ASC",
    }
    headers = {}
    if app_token:
        headers["X-App-Token"] = app_token

    url = f"{API_BASE_URL}/{dataset_id}.json"
    response = requests.get(url, params=params, headers=headers, timeout=90)
    response.raise_for_status()
    return response.json()


def ingest_dataset(
    dataset_name: str,
    dataset_id: str,
    start_date: str,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_rows: Optional[int] = None,
) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = RAW_DATA_DIR / f"chicago_{dataset_name}_raw_{timestamp}.csv"
    app_token = os.getenv("SOCRATA_APP_TOKEN")

    offset = 0
    total_rows = 0
    first_write = True
    fieldnames: list[str] | None = None

    while True:
        batch = _request_batch(
            dataset_id=dataset_id,
            start_date=start_date,
            limit=page_size,
            offset=offset,
            app_token=app_token,
        )

        if not batch:
            break

        if max_rows is not None and total_rows + len(batch) > max_rows:
            batch = batch[: max_rows - total_rows]

        if not batch:
            break

        if fieldnames is None:
            fieldnames = sorted({key for row in batch for key in row.keys()})
        mode = "w" if first_write else "a"
        with output_path.open(mode, encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(fieldnames=fieldnames, f=f, extrasaction="ignore")
            if first_write:
                writer.writeheader()
            writer.writerows(batch)

        batch_size = len(batch)
        total_rows += batch_size
        offset += batch_size
        first_write = False

        print(
            f"[ingest:{dataset_name}] fetched={batch_size} total={total_rows} "
            f"offset={offset}"
        )

        if batch_size < page_size:
            break
        if max_rows is not None and total_rows >= max_rows:
            break

    if total_rows == 0:
        raise RuntimeError(f"No rows fetched for dataset '{dataset_name}'")

    print(f"[ingest:{dataset_name}] saved {total_rows} rows -> {output_path}")
    return output_path


def ingest_all_datasets(
    start_date: str,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_rows_per_dataset: Optional[int] = None,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for dataset_name, dataset_id in DATASETS.items():
        outputs[dataset_name] = ingest_dataset(
            dataset_name=dataset_name,
            dataset_id=dataset_id,
            start_date=start_date,
            page_size=page_size,
            max_rows=max_rows_per_dataset,
        )
    return outputs
