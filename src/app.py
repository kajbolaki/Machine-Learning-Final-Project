import argparse
from typing import Optional

from pipeline_config import DEFAULT_PAGE_SIZE, DEFAULT_START_DATE


def _run_ingest(start_date: str, page_size: int, max_rows_per_dataset: Optional[int]) -> None:
    from data_ingestion import ingest_all_datasets

    outputs = ingest_all_datasets(
        start_date=start_date,
        page_size=page_size,
        max_rows_per_dataset=max_rows_per_dataset,
    )
    print("\n[ingest] completed")
    for dataset_name, output_path in outputs.items():
        print(f" - {dataset_name}: {output_path}")


def _run_prepare(start_date: str) -> None:
    from data_preparation import prepare_training_data
    from utils import db_connect

    engine = db_connect()
    outputs = prepare_training_data(engine=engine, start_date=start_date)
    print("\n[prepare] completed")
    for artifact_name, artifact_path in outputs.items():
        print(f" - {artifact_name}: {artifact_path}")


def _run_train() -> None:
    from model_training import train_model_pipeline
    from utils import db_connect

    engine = db_connect()
    outputs = train_model_pipeline(engine=engine)
    print("\n[train] completed")
    for artifact_name, artifact_path in outputs.items():
        print(f" - {artifact_name}: {artifact_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chicago Crash Severity Pipeline CLI"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["ingest", "prepare", "train"],
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Start date filter (YYYY-MM-DD) used for ingestion/preparation.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help="Socrata pagination size for ingest mode.",
    )
    parser.add_argument(
        "--max-rows-per-dataset",
        type=int,
        default=None,
        help="Optional limit for quick tests (ingest mode only).",
    )
    args = parser.parse_args()

    if args.mode == "ingest":
        _run_ingest(
            start_date=args.start_date,
            page_size=args.page_size,
            max_rows_per_dataset=args.max_rows_per_dataset,
        )
        return
    if args.mode == "prepare":
        _run_prepare(start_date=args.start_date)
        return
    if args.mode == "train":
        _run_train()
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
