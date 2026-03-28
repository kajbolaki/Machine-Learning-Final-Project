from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
SQL_DIR = PROJECT_ROOT / "sql"

DATASETS = {
    "crashes": "85ca-t3if",
    "vehicles": "68nd-jvt3",
    "people": "u6pd-qa9d",
}

API_BASE_URL = "https://data.cityofchicago.org/resource"
DEFAULT_START_DATE = "2021-01-01"
DEFAULT_PAGE_SIZE = 50000
DEFAULT_RANDOM_STATE = 42
