from pathlib import Path 

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATASET_NAME = "german_credit_data.csv"
DATASET_PATH = RAW_DATA_DIR / DATASET_NAME
TARGET_COLUMN = "Risk"  

