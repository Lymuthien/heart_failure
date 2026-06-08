from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[2]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

AGE = "Age"
SEX = "Sex"
CHEST_PAIN_TYPE = "ChestPainType"
RESTING_BP = "RestingBP"
CHOLESTEROL = "Cholesterol"
FASTING_BS = "FastingBS"
RESTING_ECG = "RestingECG"
MAX_HR = "MaxHR"
EXERCISE_ANGINA = "ExerciseAngina"
OLDPEAK = "Oldpeak"
ST_SLOPE = "ST_Slope"
HEART_DISEASE = "HeartDisease"

