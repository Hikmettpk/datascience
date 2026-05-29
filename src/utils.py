"""
Shared utilities: logging setup, paths, ICD-9 mapping, medication list.
"""

import os
import logging
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
PROCESSED_DIR = os.path.join(ROOT, "data_processed")
LOGS_DIR = os.path.join(ROOT, "logs")
FIGURES_DIR = os.path.join(ROOT, "outputs", "figures")
RESULTS_DIR = os.path.join(ROOT, "outputs", "results")

RAW_DATA_PATH = os.path.join(DATA_DIR, "diabetic_data.csv")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DIR, "diabetic_preprocessed.csv")

for d in [PROCESSED_DIR, LOGS_DIR, FIGURES_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to both console and a timestamped log file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(message)s")

        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


# ── Feature groups ─────────────────────────────────────────────────────────────
MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]

ID_COLS = ["encounter_id", "patient_nbr"]

# Discharge disposition IDs that indicate the patient cannot be readmitted
# (expired or transferred to hospice / long-term care)
EXCLUDE_DISCHARGE_IDS = {
    11,   # Expired
    13,   # Hospice / medical facility
    14,   # Hospice / home
    19,   # Expired at home (Medicaid)
    20,   # Expired in a medical facility (Medicaid)
    21,   # Expired (place unknown) (Medicaid)
}

AGE_ORDER = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]
AGE_MAP = {age: i for i, age in enumerate(AGE_ORDER)}


def icd9_to_category(code: str) -> str:
    """
    Map an ICD-9 code string to a broad disease category.
    Handles V-codes, E-codes, and numeric codes.
    Returns 'Unknown' for missing or unrecognised values.
    """
    if not isinstance(code, str) or code in ("?", "nan", ""):
        return "Unknown"

    code = code.strip()

    if code.startswith("V"):
        return "Supplementary"
    if code.startswith("E"):
        return "External_Causes"

    try:
        num = float(code)
    except ValueError:
        return "Unknown"

    if 250 <= num < 251:
        return "Diabetes"
    if 1 <= num <= 139:
        return "Infectious"
    if 140 <= num <= 239:
        return "Neoplasms"
    if 240 <= num <= 279:
        return "Endocrine_Metabolic"
    if 280 <= num <= 289:
        return "Blood"
    if 290 <= num <= 319:
        return "Mental"
    if 320 <= num <= 389:
        return "Nervous_System"
    if 390 <= num <= 459:
        return "Circulatory"
    if 460 <= num <= 519:
        return "Respiratory"
    if 520 <= num <= 579:
        return "Digestive"
    if 580 <= num <= 629:
        return "Genitourinary"
    if 630 <= num <= 679:
        return "Pregnancy"
    if 680 <= num <= 709:
        return "Skin"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if 740 <= num <= 759:
        return "Congenital"
    if 760 <= num <= 779:
        return "Perinatal"
    if 780 <= num <= 799:
        return "Symptoms"
    if 800 <= num <= 999:
        return "Injury_Poisoning"

    return "Unknown"
