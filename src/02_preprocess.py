"""
Step 2 — Preprocessing Pipeline
Run AFTER 01_explore.py and reviewing the log.

Reads  : data/diabetic_data.csv
Writes : data_processed/diabetic_preprocessed.csv
         data_processed/diabetic_preprocessed_for_rules.csv  (binarised, for Apriori)
         logs/02_preprocess_<timestamp>.log

Usage:
    python src/02_preprocess.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    RAW_DATA_PATH, PROCESSED_DIR, FIGURES_DIR, RESULTS_DIR,
    get_logger, MEDICATION_COLS, NUMERIC_COLS,
    EXCLUDE_DISCHARGE_IDS, AGE_MAP, icd9_to_category,
)

log = get_logger("02_preprocess")
SEP = "=" * 80
SEP2 = "-" * 60


# ── Admission / discharge / source label maps ──────────────────────────────────
ADMISSION_TYPE_MAP = {
    1: "Emergency", 2: "Urgent", 3: "Elective",
    4: "Newborn", 5: "Not_Available", 6: "NULL",
    7: "Trauma_Center", 8: "Not_Mapped",
}
DISCHARGE_MAP = {
    1: "Home", 2: "Short_Term_Hospital", 3: "Skilled_Nursing",
    4: "Intermediate_Care", 5: "Other_Inpatient", 6: "Home_Health_Org",
    7: "AMA", 8: "Home_IV", 9: "Admitted_Inpatient", 10: "Neonate",
    11: "Expired", 12: "Still_Patient", 13: "Hospice_Facility",
    14: "Hospice_Home", 15: "Swing_Bed", 16: "Rehab_Facility",
    17: "Long_Term_Hospital", 18: "Not_Applicable",
    19: "Expired_Medicaid_Home", 20: "Expired_Medicaid_Facility",
    21: "Expired_Unknown", 22: "Short_Term_Rehab",
    23: "Long_Term_Rehab", 24: "Nursing_Facility",
    25: "Psychiatric_Hospital", 27: "Federal_Health_Facility",
    28: "Veteran_Hospital",
}
ADMISSION_SOURCE_MAP = {
    1: "Physician_Referral", 2: "Clinic_Referral", 3: "HMO_Referral",
    4: "Transfer_Hospital", 5: "Transfer_SNF", 6: "Transfer_Other",
    7: "Emergency_Room", 8: "Court_Law", 9: "Not_Available",
    10: "Transfer_Critical_Access", 11: "Normal_Delivery",
    12: "Premature_Delivery", 13: "Sick_Baby", 14: "Extramural_Birth",
    15: "Not_Available_15", 17: "NULL", 18: "Transfer_Acute",
    19: "Not_Mapped", 20: "Unknown", 21: "Readmission",
    22: "Physician_Ext", 23: "Transfer_Outpatient", 24: "Transfer_SNF_24",
    25: "Transfer_Rehab",
}


def load_and_clean_markers(path: str) -> pd.DataFrame:
    log.info(SEP)
    log.info("DIABETES 130-US HOSPITALS — PREPROCESSING PIPELINE")
    log.info(SEP)
    log.info(f"\n[STEP 1] Load raw data and replace '?' with NaN")

    df = pd.read_csv(path, low_memory=False)
    n_before = len(df)
    log.info(f"  Loaded: {n_before:,} rows  x  {df.shape[1]} columns")

    df.replace("?", np.nan, inplace=True)
    q_count = (df.applymap(lambda x: x == "?") if hasattr(df, "applymap")
               else df.apply(lambda c: (c == "?").sum())).sum().sum()
    log.info(f"  '?' markers converted to NaN (already done via replace)")
    return df


def remove_expired(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"\n[STEP 2] Remove expired / hospice patients")
    mask = df["discharge_disposition_id"].isin(EXCLUDE_DISCHARGE_IDS)
    n_remove = mask.sum()
    df = df[~mask].copy()
    log.info(f"  Removed: {n_remove:,} rows  (discharge IDs: {EXCLUDE_DISCHARGE_IDS})")
    log.info(f"  Remaining: {len(df):,}")
    return df


def remove_invalid_gender(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"\n[STEP 3] Remove rows with Unknown/Invalid gender")
    mask = df["gender"] == "Unknown/Invalid"
    n = mask.sum()
    df = df[~mask].copy()
    log.info(f"  Removed: {n:,} rows")
    log.info(f"  Remaining: {len(df):,}")
    return df


def keep_first_encounter(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"\n[STEP 4] Keep only first encounter per patient (by encounter_id order)")
    n_before = len(df)
    df = df.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first")
    n_after = len(df)
    log.info(f"  Removed: {n_before - n_after:,} duplicate-patient rows")
    log.info(f"  Remaining: {n_after:,} unique patients")
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"\n[STEP 5] Drop low-value / high-missing columns")

    # weight: ~97% missing; payer_code: ~40% missing
    # medical_specialty: ~49% missing; max_glu_serum: ~95%; A1Cresult: ~83%
    # identifiers: encounter_id, patient_nbr
    drop_cols = [
        "weight", "payer_code", "encounter_id", "patient_nbr",
        "medical_specialty", "max_glu_serum", "A1Cresult",
    ]

    # Drop near-constant medication columns (>98% 'No')
    near_const = []
    for col in MEDICATION_COLS:
        if col in df.columns:
            pct_no = (df[col] == "No").mean()
            if pct_no >= 0.98:
                near_const.append(col)
    drop_cols += near_const

    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    log.info(f"  Dropped {len(drop_cols)} columns: {drop_cols}")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"\n[STEP 6] Encode target variable: early_readmit")
    df["early_readmit"] = (df["readmitted"] == "<30").astype(int)
    vc = df["early_readmit"].value_counts()
    pos = vc.get(1, 0)
    neg = vc.get(0, 0)
    log.info(f"  Positive (readmitted <30 days): {pos:,}  ({pos/len(df)*100:.2f}%)")
    log.info(f"  Negative                      : {neg:,}  ({neg/len(df)*100:.2f}%)")
    log.info(f"  Imbalance ratio 1:{neg/pos:.1f}")
    df = df.drop(columns=["readmitted"])
    return df


def encode_age(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"\n[STEP 7] Encode age as ordinal integer")
    missing_age = df["age"].isna().sum()
    if missing_age > 0:
        log.info(f"  Warning: {missing_age} missing age values → 'Unknown' bin (encoded as -1)")
        df["age"] = df["age"].fillna("Unknown")
        df["age_num"] = df["age"].map(AGE_MAP).fillna(-1).astype(int)
    else:
        df["age_num"] = df["age"].map(AGE_MAP)
    df = df.drop(columns=["age"])
    log.info(f"  Age ordinal range: {df['age_num'].min()} – {df['age_num'].max()}")
    return df


def map_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"\n[STEP 8] Map integer ID columns to readable labels")

    df["admission_type"] = df["admission_type_id"].map(ADMISSION_TYPE_MAP).fillna("Unknown")
    df["discharge_disposition"] = df["discharge_disposition_id"].map(DISCHARGE_MAP).fillna("Unknown")
    df["admission_source"] = df["admission_source_id"].map(ADMISSION_SOURCE_MAP).fillna("Unknown")

    df = df.drop(columns=["admission_type_id", "discharge_disposition_id", "admission_source_id"])
    log.info(f"  Created: admission_type, discharge_disposition, admission_source")
    return df


def encode_diagnoses(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"\n[STEP 9] Map ICD-9 diagnosis codes to disease categories")
    for dcol in ["diag_1", "diag_2", "diag_3"]:
        if dcol not in df.columns:
            continue
        cat_col = dcol + "_cat"
        df[cat_col] = df[dcol].fillna("Unknown").apply(icd9_to_category)
        df = df.drop(columns=[dcol])
        dist = df[cat_col].value_counts()
        log.info(f"  {dcol} → {cat_col}: {dist.to_dict()}")
    return df


def encode_medications(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"\n[STEP 10] Encode medication columns: 0=Not prescribed, 1=Prescribed (any dose)")
    active_cols = [c for c in MEDICATION_COLS if c in df.columns]
    for col in active_cols:
        df[col] = df[col].apply(lambda x: 0 if (x == "No" or pd.isna(x)) else 1)

    total_meds = df[active_cols].sum(axis=1)
    log.info(f"  Encoded {len(active_cols)} medication columns")
    log.info(f"  New feature 'n_active_meds' = sum of all active medications")
    df["n_active_meds"] = total_meds
    log.info(f"  n_active_meds  mean={total_meds.mean():.2f}  max={total_meds.max()}")
    return df


def encode_glu_a1c(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"\n[STEP 11] max_glu_serum & A1Cresult — already dropped (>83% missing), skipping")
    return df


def encode_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"\n[STEP 12] Encode binary categorical columns")
    df["change"] = (df["change"] == "Ch").astype(int)
    df["diabetesMed"] = (df["diabetesMed"] == "Yes").astype(int)
    df["gender"] = (df["gender"] == "Male").astype(int)  # 1=Male, 0=Female
    log.info(f"  change (Ch→1, No→0), diabetesMed (Yes→1), gender (Male→1)")
    return df


def one_hot_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"\n[STEP 13] One-hot encode remaining categorical columns")

    df["race"] = df["race"].fillna("Unknown")

    cat_cols = [
        "race",
        "admission_type", "discharge_disposition", "admission_source",
        "diag_1_cat", "diag_2_cat", "diag_3_cat",
    ]
    cat_cols = [c for c in cat_cols if c in df.columns]

    n_before = df.shape[1]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)
    n_after = df.shape[1]
    log.info(f"  Columns: {n_before} → {n_after}  (+{n_after - n_before} one-hot columns)")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"\n[STEP 14] Feature engineering")

    df["total_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    df["procedures_per_day"] = df["num_procedures"] / df["time_in_hospital"].clip(lower=1)
    df["meds_per_day"] = df["num_medications"] / df["time_in_hospital"].clip(lower=1)
    df["lab_per_day"] = df["num_lab_procedures"] / df["time_in_hospital"].clip(lower=1)

    log.info(f"  Created: total_visits, procedures_per_day, meds_per_day, lab_per_day")
    return df


def final_check(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"\n[STEP 15] Final quality check")
    n_null = df.isnull().sum().sum()
    log.info(f"  Total null values remaining: {n_null}")
    if n_null > 0:
        cols_with_null = df.columns[df.isnull().any()].tolist()
        log.info(f"  Columns with nulls: {cols_with_null}")
        df = df.fillna(0)
        log.info(f"  Filled remaining nulls with 0")

    log.info(f"  Final shape: {df.shape[0]:,} rows  x  {df.shape[1]} columns")
    log.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    dtypes = df.dtypes.value_counts()
    log.info(f"  Dtype breakdown: {dtypes.to_dict()}")
    return df


def save_outputs(df: pd.DataFrame) -> None:
    out_path = os.path.join(PROCESSED_DIR, "diabetic_preprocessed.csv")
    df.to_csv(out_path, index=False)
    log.info(f"\n  [Saved] {out_path}")

    # Column reference file
    col_ref = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.values})
    col_ref.to_csv(os.path.join(RESULTS_DIR, "preprocessed_columns.csv"), index=False)
    log.info(f"  [Saved] outputs/results/preprocessed_columns.csv")


def make_apriori_dataset(df: pd.DataFrame) -> None:
    """
    Create a binarised version for Apriori.
    Numeric features are binned into low/high (above/below median).
    Categorical one-hot columns are kept as-is.
    """
    log.info(f"\n[BONUS] Creating binarised dataset for Apriori association rules")

    bin_df = pd.DataFrame()

    # Binary / already 0-1 columns (pass through)
    binary_cols = (
        [c for c in df.columns if df[c].nunique() == 2 and c != "early_readmit"]
    )
    bin_df = df[binary_cols + ["early_readmit"]].copy()

    # Bin numeric features at median
    for col in NUMERIC_COLS + ["age_num", "n_active_meds", "total_visits"]:
        if col not in df.columns:
            continue
        med = df[col].median()
        bin_df[col + "_high"] = (df[col] > med).astype(int)

    # max_glu_serum / A1Cresult dropped upstream — nothing to binarise here

    out_path = os.path.join(PROCESSED_DIR, "diabetic_preprocessed_for_rules.csv")
    bin_df.to_csv(out_path, index=False)
    log.info(f"  Shape: {bin_df.shape}  →  {out_path}")


def plot_preprocessed(df: pd.DataFrame) -> None:
    log.info(f"\n[PLOTS] Preprocessed dataset visualisations")

    # Class balance after preprocessing
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    vc = df["early_readmit"].value_counts()
    axes[0].bar(["Not <30", "<30"], [vc.get(0, 0), vc.get(1, 0)],
                color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Target: Early Readmit")
    axes[0].set_ylabel("Count")

    df["age_num"].value_counts().sort_index().plot(
        kind="bar", ax=axes[1], color="#55A868", edgecolor="white")
    axes[1].set_title("Age Distribution (Ordinal)")
    axes[1].set_xlabel("Age Group Index")

    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
    df[numeric_present].corr()["time_in_hospital"].drop("time_in_hospital").sort_values().plot(
        kind="barh", ax=axes[2], color="#C44E52")
    axes[2].set_title("Correlation with time_in_hospital")
    axes[2].axvline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "02_preprocessed_overview.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  [Saved] {path}")


def main():
    df = load_and_clean_markers(RAW_DATA_PATH)
    df = remove_expired(df)
    df = remove_invalid_gender(df)
    df = keep_first_encounter(df)
    df = drop_columns(df)
    df = encode_target(df)
    df = encode_age(df)
    df = map_id_columns(df)
    df = encode_diagnoses(df)
    df = encode_medications(df)
    df = encode_glu_a1c(df)
    df = encode_binary_flags(df)
    df = one_hot_categoricals(df)
    df = feature_engineering(df)
    df = final_check(df)

    log.info(f"\n{SEP2}")
    log.info("SAVING OUTPUTS")
    log.info(SEP2)
    save_outputs(df)
    make_apriori_dataset(df)
    plot_preprocessed(df)

    log.info(f"\n{SEP}")
    log.info("PREPROCESSING COMPLETE")
    log.info(f"Next steps: run 03_clustering.py, 04_association_rules.py, 05_classification.py")
    log.info(SEP)


if __name__ == "__main__":
    main()
