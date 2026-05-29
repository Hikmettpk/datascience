"""
Step 1 — Dataset Exploration
Run this first. Reads the raw CSV, produces a detailed log under logs/,
and saves summary tables to outputs/results/.

Usage:
    python src/01_explore.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    RAW_DATA_PATH, FIGURES_DIR, RESULTS_DIR,
    get_logger, MEDICATION_COLS, NUMERIC_COLS,
    ID_COLS, AGE_ORDER, icd9_to_category,
)

log = get_logger("01_explore")

SEP = "=" * 80
SEP2 = "-" * 60


def load_raw(path: str) -> pd.DataFrame:
    log.info(SEP)
    log.info("DIABETES 130-US HOSPITALS — DATASET EXPLORATION")
    log.info(f"Timestamp : {datetime.now()}")
    log.info(f"Data file : {path}")
    log.info(SEP)

    if not os.path.exists(path):
        log.error(f"\n[ERROR] File not found: {path}")
        log.error("Place diabetic_data.csv inside the /data folder and re-run.")
        sys.exit(1)

    df = pd.read_csv(path, low_memory=False)
    log.info(f"\n[LOAD] {df.shape[0]:,} rows  x  {df.shape[1]} columns")
    log.info(f"       Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    return df


def basic_info(df: pd.DataFrame) -> pd.DataFrame:
    """Replace '?' with NaN and return the cleaned-marker version for analysis."""
    df_nan = df.replace("?", np.nan)

    log.info(f"\n{SEP2}")
    log.info("COLUMN-BY-COLUMN OVERVIEW")
    log.info(SEP2)
    log.info(
        f"  {'Column':<45} {'dtype':<10} {'missing':>8} {'miss%':>7} {'unique':>8}  top_values"
    )
    log.info("  " + "-" * 110)

    summary_rows = []
    for col in df_nan.columns:
        n_miss = df_nan[col].isna().sum()
        pct_miss = n_miss / len(df_nan) * 100
        n_unique = df_nan[col].nunique(dropna=True)
        dtype = str(df_nan[col].dtype)

        if df_nan[col].dtype == object:
            top = df_nan[col].value_counts(dropna=True).head(4).to_dict()
            top_str = str(top)[:80]
        else:
            try:
                s = df_nan[col]
                top_str = (
                    f"min={s.min():.1f}  mean={s.mean():.1f}"
                    f"  max={s.max():.1f}  std={s.std():.1f}"
                )
            except Exception:
                top_str = str(df_nan[col].value_counts(dropna=True).head(3).to_dict())[:80]

        log.info(
            f"  {col:<45} {dtype:<10} {n_miss:>8,} {pct_miss:>6.1f}%  {n_unique:>7,}  {top_str}"
        )
        summary_rows.append(
            dict(column=col, dtype=dtype, n_missing=n_miss,
                 pct_missing=round(pct_miss, 2), n_unique=n_unique)
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(RESULTS_DIR, "column_summary.csv"), index=False)
    log.info(f"\n  [Saved] outputs/results/column_summary.csv")
    return df_nan


def target_analysis(df: pd.DataFrame) -> None:
    log.info(f"\n{SEP2}")
    log.info("TARGET VARIABLE — readmitted")
    log.info(SEP2)

    vc = df["readmitted"].value_counts(dropna=False)
    for val, cnt in vc.items():
        log.info(f"  {str(val):<12} : {cnt:>7,}  ({cnt/len(df)*100:5.2f}%)")

    early = (df["readmitted"] == "<30").sum()
    log.info(f"\n  Binary target (early readmission <30 days):")
    log.info(f"    Positive (<30)  : {early:>7,}  ({early/len(df)*100:.2f}%)")
    log.info(f"    Negative (other): {len(df)-early:>7,}  ({(len(df)-early)/len(df)*100:.2f}%)")
    log.info(f"    Class imbalance ratio: 1 : {(len(df)-early)/early:.1f}")


def duplicate_analysis(df: pd.DataFrame) -> None:
    log.info(f"\n{SEP2}")
    log.info("DUPLICATE / MULTI-ENCOUNTER ANALYSIS")
    log.info(SEP2)

    dup_enc = df["encounter_id"].duplicated().sum()
    log.info(f"  Duplicate encounter_id rows : {dup_enc:,}")

    n_patients = df["patient_nbr"].nunique()
    total_enc = len(df)
    multi = df.groupby("patient_nbr").size()
    multi_patients = (multi > 1).sum()

    log.info(f"  Total encounters            : {total_enc:,}")
    log.info(f"  Unique patients             : {n_patients:,}")
    log.info(f"  Patients with >1 encounter  : {multi_patients:,}  "
             f"({multi_patients/n_patients*100:.1f}% of patients)")

    enc_dist = multi.value_counts().sort_index().head(10)
    log.info(f"\n  Encounters per patient (top 10):")
    for n, cnt in enc_dist.items():
        log.info(f"    {n} encounter(s) : {cnt:,} patients")


def expired_analysis(df: pd.DataFrame) -> None:
    log.info(f"\n{SEP2}")
    log.info("EXPIRED / HOSPICE PATIENTS (discharge_disposition_id)")
    log.info(SEP2)

    dd = df["discharge_disposition_id"].value_counts().sort_index()
    log.info("  All discharge disposition ID counts:")
    for did, cnt in dd.items():
        log.info(f"    ID {int(did):>3d} : {cnt:>6,}  ({cnt/len(df)*100:.2f}%)")

    from utils import EXCLUDE_DISCHARGE_IDS
    exclude_mask = df["discharge_disposition_id"].isin(EXCLUDE_DISCHARGE_IDS)
    n_excl = exclude_mask.sum()
    log.info(
        f"\n  Rows to exclude (expired/hospice IDs {EXCLUDE_DISCHARGE_IDS}): "
        f"{n_excl:,}  ({n_excl/len(df)*100:.2f}%)"
    )


def numeric_analysis(df: pd.DataFrame) -> None:
    log.info(f"\n{SEP2}")
    log.info("NUMERIC FEATURES — DESCRIPTIVE STATISTICS")
    log.info(SEP2)

    desc = df[NUMERIC_COLS].describe().T
    desc["cv%"] = (desc["std"] / desc["mean"] * 100).round(1)
    log.info(desc.to_string())

    skews = df[NUMERIC_COLS].skew()
    log.info(f"\n  Skewness:")
    for col, sk in skews.items():
        log.info(f"    {col:<35} {sk:+.3f}")


def categorical_analysis(df: pd.DataFrame) -> None:
    log.info(f"\n{SEP2}")
    log.info("CATEGORICAL FEATURES")
    log.info(SEP2)

    cat_cols = [
        "race", "gender", "age", "payer_code", "medical_specialty",
        "max_glu_serum", "A1Cresult", "change", "diabetesMed",
        "admission_type_id", "discharge_disposition_id", "admission_source_id",
    ]
    for col in cat_cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False).head(10)
        log.info(f"\n  {col}  (n_unique={df[col].nunique()}):")
        for val, cnt in vc.items():
            log.info(f"    {str(val):<35} {cnt:>7,}  ({cnt/len(df)*100:.2f}%)")


def medication_analysis(df: pd.DataFrame) -> None:
    log.info(f"\n{SEP2}")
    log.info("MEDICATION COLUMNS (23 drugs)")
    log.info(SEP2)
    log.info(f"  {'Drug':<35} {'No':>8} {'Steady':>8} {'Up':>7} {'Down':>7} {'%Active':>9}")
    log.info("  " + "-" * 80)

    for col in MEDICATION_COLS:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=True)
        no = vc.get("No", 0)
        steady = vc.get("Steady", 0)
        up = vc.get("Up", 0)
        down = vc.get("Down", 0)
        active_pct = (steady + up + down) / len(df) * 100
        log.info(
            f"  {col:<35} {no:>8,} {steady:>8,} {up:>7,} {down:>7,} {active_pct:>8.1f}%"
        )


def diagnosis_analysis(df: pd.DataFrame) -> None:
    log.info(f"\n{SEP2}")
    log.info("DIAGNOSIS CODES (diag_1 / diag_2 / diag_3) — ICD-9 category mapping")
    log.info(SEP2)

    for dcol in ["diag_1", "diag_2", "diag_3"]:
        if dcol not in df.columns:
            continue
        cats = df[dcol].apply(icd9_to_category).value_counts()
        n_miss = df[dcol].isna().sum()
        log.info(f"\n  {dcol}  (missing: {n_miss:,}):")
        for cat, cnt in cats.items():
            log.info(f"    {cat:<25} {cnt:>7,}  ({cnt/len(df)*100:.2f}%)")


def age_analysis(df: pd.DataFrame) -> None:
    log.info(f"\n{SEP2}")
    log.info("AGE DISTRIBUTION")
    log.info(SEP2)
    vc = df["age"].value_counts().reindex(AGE_ORDER, fill_value=0)
    for age, cnt in vc.items():
        bar = "█" * (cnt // 1000)
        log.info(f"  {age:<12} {cnt:>7,}  {bar}")


def weight_analysis(df: pd.DataFrame) -> None:
    log.info(f"\n{SEP2}")
    log.info("WEIGHT COLUMN — detailed")
    log.info(SEP2)
    vc = df["weight"].value_counts(dropna=False).head(20)
    n_miss = df["weight"].isna().sum()
    log.info(f"  Missing: {n_miss:,}  ({n_miss/len(df)*100:.1f}%)")
    log.info(f"  Present: {len(df)-n_miss:,}  ({(len(df)-n_miss)/len(df)*100:.1f}%)")
    log.info(f"  Top values (when present):")
    for val, cnt in vc.items():
        log.info(f"    {str(val):<20} {cnt:>6,}")


def plot_target(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    vc = df["readmitted"].value_counts()
    axes[0].bar(vc.index, vc.values, color=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].set_title("Readmission Distribution")
    axes[0].set_xlabel("readmitted")
    axes[0].set_ylabel("Count")
    for i, (v, c) in enumerate(vc.items()):
        axes[0].text(i, c + 200, f"{c:,}\n({c/len(df)*100:.1f}%)", ha="center", fontsize=9)

    binary = df["readmitted"].apply(lambda x: "<30" if x == "<30" else "Not <30")
    vc2 = binary.value_counts()
    axes[1].pie(vc2.values, labels=vc2.index, autopct="%1.1f%%",
                colors=["#DD8452", "#4C72B0"], startangle=90)
    axes[1].set_title("Binary Early Readmission Target")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "01_target_distribution.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"\n  [Saved] {path}")


def plot_numeric(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, col in enumerate(NUMERIC_COLS):
        axes[i].hist(df[col].dropna(), bins=30, edgecolor="white", color="#4C72B0")
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")

    plt.suptitle("Numeric Features Distribution", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "01_numeric_distributions.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  [Saved] {path}")


def plot_correlations(df: pd.DataFrame) -> None:
    corr = df[NUMERIC_COLS].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Correlation Matrix — Numeric Features")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "01_correlation_matrix.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  [Saved] {path}")


def plot_age_readmit(df: pd.DataFrame) -> None:
    tbl = (
        df.groupby("age")["readmitted"]
        .apply(lambda x: (x == "<30").mean() * 100)
        .reindex(AGE_ORDER)
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    tbl.plot(kind="bar", ax=ax, color="#DD8452", edgecolor="white")
    ax.set_title("Early Readmission Rate by Age Group")
    ax.set_ylabel("Early Readmission %")
    ax.set_xlabel("Age Group")
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "01_readmit_by_age.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  [Saved] {path}")


def data_quality_summary(df: pd.DataFrame) -> None:
    log.info(f"\n{SEP2}")
    log.info("DATA QUALITY SUMMARY & PREPROCESSING RECOMMENDATIONS")
    log.info(SEP2)

    issues = []

    # High missing
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        if pct > 40:
            issues.append(f"  [HIGH MISSING >40%]  {col}  ({pct:.1f}%) — RECOMMEND: DROP")
        elif pct > 5:
            issues.append(f"  [MISSING >5%]        {col}  ({pct:.1f}%) — RECOMMEND: IMPUTE or DROP")

    # Constant / near-constant columns
    for col in MEDICATION_COLS:
        if col not in df.columns:
            continue
        pct_no = (df[col] == "No").mean() * 100
        if pct_no > 98:
            issues.append(f"  [NEAR-CONSTANT ~No]  {col}  ({pct_no:.1f}% No) — RECOMMEND: DROP")

    # Identifier columns
    for col in ID_COLS:
        issues.append(f"  [IDENTIFIER]         {col} — RECOMMEND: DROP before modeling")

    # Gender invalid
    if "gender" in df.columns:
        n_inv = (df["gender"] == "Unknown/Invalid").sum()
        if n_inv > 0:
            issues.append(f"  [INVALID GENDER]     {n_inv} rows with 'Unknown/Invalid' — RECOMMEND: DROP rows")

    log.info(f"  Found {len(issues)} data quality notes:\n")
    for iss in issues:
        log.info(iss)

    log.info(f"\n{SEP2}")
    log.info("PREPROCESSING PLAN (to be executed in 02_preprocess.py)")
    log.info(SEP2)
    steps = [
        "1.  Replace '?' with NaN",
        "2.  Remove expired/hospice patients (discharge_disposition_id in {11,13,14,19,20,21})",
        "3.  Remove rows with Unknown/Invalid gender",
        "4.  Keep only first encounter per patient (avoid data leakage from repeat visits)",
        "5.  Drop weight (>96% missing), payer_code (>40% missing), encounter_id, patient_nbr",
        "6.  Impute medical_specialty NaN → 'Unknown'",
        "7.  Impute race NaN → 'Unknown'",
        "8.  Impute diag_1/2/3 NaN → 'Unknown', then map to ICD-9 category",
        "9.  Encode age as ordinal integer [0-10) → 0 … [90-100) → 9",
        "10. Map admission_type_id / discharge_disposition_id / admission_source_id to labels",
        "11. Encode medication columns: No → 0, Steady/Up/Down → 1 (was_prescribed)",
        "12. One-hot encode remaining nominal categoricals",
        "13. Create binary target: early_readmit = 1 if readmitted == '<30' else 0",
        "14. Save preprocessed dataset to data_processed/diabetic_preprocessed.csv",
    ]
    for s in steps:
        log.info(f"  {s}")


def main():
    df = load_raw(RAW_DATA_PATH)
    df_nan = basic_info(df)
    target_analysis(df_nan)
    duplicate_analysis(df_nan)
    expired_analysis(df_nan)
    numeric_analysis(df_nan)
    categorical_analysis(df_nan)
    medication_analysis(df_nan)
    diagnosis_analysis(df_nan)
    age_analysis(df_nan)
    weight_analysis(df_nan)

    log.info(f"\n{SEP2}")
    log.info("GENERATING VISUALIZATIONS ...")
    log.info(SEP2)
    plot_target(df_nan)
    plot_numeric(df_nan)
    plot_correlations(df_nan)
    plot_age_readmit(df_nan)

    data_quality_summary(df_nan)

    log.info(f"\n{SEP}")
    log.info("EXPLORATION COMPLETE")
    log.info(f"Check logs/ for the full report and outputs/figures/ for plots.")
    log.info(SEP)


if __name__ == "__main__":
    main()
