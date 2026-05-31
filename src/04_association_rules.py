"""
Step 4 — Association Rule Mining (Apriori)
Uses the binarised dataset produced by 02_preprocess.py.

Reads  : data_processed/diabetic_preprocessed_for_rules.csv
Writes : outputs/results/association_rules_*.csv
         outputs/figures/04_*.png
         logs/04_association_rules_<timestamp>.log

Usage:
    python src/04_association_rules.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from utils import PROCESSED_DIR, FIGURES_DIR, RESULTS_DIR, get_logger

log = get_logger("04_association_rules")
SEP = "=" * 80
SEP2 = "-" * 60

RULES_DATA_PATH = os.path.join(PROCESSED_DIR, "diabetic_preprocessed_for_rules.csv")


def load_data() -> pd.DataFrame:
    log.info(SEP)
    log.info("ASSOCIATION RULE MINING — Apriori")
    log.info(SEP)

    df = pd.read_csv(RULES_DATA_PATH)
    log.info(f"Loaded binarised dataset: {df.shape[0]:,} rows  x  {df.shape[1]} columns")

    # Verify binary
    non_binary = [c for c in df.columns if not set(df[c].dropna().unique()).issubset({0, 1})]
    if non_binary:
        log.info(f"  Warning: non-binary columns (will be binarised by >0): {non_binary[:10]}")
        for col in non_binary:
            df[col] = (df[col] > 0).astype(int)

    return df


def run_apriori(df: pd.DataFrame, min_support: float = 0.05) -> pd.DataFrame:
    log.info(f"\n{SEP2}")
    log.info(f"APRIORI — min_support={min_support}")
    log.info(SEP2)

    # mlxtend apriori expects boolean DataFrame
    df_bool = df.astype(bool)

    log.info(f"  Running Apriori on {df_bool.shape[0]:,} transactions, "
             f"{df_bool.shape[1]} items ...")

    freq_items = apriori(df_bool, min_support=min_support,
                         use_colnames=True, max_len=4)

    log.info(f"  Frequent itemsets found: {len(freq_items):,}")
    log.info(f"  Length distribution:")
    ld = freq_items["itemsets"].apply(len).value_counts().sort_index()
    for length, count in ld.items():
        log.info(f"    length {length}: {count:,} itemsets")

    return freq_items


def generate_rules(freq_items: pd.DataFrame,
                   min_confidence: float = 0.5,
                   min_lift: float = 1.0) -> pd.DataFrame:
    log.info(f"\n{SEP2}")
    log.info(f"RULE GENERATION — min_confidence={min_confidence}, min_lift={min_lift}")
    log.info(SEP2)

    rules = association_rules(freq_items, metric="lift", min_threshold=min_lift)
    rules = rules[rules["confidence"] >= min_confidence]
    rules = rules.sort_values("lift", ascending=False)

    log.info(f"  Total rules generated: {len(rules):,}")

    for metric in ["support", "confidence", "lift"]:
        log.info(f"  {metric}: min={rules[metric].min():.3f}  "
                 f"mean={rules[metric].mean():.3f}  "
                 f"max={rules[metric].max():.3f}")

    return rules


def analyze_readmission_rules(rules: pd.DataFrame) -> None:
    log.info(f"\n{SEP2}")
    log.info("RULES INVOLVING early_readmit (target-focused analysis)")
    log.info(SEP2)

    # Rules where early_readmit is in the consequent
    readmit_rules = rules[
        rules["consequents"].apply(lambda x: "early_readmit" in x)
    ].sort_values("lift", ascending=False)

    log.info(f"  Rules with early_readmit as consequent: {len(readmit_rules)}")

    if len(readmit_rules) > 0:
        top = readmit_rules.head(20)
        for _, row in top.iterrows():
            ant = ", ".join(list(row["antecedents"]))
            con = ", ".join(list(row["consequents"]))
            log.info(
                f"    {ant:<60} → {con:<20}  "
                f"sup={row['support']:.3f}  "
                f"conf={row['confidence']:.3f}  "
                f"lift={row['lift']:.3f}"
            )
        top.to_csv(os.path.join(RESULTS_DIR, "association_rules_readmit.csv"), index=False)
        log.info(f"  [Saved] outputs/results/association_rules_readmit.csv")

    # Rules against readmission (early_readmit NOT in consequent but in antecedent)
    anti_rules = rules[
        rules["antecedents"].apply(lambda x: "early_readmit" in x)
    ].sort_values("confidence", ascending=False)
    log.info(f"\n  Rules with early_readmit in antecedent: {len(anti_rules)}")


def top_rules_analysis(rules: pd.DataFrame) -> None:
    log.info(f"\n{SEP2}")
    log.info("TOP RULES BY LIFT (all, regardless of target)")
    log.info(SEP2)

    top50 = rules.head(50)
    log.info(f"  Top 50 rules by lift:")
    for i, row in top50.iterrows():
        ant = ", ".join(list(row["antecedents"]))
        con = ", ".join(list(row["consequents"]))
        log.info(
            f"    [{i:4d}] {ant:<60} → {con:<40}  "
            f"sup={row['support']:.3f}  conf={row['confidence']:.3f}  lift={row['lift']:.3f}"
        )

    top50.to_csv(os.path.join(RESULTS_DIR, "association_rules_top50.csv"), index=False)


def save_all_rules(rules: pd.DataFrame) -> None:
    rules_save = rules.copy()
    rules_save["antecedents"] = rules_save["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules_save["consequents"] = rules_save["consequents"].apply(lambda x: ", ".join(list(x)))
    path = os.path.join(RESULTS_DIR, "association_rules_all.csv")
    rules_save.to_csv(path, index=False)
    log.info(f"\n  [Saved] All {len(rules):,} rules → {path}")


def plot_rules(rules: pd.DataFrame, freq_items: pd.DataFrame) -> None:
    log.info(f"\n{SEP2}")
    log.info("PLOTTING ASSOCIATION RULES")
    log.info(SEP2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Support vs Confidence
    scatter = axes[0].scatter(
        rules["support"], rules["confidence"],
        c=rules["lift"], cmap="YlOrRd", alpha=0.5, s=10
    )
    plt.colorbar(scatter, ax=axes[0], label="Lift")
    axes[0].set_xlabel("Support")
    axes[0].set_ylabel("Confidence")
    axes[0].set_title("Support vs Confidence (colour=Lift)")

    # Lift distribution
    axes[1].hist(rules["lift"], bins=40, color="#4C72B0", edgecolor="white")
    axes[1].set_xlabel("Lift")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Lift Distribution")
    axes[1].axvline(rules["lift"].mean(), color="red", linestyle="--",
                    label=f"mean={rules['lift'].mean():.2f}")
    axes[1].legend()

    # Top 15 itemsets by support
    top_items = freq_items.sort_values("support", ascending=False).head(15).copy()
    top_items["itemsets_str"] = top_items["itemsets"].apply(lambda x: ", ".join(list(x)))
    axes[2].barh(top_items["itemsets_str"], top_items["support"],
                 color="#55A868", edgecolor="white")
    axes[2].set_xlabel("Support")
    axes[2].set_title("Top 15 Frequent Itemsets")
    axes[2].invert_yaxis()

    plt.suptitle("Apriori Association Rules Overview", fontsize=13)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "04_association_rules_overview.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  [Saved] {path}")

    # Heatmap: top antecedent-consequent pairs
    if len(rules) > 0:
        sample = rules.head(30).copy()
        sample["antecedents"] = sample["antecedents"].apply(lambda x: ", ".join(list(x)))
        sample["consequents"] = sample["consequents"].apply(lambda x: ", ".join(list(x)))
        pivot = sample.pivot_table(
            index="antecedents", columns="consequents", values="lift", aggfunc="max"
        )
        if pivot.shape[0] > 1 and pivot.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.5),
                                           max(6, pivot.shape[0] * 0.4)))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                        ax=ax, linewidths=0.5, cbar_kws={"label": "Lift"})
            ax.set_title("Lift Heatmap — Top 30 Rules")
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, "04_rules_lift_heatmap.png"),
                        dpi=120, bbox_inches="tight")
            plt.close()


def sensitivity_analysis(df: pd.DataFrame) -> None:
    """Show how rule count changes with different support thresholds."""
    log.info(f"\n{SEP2}")
    log.info("SENSITIVITY ANALYSIS — Support Threshold vs Rule Count")
    log.info(SEP2)

    df_bool = df.astype(bool)
    supports = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    rows = []

    for s in supports:
        fi = apriori(df_bool, min_support=s, use_colnames=True, max_len=3)
        if len(fi) > 0:
            r = association_rules(fi, metric="confidence", min_threshold=0.5)
            n_rules = len(r)
        else:
            n_rules = 0
        n_items = len(fi)
        rows.append(dict(min_support=s, n_itemsets=n_items, n_rules=n_rules))
        log.info(f"  support={s:.2f}  itemsets={n_items:,}  rules={n_rules:,}")

    sens_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(sens_df["min_support"], sens_df["n_itemsets"], "bo-")
    axes[0].set_title("Frequent Itemsets vs Min Support")
    axes[0].set_xlabel("Min Support"); axes[0].set_ylabel("Count")
    axes[1].plot(sens_df["min_support"], sens_df["n_rules"], "ro-")
    axes[1].set_title("Rules vs Min Support")
    axes[1].set_xlabel("Min Support"); axes[1].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "04_sensitivity_support.png"),
                dpi=120, bbox_inches="tight")
    plt.close()


def main():
    df = load_data()

    sensitivity_analysis(df)

    freq_items = run_apriori(df, min_support=0.05)
    rules = generate_rules(freq_items, min_confidence=0.5, min_lift=1.0)

    analyze_readmission_rules(rules)
    top_rules_analysis(rules)
    save_all_rules(rules)
    plot_rules(rules, freq_items)

    log.info(f"\n{SEP}")
    log.info("ASSOCIATION RULE MINING COMPLETE")
    log.info(SEP)


if __name__ == "__main__":
    main()
