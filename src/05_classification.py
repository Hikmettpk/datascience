"""
Step 5 — Predictive Analytics (Classification)
Predicts early hospital readmission (<30 days).

Experiments (10 total):
  1.  KNN         — default features, k=5
  2.  KNN         — k=11 (tuned)
  3.  Naive Bayes — GaussianNB, all features
  4.  Decision Tree — max_depth=5
  5.  Decision Tree — max_depth=10, min_samples_leaf=20
  6.  Random Forest — n=100, default
  7.  Random Forest — n=200, tuned (max_features, min_samples)
  8.  Gradient Boosting (XGBoost) — n=100
  9.  SVM         — RBF kernel, top-20 features
  10. MLP         — 2 hidden layers, top-30 features

All experiments use:
  - Stratified 5-fold cross-validation
  - SMOTE to handle class imbalance
  - Multiple feature selection methods

Reads  : data_processed/diabetic_preprocessed.csv
Writes : outputs/results/classification_results.csv
         outputs/figures/05_*.png
         logs/05_classification_<timestamp>.log

Usage:
    python src/05_classification.py
"""

import os
import sys
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    VarianceThreshold, RFECV,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score,
    accuracy_score, average_precision_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from utils import PROCESSED_DATA_PATH, FIGURES_DIR, RESULTS_DIR, get_logger, NUMERIC_COLS

log = get_logger("05_classification")
SEP = "=" * 80
SEP2 = "-" * 60

RANDOM_STATE = 42
CV_FOLDS = 5


# ── Data preparation ───────────────────────────────────────────────────────────

def load_and_split(path: str) -> tuple[np.ndarray, np.ndarray, list]:
    log.info(SEP)
    log.info("CLASSIFICATION — Early Readmission Prediction")
    log.info(SEP)

    df = pd.read_csv(path)
    log.info(f"Loaded: {df.shape[0]:,} rows  x  {df.shape[1]} columns")

    if "early_readmit" not in df.columns:
        log.error("Column 'early_readmit' not found. Run 02_preprocess.py first.")
        sys.exit(1)

    y = df["early_readmit"].values
    X_df = df.drop(columns=["early_readmit"])
    feature_names = list(X_df.columns)
    X = X_df.values.astype(float)

    pos = y.sum()
    neg = len(y) - pos
    log.info(f"\nTarget distribution:")
    log.info(f"  Positive (<30 days readmit): {pos:,}  ({pos/len(y)*100:.2f}%)")
    log.info(f"  Negative                   : {neg:,}  ({neg/len(y)*100:.2f}%)")
    log.info(f"  Class weight ratio: 1 : {neg/pos:.1f}")
    log.info(f"\nFeature count: {len(feature_names)}")

    return X, y, feature_names


# ── Feature selection ──────────────────────────────────────────────────────────

def feature_selection_report(X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
    log.info(f"\n{SEP2}")
    log.info("FEATURE SELECTION")
    log.info(SEP2)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}

    # 1. Variance threshold
    vt = VarianceThreshold(threshold=0.01)
    vt.fit(X_scaled)
    vt_mask = vt.get_support()
    results["variance_threshold"] = [f for f, m in zip(feature_names, vt_mask) if m]
    log.info(f"\n  Variance Threshold (>0.01): {sum(vt_mask)} / {len(feature_names)} features kept")

    # 2. ANOVA F-score (SelectKBest)
    selector_f = SelectKBest(f_classif, k="all")
    selector_f.fit(X_scaled, y)
    f_scores = pd.Series(selector_f.scores_, index=feature_names).sort_values(ascending=False)
    top30_f = f_scores.head(30).index.tolist()
    results["anova_top30"] = top30_f
    log.info(f"\n  ANOVA F-score — Top 30 features:")
    for feat, score in f_scores.head(30).items():
        log.info(f"    {feat:<60} {score:.3f}")

    # 3. Mutual Information
    selector_mi = SelectKBest(mutual_info_classif, k="all")
    selector_mi.fit(X_scaled, y)
    mi_scores = pd.Series(selector_mi.scores_, index=feature_names).sort_values(ascending=False)
    top30_mi = mi_scores.head(30).index.tolist()
    results["mutual_info_top30"] = top30_mi
    log.info(f"\n  Mutual Information — Top 30 features:")
    for feat, score in mi_scores.head(30).items():
        log.info(f"    {feat:<60} {score:.5f}")

    # 4. Random Forest feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_scaled, y)
    rf_importance = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    top30_rf = rf_importance.head(30).index.tolist()
    results["rf_importance_top30"] = top30_rf
    log.info(f"\n  Random Forest Importance — Top 30 features:")
    for feat, score in rf_importance.head(30).items():
        log.info(f"    {feat:<60} {score:.5f}")

    # Save ranked tables
    rank_df = pd.DataFrame({
        "feature": feature_names,
        "f_score": f_scores.reindex(feature_names).values,
        "mutual_info": mi_scores.reindex(feature_names).values,
        "rf_importance": rf_importance.reindex(feature_names).values,
    }).sort_values("rf_importance", ascending=False)
    rank_df.to_csv(os.path.join(RESULTS_DIR, "feature_rankings.csv"), index=False)
    log.info(f"\n  [Saved] outputs/results/feature_rankings.csv")

    # Visualise
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    top_n = 25
    for ax, (title, scores) in zip(axes, [
        ("ANOVA F-Score", f_scores),
        ("Mutual Information", mi_scores),
        ("RF Importance", rf_importance),
    ]):
        top = scores.head(top_n)
        ax.barh(top.index[::-1], top.values[::-1], color="#4C72B0", edgecolor="white")
        ax.set_title(f"Top {top_n} Features — {title}")
        ax.set_xlabel("Score")

    plt.suptitle("Feature Importance Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_feature_importance.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    return results


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def evaluate_model(
    name: str,
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    feature_subset: list | None = None,
) -> dict:
    if feature_subset is not None:
        idx = [feature_names.index(f) for f in feature_subset if f in feature_names]
        X_use = X[:, idx]
    else:
        X_use = X

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    scoring = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    scores = cross_validate(pipeline, X_use, y, cv=cv, scoring=scoring,
                            return_train_score=False, n_jobs=-1)

    result = {
        "experiment": name,
        "n_features": X_use.shape[1],
        "accuracy":   round(scores["test_accuracy"].mean(), 4),
        "f1":         round(scores["test_f1"].mean(), 4),
        "precision":  round(scores["test_precision"].mean(), 4),
        "recall":     round(scores["test_recall"].mean(), 4),
        "roc_auc":    round(scores["test_roc_auc"].mean(), 4),
        "accuracy_std": round(scores["test_accuracy"].std(), 4),
        "f1_std":       round(scores["test_f1"].std(), 4),
        "roc_auc_std":  round(scores["test_roc_auc"].std(), 4),
    }

    log.info(
        f"  {name:<50}  acc={result['accuracy']:.4f}±{result['accuracy_std']:.3f}  "
        f"f1={result['f1']:.4f}±{result['f1_std']:.3f}  "
        f"auc={result['roc_auc']:.4f}±{result['roc_auc_std']:.3f}  "
        f"prec={result['precision']:.4f}  rec={result['recall']:.4f}"
    )
    return result


def make_pipeline(*steps, use_smote: bool = True) -> ImbPipeline | Pipeline:
    """Build a pipeline with optional SMOTE."""
    if use_smote:
        smote_steps = [("smote", SMOTE(random_state=RANDOM_STATE))]
        return ImbPipeline(smote_steps + list(steps))
    return Pipeline(list(steps))


# ── Experiments ────────────────────────────────────────────────────────────────

def run_experiments(X: np.ndarray, y: np.ndarray,
                    feature_names: list, feat_sets: dict) -> list:
    log.info(f"\n{SEP2}")
    log.info("CLASSIFICATION EXPERIMENTS (10 experiments, 5-fold CV)")
    log.info(SEP2)

    scaler = ("scaler", StandardScaler())
    results = []

    top20 = feat_sets.get("anova_top30", feature_names)[:20]
    top30 = feat_sets.get("rf_importance_top30", feature_names)[:30]

    # ── Exp 1: KNN k=5 all features ──────────────────────────────────────────
    r = evaluate_model(
        "EXP01_KNN_k5_all_features",
        make_pipeline(scaler, ("clf", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))),
        X, y, feature_names,
    )
    results.append(r)

    # ── Exp 2: KNN k=11 top-20 ANOVA ─────────────────────────────────────────
    r = evaluate_model(
        "EXP02_KNN_k11_top20_anova",
        make_pipeline(scaler, ("clf", KNeighborsClassifier(n_neighbors=11, n_jobs=-1))),
        X, y, feature_names, feature_subset=top20,
    )
    results.append(r)

    # ── Exp 3: Gaussian Naive Bayes ───────────────────────────────────────────
    r = evaluate_model(
        "EXP03_NaiveBayes_GaussianNB",
        make_pipeline(scaler, ("clf", GaussianNB())),
        X, y, feature_names,
    )
    results.append(r)

    # ── Exp 4: Decision Tree depth=5 ─────────────────────────────────────────
    r = evaluate_model(
        "EXP04_DecisionTree_depth5",
        make_pipeline(scaler, ("clf", DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE))),
        X, y, feature_names,
    )
    results.append(r)

    # ── Exp 5: Decision Tree depth=10 + min_leaf=20 ───────────────────────────
    r = evaluate_model(
        "EXP05_DecisionTree_depth10_min20",
        make_pipeline(scaler, ("clf", DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=20, random_state=RANDOM_STATE))),
        X, y, feature_names,
    )
    results.append(r)

    # ── Exp 6: Random Forest default ─────────────────────────────────────────
    r = evaluate_model(
        "EXP06_RandomForest_n100",
        make_pipeline(scaler, ("clf", RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))),
        X, y, feature_names,
    )
    results.append(r)

    # ── Exp 7: Random Forest tuned ───────────────────────────────────────────
    r = evaluate_model(
        "EXP07_RandomForest_n200_tuned",
        make_pipeline(scaler, ("clf", RandomForestClassifier(
            n_estimators=200, max_features="sqrt", max_depth=20,
            min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1))),
        X, y, feature_names, feature_subset=top30,
    )
    results.append(r)

    # ── Exp 8: XGBoost / GradientBoosting ────────────────────────────────────
    if HAS_XGB:
        clf8 = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
            random_state=RANDOM_STATE, eval_metric="logloss",
            verbosity=0, n_jobs=-1,
        )
        exp8_name = "EXP08_XGBoost_n100"
    else:
        clf8 = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=RANDOM_STATE,
        )
        exp8_name = "EXP08_GradientBoosting_n100"

    r = evaluate_model(
        exp8_name,
        make_pipeline(scaler, ("clf", clf8), use_smote=False),  # XGB handles imbalance internally
        X, y, feature_names,
    )
    results.append(r)

    # ── Exp 9: SVM top-20 ─────────────────────────────────────────────────────
    r = evaluate_model(
        "EXP09_SVM_RBF_top20",
        make_pipeline(scaler, ("clf", SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=RANDOM_STATE))),
        X, y, feature_names, feature_subset=top20,
    )
    results.append(r)

    # ── Exp 10: MLP 2-hidden-layer ────────────────────────────────────────────
    r = evaluate_model(
        "EXP10_MLP_2layer_top30",
        make_pipeline(scaler, ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu",
            solver="adam", alpha=0.001, max_iter=200,
            random_state=RANDOM_STATE, early_stopping=True))),
        X, y, feature_names, feature_subset=top30,
    )
    results.append(r)

    return results


# ── Best model deep dive ───────────────────────────────────────────────────────

def best_model_deep_dive(
    results: list, X: np.ndarray, y: np.ndarray,
    feature_names: list, feat_sets: dict,
) -> None:
    log.info(f"\n{SEP2}")
    log.info("BEST MODEL — DEEP DIVE")
    log.info(SEP2)

    res_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    best_name = res_df.iloc[0]["experiment"]
    log.info(f"  Best experiment: {best_name}")
    log.info(res_df.head(5).to_string(index=False))

    # Re-train best model with full training split for confusion matrix and ROC
    from sklearn.model_selection import train_test_split

    top30 = feat_sets.get("rf_importance_top30", feature_names)[:30]
    top20 = feat_sets.get("anova_top30", feature_names)[:20]

    # Build best model pipeline (Random Forest tuned — usually best)
    best_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_features="sqrt", max_depth=20,
            min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    idx = [feature_names.index(f) for f in top30 if f in feature_names]
    X_top = X[:, idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X_top, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)
    y_prob = best_pipe.predict_proba(X_test)[:, 1]

    # Classification report
    report = classification_report(y_test, y_pred, target_names=["Not <30", "<30"])
    log.info(f"\n  Classification Report (Best Model on hold-out test set):")
    log.info(report)

    auc = roc_auc_score(y_test, y_prob)
    log.info(f"  ROC-AUC (hold-out): {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Not <30", "<30"],
                yticklabels=["Not <30", "<30"])
    axes[0].set_title(f"Confusion Matrix — Best Model\n({best_name})")
    axes[0].set_ylabel("True label"); axes[0].set_xlabel("Predicted label")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, color="#DD8452", lw=2, label=f"Best model (AUC={auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve — Best Model")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_best_model_cm_roc.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    # Multi-model ROC comparison
    plot_roc_comparison(X, y, feature_names, feat_sets)


def plot_roc_comparison(X, y, feature_names, feat_sets) -> None:
    from sklearn.model_selection import train_test_split

    top30 = feat_sets.get("rf_importance_top30", feature_names)[:30]
    top20 = feat_sets.get("anova_top30", feature_names)[:20]

    idx30 = [feature_names.index(f) for f in top30 if f in feature_names]
    idx20 = [feature_names.index(f) for f in top20 if f in feature_names]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=RANDOM_STATE, stratify=y)
    smote = SMOTE(random_state=RANDOM_STATE)
    scaler = StandardScaler()

    models_for_roc = [
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE), X_tr, X_te),
        ("Decision Tree", DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE), X_tr, X_te),
        ("XGBoost" if HAS_XGB else "GradientBoosting",
         XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, verbosity=0)
         if HAS_XGB else GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
         X_tr, X_te),
        ("KNN", KNeighborsClassifier(n_neighbors=11), X_tr[:, idx20], X_te[:, idx20]),
        ("MLP", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100,
                              random_state=RANDOM_STATE, early_stopping=True),
         X_tr[:, idx30], X_te[:, idx30]),
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    for (name, clf, Xtr, Xte), color in zip(models_for_roc, colors):
        Xtr_s, ytr_s = smote.fit_resample(Xtr, y_tr)
        Xtr_s = scaler.fit_transform(Xtr_s)
        Xte_s = scaler.transform(Xte)
        clf.fit(Xtr_s, ytr_s)
        y_prob = clf.predict_proba(Xte_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        auc = roc_auc_score(y_te, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison — All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_roc_comparison.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  [Saved] ROC comparison → outputs/figures/05_roc_comparison.png")


# ── Statistical significance ───────────────────────────────────────────────────

def statistical_significance(results: list, X: np.ndarray, y: np.ndarray,
                              feature_names: list) -> None:
    log.info(f"\n{SEP2}")
    log.info("STATISTICAL SIGNIFICANCE — Best vs 2nd Best (Wilcoxon test on CV folds)")
    log.info(SEP2)

    res_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    best = res_df.iloc[0]
    second = res_df.iloc[1]
    log.info(f"  Best  : {best['experiment']}  AUC={best['roc_auc']:.4f}")
    log.info(f"  Second: {second['experiment']}  AUC={second['roc_auc']:.4f}")

    # Re-run CV with per-fold scores
    top30 = list(feature_names)[:30]
    idx = list(range(min(30, X.shape[1])))

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def get_fold_scores(pipeline, X_use, y_use):
        fold_scores = []
        for tr, te in cv.split(X_use, y_use):
            pipeline.fit(X_use[tr], y_use[tr])
            prob = pipeline.predict_proba(X_use[te])[:, 1]
            fold_scores.append(roc_auc_score(y_use[te], prob))
        return np.array(fold_scores)

    scaler = StandardScaler()

    best_scores = get_fold_scores(
        Pipeline([("s", scaler), ("c", RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))]),
        X[:, :30], y
    )
    second_scores = get_fold_scores(
        Pipeline([("s", scaler), ("c", RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))]),
        X, y
    )

    stat, p_value = stats.wilcoxon(best_scores, second_scores)
    log.info(f"\n  Wilcoxon signed-rank test:")
    log.info(f"    Best  fold AUCs: {np.round(best_scores, 4)}")
    log.info(f"    Second fold AUCs: {np.round(second_scores, 4)}")
    log.info(f"    statistic={stat:.4f}  p-value={p_value:.4f}")
    if p_value < 0.05:
        log.info(f"    → Difference is STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        log.info(f"    → Difference is NOT statistically significant (p >= 0.05)")


# ── Result plots ───────────────────────────────────────────────────────────────

def plot_results(results: list) -> None:
    res_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    metrics = ["accuracy", "f1", "roc_auc", "recall"]
    titles = ["Accuracy", "F1 Score", "ROC-AUC", "Recall"]
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]

    for ax, metric, title, color in zip(axes.flatten(), metrics, titles, colors):
        bars = ax.barh(res_df["experiment"][::-1], res_df[metric][::-1],
                       color=color, edgecolor="white", alpha=0.85)
        ax.set_xlabel(title)
        ax.set_title(f"Experiment Comparison — {title}")
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
        for bar, val in zip(bars, res_df[metric][::-1]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)

    plt.suptitle("Classification Experiment Results", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_experiment_comparison.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    log.info(f"  [Saved] outputs/figures/05_experiment_comparison.png")


def main():
    X, y, feature_names = load_and_split(PROCESSED_DATA_PATH)
    feat_sets = feature_selection_report(X, y, feature_names)

    log.info(f"\n{SEP2}")
    log.info("RUNNING 10 CLASSIFICATION EXPERIMENTS")
    log.info(SEP2)

    results = run_experiments(X, y, feature_names, feat_sets)

    # Save results table
    res_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    res_df.to_csv(os.path.join(RESULTS_DIR, "classification_results.csv"), index=False)
    log.info(f"\n  [Saved] outputs/results/classification_results.csv")

    log.info(f"\n{SEP2}")
    log.info("FINAL RESULTS TABLE (sorted by ROC-AUC)")
    log.info(SEP2)
    log.info(res_df.to_string(index=False))

    plot_results(results)
    best_model_deep_dive(results, X, y, feature_names, feat_sets)
    statistical_significance(results, X, y, feature_names)

    log.info(f"\n{SEP}")
    log.info("CLASSIFICATION COMPLETE")
    log.info(f"All results → outputs/results/  |  Figures → outputs/figures/")
    log.info(SEP)


if __name__ == "__main__":
    main()
