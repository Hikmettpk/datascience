"""
Step 5 — Predictive Analytics (Classification)
Predicts early hospital readmission (<30 days).

Experiments (10 total):
  1.  KNN         — k=5, all features, SMOTE
  2.  KNN         — k=11, top-20 ANOVA, SMOTE
  3.  Naive Bayes — GaussianNB, all features, SMOTE
  4.  Decision Tree — max_depth=5, SMOTE
  5.  Decision Tree — max_depth=10, min_leaf=20, SMOTE
  6.  Random Forest — n=100, class_weight=balanced (no SMOTE)
  7.  Random Forest — n=200 tuned, class_weight=balanced (no SMOTE)
  8.  XGBoost     — n=100, scale_pos_weight (no SMOTE)
  9.  LinearSVM   — top-20 features, SMOTE
  10. MLP         — 2 hidden layers, top-30 features, SMOTE

Reads  : data_processed/diabetic_preprocessed.csv
Writes : outputs/results/classification_results.csv
         outputs/results/feature_rankings.csv
         outputs/figures/05_*.png
         logs/05_classification_<timestamp>.log

Usage:
    python src/05_classification.py
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
from scipy import stats

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
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
SEP  = "=" * 80
SEP2 = "-" * 60

RANDOM_STATE = 42
CV_FOLDS     = 5


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(path: str) -> tuple[np.ndarray, np.ndarray, list]:
    log.info(SEP)
    log.info("CLASSIFICATION — Early Readmission Prediction")
    log.info(SEP)

    df = pd.read_csv(path)
    log.info(f"Loaded: {df.shape[0]:,} rows  x  {df.shape[1]} columns")

    if "early_readmit" not in df.columns:
        log.error("Column 'early_readmit' not found. Run 02_preprocess.py first.")
        sys.exit(1)

    y    = df["early_readmit"].values
    X_df = df.drop(columns=["early_readmit"])
    feature_names = list(X_df.columns)
    X = X_df.values.astype(float)

    pos, neg = y.sum(), len(y) - y.sum()
    log.info(f"\nTarget: positive={pos:,} ({pos/len(y)*100:.1f}%)  "
             f"negative={neg:,} ({neg/len(y)*100:.1f}%)  "
             f"imbalance ratio 1:{neg/pos:.1f}")
    log.info(f"Features: {len(feature_names)}")
    return X, y, feature_names


# ── Feature selection ──────────────────────────────────────────────────────────

def feature_selection_report(X: np.ndarray, y: np.ndarray,
                              feature_names: list) -> dict:
    log.info(f"\n{SEP2}\nFEATURE SELECTION\n{SEP2}")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    results  = {}

    # 1. Variance threshold
    vt = VarianceThreshold(threshold=0.01)
    vt.fit(X_scaled)
    vt_mask = vt.get_support()
    results["variance_threshold"] = [f for f, m in zip(feature_names, vt_mask) if m]
    log.info(f"\n  Variance Threshold (>0.01): {sum(vt_mask)} / {len(feature_names)} features kept")

    # 2. ANOVA F-score
    f_scores = pd.Series(
        SelectKBest(f_classif, k="all").fit(X_scaled, y).scores_,
        index=feature_names,
    ).sort_values(ascending=False)
    results["anova_top30"] = f_scores.head(30).index.tolist()
    log.info(f"\n  ANOVA F-score — Top 30:")
    for feat, s in f_scores.head(30).items():
        log.info(f"    {feat:<60} {s:.3f}")

    # 3. Mutual Information
    mi_scores = pd.Series(
        SelectKBest(mutual_info_classif, k="all").fit(X_scaled, y).scores_,
        index=feature_names,
    ).sort_values(ascending=False)
    results["mutual_info_top30"] = mi_scores.head(30).index.tolist()
    log.info(f"\n  Mutual Information — Top 30:")
    for feat, s in mi_scores.head(30).items():
        log.info(f"    {feat:<60} {s:.5f}")

    # 4. Random Forest importance
    rf_imp = RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(X_scaled, y).feature_importances_
    rf_scores = pd.Series(rf_imp, index=feature_names).sort_values(ascending=False)
    results["rf_importance_top30"] = rf_scores.head(30).index.tolist()
    log.info(f"\n  Random Forest Importance — Top 30:")
    for feat, s in rf_scores.head(30).items():
        log.info(f"    {feat:<60} {s:.5f}")

    # Save
    rank_df = pd.DataFrame({
        "feature":       feature_names,
        "f_score":       f_scores.reindex(feature_names).values,
        "mutual_info":   mi_scores.reindex(feature_names).values,
        "rf_importance": rf_scores.reindex(feature_names).values,
    }).sort_values("rf_importance", ascending=False)
    rank_df.to_csv(os.path.join(RESULTS_DIR, "feature_rankings.csv"), index=False)
    log.info(f"\n  [Saved] outputs/results/feature_rankings.csv")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    for ax, (title, scores) in zip(axes, [
        ("ANOVA F-Score", f_scores),
        ("Mutual Information", mi_scores),
        ("RF Importance", rf_scores),
    ]):
        top = scores.head(25)
        ax.barh(top.index[::-1], top.values[::-1], color="#4C72B0", edgecolor="white")
        ax.set_title(f"Top 25 — {title}")
        ax.set_xlabel("Score")
    plt.suptitle("Feature Importance Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_feature_importance.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    return results


# ── Pipeline builders ──────────────────────────────────────────────────────────

def _smote_pipe(*steps) -> ImbPipeline:
    return ImbPipeline([("smote", SMOTE(random_state=RANDOM_STATE))] + list(steps))

def _plain_pipe(*steps) -> Pipeline:
    return Pipeline(list(steps))

def _scaler():
    return ("scaler", StandardScaler())


# ── CV evaluation ──────────────────────────────────────────────────────────────

def evaluate_model(
    name: str,
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    feature_subset: list | None = None,
) -> dict:
    if feature_subset is not None:
        idx   = [feature_names.index(f) for f in feature_subset if f in feature_names]
        X_use = X[:, idx]
    else:
        X_use = X

    cv     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(pipeline, X_use, y, cv=cv,
                            scoring=["accuracy", "f1", "precision", "recall", "roc_auc"],
                            return_train_score=False, n_jobs=1)

    result = dict(
        experiment   = name,
        n_features   = X_use.shape[1],
        accuracy     = round(scores["test_accuracy"].mean(),  4),
        f1           = round(scores["test_f1"].mean(),        4),
        precision    = round(scores["test_precision"].mean(), 4),
        recall       = round(scores["test_recall"].mean(),    4),
        roc_auc      = round(scores["test_roc_auc"].mean(),   4),
        accuracy_std = round(scores["test_accuracy"].std(),   4),
        f1_std       = round(scores["test_f1"].std(),         4),
        roc_auc_std  = round(scores["test_roc_auc"].std(),    4),
    )
    log.info(
        f"  {name:<52} acc={result['accuracy']:.4f}±{result['accuracy_std']:.3f}  "
        f"f1={result['f1']:.4f}±{result['f1_std']:.3f}  "
        f"auc={result['roc_auc']:.4f}±{result['roc_auc_std']:.3f}  "
        f"prec={result['precision']:.4f}  rec={result['recall']:.4f}"
    )
    return result


# ── 10 Experiments ─────────────────────────────────────────────────────────────

def run_experiments(X: np.ndarray, y: np.ndarray,
                    feature_names: list, feat_sets: dict) -> list:
    log.info(f"\n{SEP2}\nCLASSIFICATION EXPERIMENTS (10 experiments, {CV_FOLDS}-fold CV)\n{SEP2}")

    results = []
    top20   = feat_sets.get("anova_top30",        feature_names)[:20]
    top30   = feat_sets.get("rf_importance_top30", feature_names)[:30]
    s       = _scaler()

    # EXP01 — KNN k=5, all features
    results.append(evaluate_model(
        "EXP01_KNN_k5_all_features",
        _smote_pipe(s, ("clf", KNeighborsClassifier(n_neighbors=5,  n_jobs=-1))),
        X, y, feature_names,
    ))

    # EXP02 — KNN k=11, top-20 ANOVA
    results.append(evaluate_model(
        "EXP02_KNN_k11_top20_anova",
        _smote_pipe(s, ("clf", KNeighborsClassifier(n_neighbors=11, n_jobs=-1))),
        X, y, feature_names, feature_subset=top20,
    ))

    # EXP03 — Naive Bayes
    results.append(evaluate_model(
        "EXP03_NaiveBayes_GaussianNB",
        _smote_pipe(s, ("clf", GaussianNB())),
        X, y, feature_names,
    ))

    # EXP04 — Decision Tree depth=5
    results.append(evaluate_model(
        "EXP04_DecisionTree_depth5",
        _smote_pipe(s, ("clf", DecisionTreeClassifier(max_depth=5,
                                                      random_state=RANDOM_STATE))),
        X, y, feature_names,
    ))

    # EXP05 — Decision Tree depth=10
    results.append(evaluate_model(
        "EXP05_DecisionTree_depth10_min20",
        _smote_pipe(s, ("clf", DecisionTreeClassifier(max_depth=10,
                                                      min_samples_leaf=20,
                                                      random_state=RANDOM_STATE))),
        X, y, feature_names,
    ))

    # EXP06 — Random Forest n=100, class_weight=balanced (no SMOTE)
    # Note: class_weight handles imbalance at the loss level; more effective for RF than SMOTE.
    results.append(evaluate_model(
        "EXP06_RandomForest_n100_balanced",
        _plain_pipe(s, ("clf", RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1))),
        X, y, feature_names,
    ))

    # EXP07 — Random Forest n=200 tuned, class_weight=balanced, top-30 features
    results.append(evaluate_model(
        "EXP07_RandomForest_n200_tuned_balanced",
        _plain_pipe(s, ("clf", RandomForestClassifier(
            n_estimators=200, max_features="sqrt", max_depth=20,
            min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1))),
        X, y, feature_names, feature_subset=top30,
    ))

    # EXP08 — XGBoost with scale_pos_weight (no SMOTE)
    scale_pw = float((y == 0).sum()) / float((y == 1).sum())
    if HAS_XGB:
        clf8     = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=scale_pw,
                                 random_state=RANDOM_STATE, eval_metric="logloss",
                                 verbosity=0, n_jobs=-1)
        exp8_name = "EXP08_XGBoost_n100"
    else:
        clf8      = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                               learning_rate=0.1, subsample=0.8,
                                               random_state=RANDOM_STATE)
        exp8_name = "EXP08_GradientBoosting_n100"
    results.append(evaluate_model(
        exp8_name,
        _plain_pipe(s, ("clf", clf8)),
        X, y, feature_names,
    ))

    # EXP09 — Linear SVM, top-20, SMOTE
    linear_svm = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE), cv=3)
    results.append(evaluate_model(
        "EXP09_LinearSVM_top20",
        _smote_pipe(s, ("clf", linear_svm)),
        X, y, feature_names, feature_subset=top20,
    ))

    # EXP10 — MLP 2-layer, top-30, SMOTE
    results.append(evaluate_model(
        "EXP10_MLP_2layer_top30",
        _smote_pipe(s, ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu",
            solver="adam", alpha=0.001, max_iter=200,
            early_stopping=True, random_state=RANDOM_STATE))),
        X, y, feature_names, feature_subset=top30,
    ))

    return results


# ── Best-model pipeline builder ────────────────────────────────────────────────

def build_pipeline_for(exp_name: str, y_train: np.ndarray,
                       feat_sets: dict, feature_names: list):
    """Return (pipeline, feature_subset_or_None) for a given experiment name."""
    s        = _scaler()
    top20    = feat_sets.get("anova_top30",        feature_names)[:20]
    top30    = feat_sets.get("rf_importance_top30", feature_names)[:30]
    scale_pw = float((y_train == 0).sum()) / float((y_train == 1).sum())

    if "XGBoost" in exp_name or "GradientBoosting" in exp_name:
        clf = (XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=scale_pw,
                             random_state=RANDOM_STATE, eval_metric="logloss",
                             verbosity=0, n_jobs=-1)
               if HAS_XGB else
               GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                          learning_rate=0.1, subsample=0.8,
                                          random_state=RANDOM_STATE))
        return _plain_pipe(s, ("clf", clf)), None  # all features

    elif "LinearSVM" in exp_name:
        clf = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE), cv=3)
        return _smote_pipe(s, ("clf", clf)), top20

    elif "n200" in exp_name:
        clf = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                     max_depth=20, min_samples_leaf=5,
                                     class_weight="balanced",
                                     random_state=RANDOM_STATE, n_jobs=-1)
        return _plain_pipe(s, ("clf", clf)), top30

    elif "n100" in exp_name and "RandomForest" in exp_name:
        clf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                     random_state=RANDOM_STATE, n_jobs=-1)
        return _plain_pipe(s, ("clf", clf)), None

    elif "KNN_k5" in exp_name:
        clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        return _smote_pipe(s, ("clf", clf)), None

    elif "KNN_k11" in exp_name:
        clf = KNeighborsClassifier(n_neighbors=11, n_jobs=-1)
        return _smote_pipe(s, ("clf", clf)), top20

    elif "DecisionTree_depth5" in exp_name:
        clf = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
        return _smote_pipe(s, ("clf", clf)), None

    elif "DecisionTree_depth10" in exp_name:
        clf = DecisionTreeClassifier(max_depth=10, min_samples_leaf=20,
                                     random_state=RANDOM_STATE)
        return _smote_pipe(s, ("clf", clf)), None

    elif "MLP" in exp_name:
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                            solver="adam", alpha=0.001, max_iter=200,
                            early_stopping=True, random_state=RANDOM_STATE)
        return _smote_pipe(s, ("clf", clf)), top30

    elif "NaiveBayes" in exp_name:
        return _smote_pipe(s, ("clf", GaussianNB())), None

    # fallback
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                 random_state=RANDOM_STATE, n_jobs=-1)
    return _plain_pipe(s, ("clf", clf)), None


def get_feature_array(X: np.ndarray, subset: list | None,
                      feature_names: list) -> np.ndarray:
    if subset is None:
        return X
    idx = [feature_names.index(f) for f in subset if f in feature_names]
    return X[:, idx]


# ── Best model deep dive ───────────────────────────────────────────────────────

def best_model_deep_dive(results: list, X: np.ndarray, y: np.ndarray,
                         feature_names: list, feat_sets: dict) -> None:
    log.info(f"\n{SEP2}\nBEST MODEL — DEEP DIVE\n{SEP2}")

    res_df    = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    best_name = res_df.iloc[0]["experiment"]
    log.info(f"  Best experiment (by ROC-AUC): {best_name}")
    log.info(res_df.head(5).to_string(index=False))

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    pipe, subset = build_pipeline_for(best_name, y_train, feat_sets, feature_names)
    X_tr = get_feature_array(X_train, subset, feature_names)
    X_te = get_feature_array(X_test,  subset, feature_names)

    pipe.fit(X_tr, y_train)
    y_pred = pipe.predict(X_te)
    y_prob = pipe.predict_proba(X_te)[:, 1]

    # ── Classification report at default threshold ────────────────────────────
    log.info(f"\n  Classification Report — default threshold (0.5):")
    log.info(classification_report(y_test, y_pred, target_names=["Not <30", "<30"]))
    auc = roc_auc_score(y_test, y_prob)
    log.info(f"  ROC-AUC (hold-out): {auc:.4f}")

    # ── Optimal threshold (max F1) ────────────────────────────────────────────
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    f1s = [f1_score(y_test, (y_prob >= t).astype(int), zero_division=0)
           for t in thresholds]
    best_thresh = thresholds[int(np.argmax(f1s))]
    y_pred_opt  = (y_prob >= best_thresh).astype(int)
    log.info(f"\n  Optimal threshold (max F1): {best_thresh:.3f}")
    log.info(f"  Classification Report — optimal threshold:")
    log.info(classification_report(y_test, y_pred_opt, target_names=["Not <30", "<30"]))

    # ── Figure: CM (default) | CM (optimal) | ROC ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for ax, (preds, title) in zip(axes[:2], [
        (y_pred,     f"Confusion Matrix — Default Threshold (0.50)"),
        (y_pred_opt, f"Confusion Matrix — Optimal Threshold ({best_thresh:.2f})"),
    ]):
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Not <30", "<30"],
                    yticklabels=["Not <30", "<30"])
        ax.set_title(f"{title}\n({best_name})", fontsize=9)
        ax.set_ylabel("True"); ax.set_xlabel("Predicted")

    axes[2].plot(fpr, tpr, color="#DD8452", lw=2, label=f"AUC = {auc:.3f}")
    axes[2].plot([0, 1], [0, 1], "k--", lw=1)
    axes[2].axvline(fpr[int(np.argmax(f1s))], color="gray", linestyle=":",
                    label=f"Opt. threshold = {best_thresh:.2f}")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].set_title(f"ROC Curve — {best_name}")
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_best_model_cm_roc.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  [Saved] 05_best_model_cm_roc.png")

    # ── F1 vs Threshold plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, f1s, color="#4C72B0", lw=2)
    ax.axvline(best_thresh, color="#DD8452", linestyle="--",
               label=f"Optimal = {best_thresh:.2f}  (F1={max(f1s):.3f})")
    ax.axvline(0.5, color="gray", linestyle=":", label="Default = 0.50")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"F1 Score vs Decision Threshold — {best_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_threshold_f1.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  [Saved] 05_threshold_f1.png")

    # Continue with ROC comparison
    plot_roc_comparison(X, y, feature_names, feat_sets)


# ── Multi-model ROC comparison ─────────────────────────────────────────────────

def plot_roc_comparison(X: np.ndarray, y: np.ndarray,
                        feature_names: list, feat_sets: dict) -> None:
    log.info(f"\n  Plotting multi-model ROC comparison ...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    experiments_for_roc = [
        "EXP08_XGBoost_n100",
        "EXP06_RandomForest_n100_balanced",
        "EXP05_DecisionTree_depth10_min20",
        "EXP09_LinearSVM_top20",
        "EXP10_MLP_2layer_top30",
    ]
    colors = ["#DD8452", "#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(9, 7))

    for exp_name, color in zip(experiments_for_roc, colors):
        pipe, subset = build_pipeline_for(exp_name, y_train, feat_sets, feature_names)
        X_tr = get_feature_array(X_train, subset, feature_names)
        X_te = get_feature_array(X_test,  subset, feature_names)
        pipe.fit(X_tr, y_train)
        y_prob = pipe.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        label = exp_name.replace("EXP0", "").replace("EXP1", "10_")
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC={auc:.3f})")
        log.info(f"    {exp_name:<45} AUC={auc:.4f}")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison — Top 5 Models")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_roc_comparison.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  [Saved] 05_roc_comparison.png")


# ── Statistical significance (Wilcoxon) ───────────────────────────────────────

def statistical_significance(results: list, X: np.ndarray, y: np.ndarray,
                              feature_names: list, feat_sets: dict) -> None:
    log.info(f"\n{SEP2}\nSTATISTICAL SIGNIFICANCE — Best vs 2nd Best (Wilcoxon)\n{SEP2}")

    res_df      = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    best_name   = res_df.iloc[0]["experiment"]
    second_name = res_df.iloc[1]["experiment"]
    log.info(f"  Best  : {best_name}   AUC={res_df.iloc[0]['roc_auc']:.4f}")
    log.info(f"  Second: {second_name} AUC={res_df.iloc[1]['roc_auc']:.4f}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def get_fold_aucs(exp_name: str) -> np.ndarray:
        pipe, subset = build_pipeline_for(exp_name, y, feat_sets, feature_names)
        X_use = get_feature_array(X, subset, feature_names)
        aucs  = []
        for tr, te in cv.split(X_use, y):
            pipe.fit(X_use[tr], y[tr])
            prob = pipe.predict_proba(X_use[te])[:, 1]
            aucs.append(roc_auc_score(y[te], prob))
        return np.array(aucs)

    log.info(f"\n  Computing per-fold AUCs (this may take a few minutes) ...")
    best_aucs   = get_fold_aucs(best_name)
    second_aucs = get_fold_aucs(second_name)

    stat, p = stats.wilcoxon(best_aucs, second_aucs)

    log.info(f"\n  {best_name} fold AUCs   : {np.round(best_aucs,   4)}")
    log.info(f"  {second_name} fold AUCs: {np.round(second_aucs, 4)}")
    log.info(f"\n  Wilcoxon signed-rank test:")
    log.info(f"    statistic = {stat:.4f}")
    log.info(f"    p-value   = {p:.4f}")
    if p < 0.05:
        log.info(f"    → STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        log.info(f"    → Not statistically significant (p ≥ 0.05)")
        log.info(f"    → Models perform similarly; both are valid choices.")


# ── Result comparison plot ─────────────────────────────────────────────────────

def plot_results(results: list) -> None:
    res_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    metrics = ["accuracy", "f1", "roc_auc", "recall"]
    titles  = ["Accuracy", "F1 Score", "ROC-AUC", "Recall"]
    colors  = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]

    for ax, metric, title, color in zip(axes.flatten(), metrics, titles, colors):
        bars = ax.barh(res_df["experiment"][::-1], res_df[metric][::-1],
                       color=color, edgecolor="white", alpha=0.85)
        ax.set_xlabel(title)
        ax.set_title(f"Experiment Comparison — {title}")
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
        for bar, val in zip(bars, res_df[metric][::-1]):
            ax.text(bar.get_width() + 0.003,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)

    plt.suptitle("Classification Experiment Results", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_experiment_comparison.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  [Saved] 05_experiment_comparison.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    X, y, feature_names = load_data(PROCESSED_DATA_PATH)
    feat_sets = feature_selection_report(X, y, feature_names)

    log.info(f"\n{SEP2}\nRUNNING 10 CLASSIFICATION EXPERIMENTS\n{SEP2}")
    results = run_experiments(X, y, feature_names, feat_sets)

    res_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    res_df.to_csv(os.path.join(RESULTS_DIR, "classification_results.csv"), index=False)

    log.info(f"\n{SEP2}\nFINAL RESULTS TABLE (sorted by ROC-AUC)\n{SEP2}")
    log.info(res_df.to_string(index=False))

    plot_results(results)
    best_model_deep_dive(results, X, y, feature_names, feat_sets)
    statistical_significance(results, X, y, feature_names, feat_sets)

    log.info(f"\n{SEP}\nCLASSIFICATION COMPLETE\n{SEP}")


if __name__ == "__main__":
    main()
