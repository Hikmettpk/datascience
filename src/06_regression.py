"""
Step 6 — Predictive Analytics (Regression)
Predicts time_in_hospital (length of stay, days 1-14).

Experiments (10 total):
  1.  Linear Regression  — all features (baseline)
  2.  Ridge              — alpha=1.0,  all features
  3.  Ridge              — alpha=10.0, top-20 ANOVA
  4.  Lasso              — alpha=0.01, all features
  5.  Decision Tree      — max_depth=5
  6.  Decision Tree      — max_depth=10, min_samples_leaf=20
  7.  Random Forest      — n=100, all features
  8.  XGBoost / GBR     — n=100, all features
  9.  XGBoost            — no discharge_disposition (leakage sensitivity test)
  10. Poisson Regression — all features (count data baseline)

Leakage guard: lab_per_day, meds_per_day, procedures_per_day are derived
from time_in_hospital and are dropped before modelling.

Reads  : data_processed/diabetic_preprocessed.csv
Writes : outputs/results/regression_results.csv
         outputs/results/regression_feature_rankings.csv
         outputs/figures/06_*.png
         logs/06_regression_<timestamp>.log

Usage:
    python src/06_regression.py
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

from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, PoissonRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from utils import PROCESSED_DATA_PATH, FIGURES_DIR, RESULTS_DIR, get_logger

log = get_logger("06_regression")
SEP  = "=" * 80
SEP2 = "-" * 60

RANDOM_STATE = 42
CV_FOLDS     = 5

# Derived features that use time_in_hospital as denominator — drop to prevent leakage
LEAKY_FEATURES = {"lab_per_day", "meds_per_day", "procedures_per_day"}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(path: str) -> tuple[np.ndarray, np.ndarray, list]:
    log.info(SEP)
    log.info("REGRESSION — time_in_hospital Prediction (Length of Stay)")
    log.info(SEP)

    df = pd.read_csv(path)
    log.info(f"Loaded: {df.shape[0]:,} rows  x  {df.shape[1]} columns")

    if "time_in_hospital" not in df.columns:
        log.error("Column 'time_in_hospital' not found. Run 02_preprocess.py first.")
        sys.exit(1)

    y    = df["time_in_hospital"].values.astype(float)
    drop = {"time_in_hospital"} | LEAKY_FEATURES
    X_df = df.drop(columns=[c for c in drop if c in df.columns])
    feature_names = list(X_df.columns)
    X = X_df.values.astype(float)

    log.info(f"\nTarget — time_in_hospital:")
    log.info(f"  mean={y.mean():.2f}  median={np.median(y):.1f}  "
             f"std={y.std():.2f}  min={y.min():.0f}  max={y.max():.0f}")
    val_counts = pd.Series(y.astype(int)).value_counts().sort_index().to_dict()
    log.info(f"  Value counts: {val_counts}")
    log.info(f"\nFeatures: {len(feature_names)}  "
             f"(dropped leaky: {LEAKY_FEATURES & set(df.columns)})")
    return X, y, feature_names


# ── Target distribution ────────────────────────────────────────────────────────

def plot_target_distribution(y: np.ndarray) -> None:
    counts = pd.Series(y.astype(int)).value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(counts.index, counts.values, color="#4C72B0",
                edgecolor="white", width=0.7)
    axes[0].set_xlabel("Days in Hospital")
    axes[0].set_ylabel("Patient Count")
    axes[0].set_title("Target Distribution — time_in_hospital")
    axes[0].set_xticks(counts.index)
    for v, c in zip(counts.index, counts.values):
        axes[0].text(v, c + 300, str(c), ha="center", fontsize=7)

    axes[1].boxplot(y, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", color="white"),
                    medianprops=dict(color="white", linewidth=2))
    axes[1].set_ylabel("Days in Hospital")
    axes[1].set_title(f"Boxplot  (mean={y.mean():.2f}, std={y.std():.2f})")
    axes[1].set_xticks([])

    plt.suptitle("Length of Stay — Target Variable Analysis", fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "06_target_distribution.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  [Saved] {out}")


# ── Feature selection ──────────────────────────────────────────────────────────

def feature_selection_report(X: np.ndarray, y: np.ndarray,
                              feature_names: list) -> dict:
    log.info(f"\n{SEP2}\nFEATURE SELECTION\n{SEP2}")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    results  = {}

    # 1. ANOVA F-score (linear correlation proxy)
    f_scores = pd.Series(
        SelectKBest(f_regression, k="all").fit(X_scaled, y).scores_,
        index=feature_names,
    ).sort_values(ascending=False)
    results["anova_top30"] = f_scores.head(30).index.tolist()
    log.info(f"\n  ANOVA F-score — Top 20:")
    for feat, s in f_scores.head(20).items():
        log.info(f"    {feat:<60} {s:.3f}")

    # 2. Mutual Information (nonlinear relationships)
    mi_scores = pd.Series(
        mutual_info_regression(X_scaled, y, random_state=RANDOM_STATE),
        index=feature_names,
    ).sort_values(ascending=False)
    results["mutual_info_top30"] = mi_scores.head(30).index.tolist()
    log.info(f"\n  Mutual Information — Top 20:")
    for feat, s in mi_scores.head(20).items():
        log.info(f"    {feat:<60} {s:.5f}")

    # 3. Random Forest importance (multivariate, nonlinear)
    rf_imp = RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(X_scaled, y).feature_importances_
    rf_scores = pd.Series(rf_imp, index=feature_names).sort_values(ascending=False)
    results["rf_importance_top30"] = rf_scores.head(30).index.tolist()
    log.info(f"\n  Random Forest Importance — Top 20:")
    for feat, s in rf_scores.head(20).items():
        log.info(f"    {feat:<60} {s:.5f}")

    # Save
    rank_df = pd.DataFrame({
        "feature":       feature_names,
        "f_score":       f_scores.reindex(feature_names).values,
        "mutual_info":   mi_scores.reindex(feature_names).values,
        "rf_importance": rf_scores.reindex(feature_names).values,
    }).sort_values("rf_importance", ascending=False)
    rank_df.to_csv(
        os.path.join(RESULTS_DIR, "regression_feature_rankings.csv"), index=False
    )
    log.info(f"\n  [Saved] outputs/results/regression_feature_rankings.csv")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    for ax, (title, scores) in zip(axes, [
        ("ANOVA F-Score",     f_scores),
        ("Mutual Information", mi_scores),
        ("RF Importance",      rf_scores),
    ]):
        top = scores.head(25)
        ax.barh(top.index[::-1], top.values[::-1], color="#4C72B0", edgecolor="white")
        ax.set_title(f"Top 25 — {title}")
        ax.set_xlabel("Score")
    plt.suptitle("Feature Importance — Regression (time_in_hospital)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "06_feature_importance.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  [Saved] outputs/figures/06_feature_importance.png")

    return results


# ── CV evaluation helper ───────────────────────────────────────────────────────

def evaluate_model(
    name: str,
    pipeline: Pipeline,
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

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        pipeline, X_use, y, cv=cv,
        scoring=["neg_mean_absolute_error",
                 "neg_root_mean_squared_error",
                 "r2"],
        return_train_score=False, n_jobs=1,
    )

    mae  = -scores["test_neg_mean_absolute_error"]
    rmse = -scores["test_neg_root_mean_squared_error"]
    r2   =  scores["test_r2"]

    result = dict(
        experiment = name,
        n_features = X_use.shape[1],
        mae        = round(mae.mean(),  4),
        rmse       = round(rmse.mean(), 4),
        r2         = round(r2.mean(),   4),
        mae_std    = round(mae.std(),   4),
        rmse_std   = round(rmse.std(),  4),
        r2_std     = round(r2.std(),    4),
        _rmse_folds = rmse.tolist(),
    )
    log.info(
        f"  {name:<52} "
        f"MAE={result['mae']:.4f}±{result['mae_std']:.3f}  "
        f"RMSE={result['rmse']:.4f}±{result['rmse_std']:.3f}  "
        f"R²={result['r2']:.4f}±{result['r2_std']:.3f}"
    )
    return result


# ── 8 Experiments ──────────────────────────────────────────────────────────────

def run_experiments(X: np.ndarray, y: np.ndarray,
                    feature_names: list, feat_sets: dict) -> list:
    log.info(f"\n{SEP2}\nREGRESSION EXPERIMENTS (10 experiments, {CV_FOLDS}-fold CV)\n{SEP2}")

    results = []
    top20   = feat_sets.get("anova_top30", feature_names)[:20]

    # Features without discharge_disposition — leakage sensitivity test
    no_leak_feats = [f for f in feature_names if not f.startswith("discharge_disposition")]

    def pipe(reg):
        return Pipeline([("scaler", StandardScaler()), ("reg", reg)])

    # EXP01 — Linear Regression (baseline)
    log.info(f"\n  EXP01 — LinearRegression (baseline, all features)")
    results.append(evaluate_model(
        "EXP01_LinearRegression_all",
        pipe(LinearRegression()),
        X, y, feature_names,
    ))

    # EXP02 — Ridge α=1.0, all features
    log.info(f"\n  EXP02 — Ridge alpha=1.0, all features")
    results.append(evaluate_model(
        "EXP02_Ridge_a1_all",
        pipe(Ridge(alpha=1.0)),
        X, y, feature_names,
    ))

    # EXP03 — Ridge α=10.0, top-20 ANOVA features
    log.info(f"\n  EXP03 — Ridge alpha=10.0, top-20 ANOVA features")
    results.append(evaluate_model(
        "EXP03_Ridge_a10_top20",
        pipe(Ridge(alpha=10.0)),
        X, y, feature_names, feature_subset=top20,
    ))

    # EXP04 — Lasso α=0.01, all features
    log.info(f"\n  EXP04 — Lasso alpha=0.01, all features")
    results.append(evaluate_model(
        "EXP04_Lasso_a001_all",
        pipe(Lasso(alpha=0.01, max_iter=5000)),
        X, y, feature_names,
    ))

    # EXP05 — Decision Tree max_depth=5
    log.info(f"\n  EXP05 — DecisionTreeRegressor max_depth=5")
    results.append(evaluate_model(
        "EXP05_DecisionTree_depth5",
        pipe(DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)),
        X, y, feature_names,
    ))

    # EXP06 — Decision Tree max_depth=10, min_samples_leaf=20
    log.info(f"\n  EXP06 — DecisionTreeRegressor max_depth=10 min_leaf=20")
    results.append(evaluate_model(
        "EXP06_DecisionTree_depth10_min20",
        pipe(DecisionTreeRegressor(max_depth=10, min_samples_leaf=20,
                                   random_state=RANDOM_STATE)),
        X, y, feature_names,
    ))

    # EXP07 — Random Forest n=100
    log.info(f"\n  EXP07 — RandomForestRegressor n=100")
    results.append(evaluate_model(
        "EXP07_RandomForest_n100",
        pipe(RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE,
                                   n_jobs=-1)),
        X, y, feature_names,
    ))

    # EXP08 — XGBoost or GradientBoosting n=100
    if HAS_XGB:
        log.info(f"\n  EXP08 — XGBoostRegressor n=100")
        reg08    = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=RANDOM_STATE, verbosity=0)
        name08   = "EXP08_XGBoost_n100"
    else:
        log.info(f"\n  EXP08 — GradientBoostingRegressor n=100 (XGBoost not installed)")
        reg08    = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                             max_depth=5, subsample=0.8,
                                             random_state=RANDOM_STATE)
        name08   = "EXP08_GradientBoosting_n100"

    results.append(evaluate_model(name08, pipe(reg08), X, y, feature_names))

    # EXP09 — XGBoost without discharge_disposition features (leakage sensitivity)
    log.info(f"\n  EXP09 — XGBoost without discharge_disposition (leakage test, "
             f"{len(no_leak_feats)} features)")
    if HAS_XGB:
        reg09  = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RANDOM_STATE, verbosity=0)
        name09 = "EXP09_XGBoost_NoLeak"
    else:
        reg09  = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                           max_depth=5, subsample=0.8,
                                           random_state=RANDOM_STATE)
        name09 = "EXP09_GBR_NoLeak"
    results.append(evaluate_model(
        name09, pipe(reg09), X, y, feature_names, feature_subset=no_leak_feats,
    ))

    # EXP10 — Poisson Regression (statistically correct for count data)
    log.info(f"\n  EXP10 — PoissonRegressor (count data, all features)")
    results.append(evaluate_model(
        "EXP10_Poisson_all",
        pipe(PoissonRegressor(alpha=1.0, max_iter=1000)),
        X, y, feature_names,
    ))

    return results


# ── Save results CSV ───────────────────────────────────────────────────────────

def save_results(results: list) -> pd.DataFrame:
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results]).sort_values("mae")
    out = os.path.join(RESULTS_DIR, "regression_results.csv")
    df.to_csv(out, index=False)
    log.info(f"\n{SEP2}\nFINAL RESULTS (sorted by MAE ascending)\n{SEP2}")
    log.info(df.to_string(index=False))
    log.info(f"\n  [Saved] {out}")
    return df


# ── Experiment comparison plot ─────────────────────────────────────────────────

def plot_experiment_comparison(results: list) -> None:
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    palette = sns.color_palette("muted", len(df))

    for ax, (metric, err_col, label, ascending) in zip(axes, [
        ("mae",  "mae_std",  "MAE  — lower is better",  True),
        ("rmse", "rmse_std", "RMSE — lower is better",  True),
        ("r2",   "r2_std",   "R²   — higher is better", False),
    ]):
        ordered   = df.sort_values(metric, ascending=ascending)
        short     = [n.split("_", 1)[1] for n in ordered["experiment"]]
        vals      = ordered[metric].values
        errs      = ordered[err_col].values
        bars = ax.barh(short, vals, xerr=errs,
                       color=palette[:len(ordered)],
                       capsize=3, edgecolor="white")
        ax.set_xlabel(label)
        ax.set_title(label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + errs.max() * 0.05,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}", va="center", fontsize=8)

    plt.suptitle("Regression Experiment Comparison (5-fold CV)", fontsize=14)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "06_experiment_comparison.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  [Saved] {out}")


# ── Best model deep dive ───────────────────────────────────────────────────────

def _rebuild_pipeline(exp_name: str, feature_names: list,
                      feat_sets: dict) -> tuple[Pipeline, list]:
    """Re-create the winning pipeline and return (pipeline, feature_names_used)."""
    top20 = feat_sets.get("anova_top30", feature_names)[:20]

    if "top20" in exp_name:
        feats = [f for f in top20 if f in feature_names]
    elif "NoLeak" in exp_name:
        feats = [f for f in feature_names if not f.startswith("discharge_disposition")]
    else:
        feats = feature_names

    if "XGBoost" in exp_name and HAS_XGB:
        reg = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=RANDOM_STATE, verbosity=0)
    elif "GradientBoosting" in exp_name:
        reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                        max_depth=5, subsample=0.8,
                                        random_state=RANDOM_STATE)
    elif "RandomForest" in exp_name:
        reg = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE,
                                    n_jobs=-1)
    elif "depth10" in exp_name:
        reg = DecisionTreeRegressor(max_depth=10, min_samples_leaf=20,
                                    random_state=RANDOM_STATE)
    elif "depth5" in exp_name:
        reg = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)
    elif "Poisson" in exp_name:
        reg = PoissonRegressor(alpha=1.0, max_iter=1000)
    elif "Lasso" in exp_name:
        reg = Lasso(alpha=0.01, max_iter=5000)
    elif "Ridge_a10" in exp_name:
        reg = Ridge(alpha=10.0)
    elif "Ridge" in exp_name:
        reg = Ridge(alpha=1.0)
    else:
        reg = LinearRegression()

    return Pipeline([("scaler", StandardScaler()), ("reg", reg)]), feats


def best_model_deep_dive(results: list, X: np.ndarray, y: np.ndarray,
                         feature_names: list, feat_sets: dict) -> None:
    log.info(f"\n{SEP2}\nBEST MODEL — DEEP DIVE\n{SEP2}")

    best = min(results, key=lambda r: r["mae"])
    log.info(f"  Best experiment (by MAE): {best['experiment']}")

    pipe, feats = _rebuild_pipeline(best["experiment"], feature_names, feat_sets)
    idx   = [feature_names.index(f) for f in feats]
    X_use = X[:, idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_use, y, test_size=0.2, random_state=RANDOM_STATE,
    )
    pipe.fit(X_tr, y_tr)
    y_pred    = pipe.predict(X_te)
    residuals = y_te - y_pred

    mae_h  = mean_absolute_error(y_te, y_pred)
    rmse_h = np.sqrt(mean_squared_error(y_te, y_pred))
    r2_h   = r2_score(y_te, y_pred)

    log.info(f"\n  Hold-out test metrics:")
    log.info(f"    MAE  = {mae_h:.4f}")
    log.info(f"    RMSE = {rmse_h:.4f}")
    log.info(f"    R²   = {r2_h:.4f}")
    log.info(f"\n  Residual analysis:")
    log.info(f"    mean = {residuals.mean():.4f}  std = {residuals.std():.4f}")
    log.info(f"    |error| > 3 days: "
             f"{(np.abs(residuals) > 3).sum()} / {len(residuals)} "
             f"({(np.abs(residuals) > 3).mean() * 100:.1f}%)")

    # ── Figure 1: Actual vs Predicted | Residual Distribution | Residuals vs Fitted
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    ax = axes[0]
    ax.scatter(y_te, y_pred, alpha=0.15, s=5, color="#4C72B0", rasterized=True)
    lo, hi = 1, 14
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlim(lo, hi); ax.set_ylim(lo - 1, hi + 1)
    ax.set_xlabel("Actual time_in_hospital (days)")
    ax.set_ylabel("Predicted (days)")
    ax.set_title(f"Actual vs Predicted\n{best['experiment']}")
    ax.text(0.05, 0.92, f"MAE={mae_h:.3f}  RMSE={rmse_h:.3f}  R²={r2_h:.3f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(residuals, bins=50, color="#55A868", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red",    linestyle="--", lw=1.5, label="Zero error")
    ax.axvline(residuals.mean(), color="orange", linestyle="--", lw=1.5,
               label=f"Mean={residuals.mean():.2f}")
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.legend(fontsize=9)

    ax = axes[2]
    ax.scatter(y_pred, residuals, alpha=0.15, s=5, color="#DD8452", rasterized=True)
    ax.axhline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel("Predicted value (days)")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Fitted Values")

    plt.suptitle(f"Best Model Analysis — {best['experiment']}", fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "06_best_model_residuals.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  [Saved] {out}")

    # ── Figure 2: Actual vs Predicted distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.arange(0.5, 16.5, 1)
    ax.hist(y_te,   bins=bins, alpha=0.6, label="Actual",    color="#4C72B0",
            density=True, edgecolor="white")
    ax.hist(y_pred, bins=bins, alpha=0.6, label="Predicted", color="#DD8452",
            density=True, edgecolor="white")
    ax.set_xlabel("time_in_hospital (days)")
    ax.set_ylabel("Density")
    ax.set_title(f"Actual vs Predicted Distribution — {best['experiment']}")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "06_prediction_distribution.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  [Saved] {out}")

    # ── Figure 3: Feature importances (tree-based models only)
    reg = pipe.named_steps["reg"]
    if hasattr(reg, "feature_importances_"):
        imp = pd.Series(reg.feature_importances_, index=feats).sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(imp.index[::-1], imp.values[::-1], color="#4C72B0", edgecolor="white")
        ax.set_title(f"Top 20 Feature Importances — {best['experiment']}")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, "06_best_model_feature_importance.png")
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        log.info(f"  [Saved] {out}")


# ── Statistical significance (Wilcoxon on per-fold RMSE) ──────────────────────

def statistical_significance(results: list) -> None:
    log.info(f"\n{SEP2}\nSTATISTICAL SIGNIFICANCE — Best vs 2nd Best (Wilcoxon)\n{SEP2}")

    sorted_res = sorted(results, key=lambda r: r["mae"])
    best   = sorted_res[0]
    second = sorted_res[1]

    best_rmses   = np.array(best["_rmse_folds"])
    second_rmses = np.array(second["_rmse_folds"])

    log.info(f"  Best  : {best['experiment']}   MAE={best['mae']:.4f}")
    log.info(f"  Second: {second['experiment']} MAE={second['mae']:.4f}")
    log.info(f"\n  Fold RMSE comparison:")
    for i, (b, s) in enumerate(zip(best_rmses, second_rmses), 1):
        log.info(f"    Fold {i}: best={b:.4f}  second={s:.4f}  diff={b - s:+.4f}")
    log.info(f"\n  Mean RMSE: best={best_rmses.mean():.4f}  "
             f"second={second_rmses.mean():.4f}")

    if np.allclose(best_rmses, second_rmses):
        log.info("  Wilcoxon skipped — fold scores are identical")
        return

    stat, p = stats.wilcoxon(best_rmses, second_rmses)
    log.info(f"\n  Wilcoxon signed-rank test:")
    log.info(f"    statistic = {stat:.3f}  p = {p:.4f}")
    if p < 0.05:
        log.info(f"    → STATISTICALLY SIGNIFICANT (p < 0.05) — best model is superior")
    else:
        log.info(f"    → Not statistically significant (p ≥ 0.05) — models are equivalent")
        log.info(f"    → Note: n={CV_FOLDS} folds gives low statistical power; "
                 f"differences may be real but undetectable at this sample size")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    X, y, feature_names = load_data(PROCESSED_DATA_PATH)

    plot_target_distribution(y)

    feat_sets = feature_selection_report(X, y, feature_names)

    results = run_experiments(X, y, feature_names, feat_sets)

    save_results(results)

    plot_experiment_comparison(results)

    best_model_deep_dive(results, X, y, feature_names, feat_sets)

    statistical_significance(results)

    best = min(results, key=lambda r: r["mae"])
    log.info(f"\n{SEP}")
    log.info("REGRESSION ANALYSIS COMPLETE")
    log.info(f"  Best model : {best['experiment']}")
    log.info(f"  CV MAE     : {best['mae']:.4f} ± {best['mae_std']:.4f}")
    log.info(f"  CV RMSE    : {best['rmse']:.4f} ± {best['rmse_std']:.4f}")
    log.info(f"  CV R²      : {best['r2']:.4f} ± {best['r2_std']:.4f}")
    log.info(SEP)


if __name__ == "__main__":
    main()
