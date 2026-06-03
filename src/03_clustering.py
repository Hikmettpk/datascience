"""
Step 3 — Clustering Analysis
K-Means, Hierarchical (Agglomerative), DBSCAN

Reads  : data_processed/diabetic_preprocessed.csv
Writes : outputs/figures/03_*.png
         outputs/results/clustering_*.csv
         logs/03_clustering_<timestamp>.log

Usage:
    python src/03_clustering.py
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    PROCESSED_DATA_PATH, FIGURES_DIR, RESULTS_DIR,
    get_logger, NUMERIC_COLS,
)

log = get_logger("03_clustering")
SEP = "=" * 80
SEP2 = "-" * 60

RANDOM_STATE = 42


# ── Data loading and scaling ───────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, np.ndarray, list]:
    log.info(SEP)
    log.info("CLUSTERING ANALYSIS — Diabetes 130-US Hospitals")
    log.info(SEP)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    log.info(f"Loaded: {df.shape[0]:,} rows  x  {df.shape[1]} columns")

    # Use numeric features + selected engineered features for clustering
    cluster_features = [c for c in NUMERIC_COLS if c in df.columns] + [
        "age_num", "n_active_meds", "total_visits",
        "change", "diabetesMed",
        "procedures_per_day", "meds_per_day",
    ]
    cluster_features = [c for c in cluster_features if c in df.columns]

    log.info(f"\nClustering features ({len(cluster_features)}): {cluster_features}")

    X = df[cluster_features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    log.info(f"Features standardised (StandardScaler)")
    return df, X_scaled, cluster_features


def pca_reduction(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    log.info(f"\nPCA: {X.shape[1]} → {X_pca.shape[1]} components  "
             f"(explains {pca.explained_variance_ratio_.sum()*100:.1f}% variance)")

    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca2.fit_transform(X)
    return X_pca, X_2d


# ── K-Means ────────────────────────────────────────────────────────────────────

def kmeans_analysis(X: np.ndarray, X_2d: np.ndarray, df: pd.DataFrame) -> None:
    log.info(f"\n{SEP2}")
    log.info("K-MEANS CLUSTERING")
    log.info(SEP2)

    k_range = range(2, 12)
    inertias, sil_scores, db_scores, ch_scores = [], [], [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels, sample_size=5000))
        db_scores.append(davies_bouldin_score(X, labels))
        ch_scores.append(calinski_harabasz_score(X, labels))
        log.info(f"  k={k:2d}  inertia={km.inertia_:>12.1f}  "
                 f"silhouette={sil_scores[-1]:.4f}  "
                 f"davies-bouldin={db_scores[-1]:.4f}  "
                 f"calinski={ch_scores[-1]:.1f}")

    # Elbow plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(k_range, inertias, "bo-")
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")

    axes[1].plot(k_range, sil_scores, "go-")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Score")

    axes[2].plot(k_range, db_scores, "ro-")
    axes[2].set_title("Davies-Bouldin Score (lower=better)")
    axes[2].set_xlabel("k"); axes[2].set_ylabel("Score")

    axes[3].plot(k_range, ch_scores, "mo-")
    axes[3].set_title("Calinski-Harabasz Score")
    axes[3].set_xlabel("k"); axes[3].set_ylabel("Score")

    plt.suptitle("K-Means Cluster Evaluation Metrics", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_kmeans_elbow.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # Choose best k by silhouette
    best_k = list(k_range)[sil_scores.index(max(sil_scores))]
    log.info(f"\n  Best k by silhouette: {best_k}  (score={max(sil_scores):.4f})")

    # Fit best model
    km_best = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    labels = km_best.fit_predict(X)

    # 2D scatter
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels,
                         cmap="tab10", alpha=0.4, s=5)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(f"K-Means (k={best_k}) — PCA 2D Projection")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_kmeans_scatter.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # Cluster profiles
    df_cluster = df.copy()
    df_cluster["kmeans_label"] = labels

    profile_cols = [c for c in NUMERIC_COLS if c in df.columns] + [
        "age_num", "n_active_meds", "early_readmit",
    ]
    profile_cols = [c for c in profile_cols if c in df_cluster.columns]
    profile = df_cluster.groupby("kmeans_label")[profile_cols].mean().round(3)
    log.info(f"\n  K-Means cluster profiles (k={best_k}):")
    log.info(profile.to_string())
    profile.to_csv(os.path.join(RESULTS_DIR, "kmeans_cluster_profiles.csv"))

    # Readmission rate per cluster
    readmit_rate = df_cluster.groupby("kmeans_label")["early_readmit"].mean() * 100
    log.info(f"\n  Early readmission rate per cluster (%):")
    log.info(readmit_rate.round(2).to_string())

    # Heatmap of cluster profiles
    fig, ax = plt.subplots(figsize=(14, max(4, best_k)))
    norm_profile = (profile - profile.mean()) / profile.std()
    sns.heatmap(norm_profile, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title(f"K-Means Cluster Profiles (Normalised)  k={best_k}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_kmeans_profiles_heatmap.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    log.info(f"\n  [Saved] K-Means figures → outputs/figures/")


# ── Hierarchical Clustering ────────────────────────────────────────────────────
#
# Naive AgglomerativeClustering requires an O(n²) distance matrix.
# For n=70K that is ~20 GB — not feasible on any consumer machine.
#
# Solution: k-NN connectivity graph (official sklearn recommendation for large n).
#   • kneighbors_graph(k=10) builds a sparse adjacency matrix: ~8 MB total
#   • AgglomerativeClustering(connectivity=...) only merges neighboring clusters
#   • The FULL 69,987-row dataset is used — zero rows are removed
#   • Reference: Pedregosa et al. (2011), sklearn User Guide §2.3.6
#
# Dendrogram is purely a visualisation artefact. scipy.linkage still needs a
# dense matrix, so a 1000-row subsample is used for the plot ONLY.
# All metric computation and cluster assignments use all rows.

KNN_K = 10   # neighbours for the sparse connectivity graph


def hierarchical_analysis(X: np.ndarray, X_2d: np.ndarray, df: pd.DataFrame) -> None:
    log.info(f"\n{SEP2}")
    log.info("HIERARCHICAL (AGGLOMERATIVE) CLUSTERING — full dataset via k-NN connectivity")
    log.info(SEP2)
    log.info(f"  Dataset : {len(X):,} rows  (ALL rows used)")
    log.info(f"  Method  : Ward / complete / average linkage + k-NN graph (k={KNN_K})")
    log.info(f"  Memory  : O(n·k) sparse matrix — no full distance matrix needed")

    # ── 1. Build k-NN connectivity graph once ─────────────────────────────────
    log.info(f"\n  Building k-NN graph ...")
    conn = kneighbors_graph(X, n_neighbors=KNN_K, include_self=False, n_jobs=-1)
    conn = 0.5 * (conn + conn.T)          # symmetrise (required by Ward)
    log.info(f"  Done. Non-zero entries: {conn.nnz:,}  "
             f"({conn.nnz * 8 / 1024**2:.1f} MB)")

    # ── 2. Dendrogram — subsample for visualisation only ─────────────────────
    dend_n   = 1000
    dend_idx = np.random.RandomState(RANDOM_STATE).choice(len(X), dend_n, replace=False)
    Z        = linkage(X[dend_idx], method="ward")

    fig, ax = plt.subplots(figsize=(18, 6))
    dendrogram(Z, ax=ax, no_labels=True, truncate_mode="lastp", p=50,
               leaf_font_size=8, color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title(
        f"Hierarchical Clustering Dendrogram — Ward linkage\n"
        f"(n={dend_n} subsample for visualisation only; "
        f"clustering performed on all {len(X):,} rows)"
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_hierarchical_dendrogram.png"),
                dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  Dendrogram saved (visualisation only, n={dend_n})")

    # ── 3. Evaluate linkages × k on FULL data ─────────────────────────────────
    linkages = ["ward", "complete", "average"]
    k_range  = range(2, 9)
    results  = []

    for link in linkages:
        for k in k_range:
            agg    = AgglomerativeClustering(n_clusters=k, linkage=link,
                                             connectivity=conn)
            labels  = agg.fit_predict(X)
            uniq    = np.unique(labels)
            counts  = np.bincount(labels.astype(int))

            # Skip degenerate results: fewer than 2 real clusters
            # OR a cluster so small the 5000-point sample misses it
            if len(uniq) < 2 or counts.min() < 10:
                log.info(f"  linkage={link:<10} k={k}  SKIP — "
                         f"degenerate ({len(uniq)} cluster(s), min size={counts.min()})")
                continue

            try:
                sil = silhouette_score(X, labels, sample_size=5000,
                                       random_state=RANDOM_STATE)
                db  = davies_bouldin_score(X, labels)
                ch  = calinski_harabasz_score(X, labels)
            except ValueError as e:
                log.info(f"  linkage={link:<10} k={k}  SKIP — {e}")
                continue

            results.append(dict(linkage=link, k=k,
                                silhouette=round(sil, 4),
                                davies_bouldin=round(db, 4),
                                calinski_harabasz=round(ch, 1)))
            log.info(f"  linkage={link:<10} k={k}  "
                     f"sil={sil:.4f}  db={db:.4f}  ch={ch:.1f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(RESULTS_DIR, "hierarchical_scores.csv"), index=False)

    best_row  = res_df.loc[res_df["silhouette"].idxmax()]
    best_link = best_row["linkage"]
    best_k    = int(best_row["k"])
    log.info(f"\n  Best: linkage={best_link}, k={best_k}, "
             f"silhouette={best_row['silhouette']:.4f}")

    # ── 4. Final model on full data ────────────────────────────────────────────
    agg_best = AgglomerativeClustering(n_clusters=best_k, linkage=best_link,
                                       connectivity=conn)
    labels   = agg_best.fit_predict(X)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter  = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels,
                          cmap="tab10", alpha=0.4, s=4)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(
        f"Hierarchical Clustering — linkage={best_link}, k={best_k}\n"
        f"PCA 2D  |  full dataset n={len(X):,}"
    )
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_hierarchical_scatter.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    # ── 5. Cluster profiles & readmission rates ────────────────────────────────
    df_hier = df.copy()
    df_hier["hier_label"] = labels

    profile_cols = [c for c in NUMERIC_COLS if c in df.columns] + [
        "age_num", "n_active_meds", "early_readmit",
    ]
    profile_cols = [c for c in profile_cols if c in df_hier.columns]
    profile      = df_hier.groupby("hier_label")[profile_cols].mean().round(3)
    log.info(f"\n  Cluster profiles (full dataset, n={len(X):,}):")
    log.info(profile.to_string())
    profile.to_csv(os.path.join(RESULTS_DIR, "hierarchical_cluster_profiles.csv"))

    readmit_rate = df_hier.groupby("hier_label")["early_readmit"].mean() * 100
    log.info(f"\n  Early readmission rate per cluster (%):")
    log.info(readmit_rate.round(2).to_string())

    # Profile heatmap
    disp_cols = [c for c in profile_cols if c != "early_readmit"]
    norm_p    = (profile[disp_cols] - profile[disp_cols].mean()) / profile[disp_cols].std()
    fig, ax   = plt.subplots(figsize=(14, max(4, best_k)))
    sns.heatmap(norm_p, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title(
        f"Hierarchical Cluster Profiles (Normalised)\n"
        f"linkage={best_link}  k={best_k}  |  n={len(X):,} (full dataset)"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_hierarchical_profiles_heatmap.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    log.info(f"  [Saved] Hierarchical figures → outputs/figures/")


# ── DBSCAN ─────────────────────────────────────────────────────────────────────

def dbscan_analysis(X: np.ndarray, X_2d: np.ndarray, df: pd.DataFrame) -> None:
    log.info(f"\n{SEP2}")
    log.info("DBSCAN CLUSTERING")
    log.info(SEP2)

    # Use PCA-reduced data for DBSCAN efficiency
    pca_db = PCA(n_components=10, random_state=RANDOM_STATE)
    X_db = pca_db.fit_transform(X)

    # Epsilon selection via k-NN distance plot
    from sklearn.neighbors import NearestNeighbors
    k = 5
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_db)
    distances, _ = nn.kneighbors(X_db)
    kth_dist = np.sort(distances[:, k - 1])[::-1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(kth_dist, color="#4C72B0")
    ax.set_title(f"DBSCAN: {k}-NN Distance Plot (sorted desc) for ε selection")
    ax.set_xlabel("Points (sorted)")
    ax.set_ylabel(f"{k}-NN Distance")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_dbscan_eps_selection.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    # Grid search over eps and min_samples
    eps_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    min_samples_values = [5, 10, 20, 50]
    results = []

    for eps in eps_values:
        for ms in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1)
            labels = db.fit_predict(X_db)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            noise_pct = n_noise / len(labels) * 100

            if n_clusters >= 2 and noise_pct < 50:
                sil = silhouette_score(X_db, labels, sample_size=5000)
            else:
                sil = np.nan

            results.append(dict(eps=eps, min_samples=ms, n_clusters=n_clusters,
                                n_noise=n_noise, noise_pct=round(noise_pct, 1), silhouette=sil))
            sil_str = f"{sil:.4f}" if not np.isnan(sil) else "N/A"
            log.info(f"  eps={eps:.1f}  min_samples={ms:3d}  "
                     f"clusters={n_clusters:3d}  noise={noise_pct:5.1f}%  sil={sil_str}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(RESULTS_DIR, "dbscan_scores.csv"), index=False)

    # Best valid configuration
    valid = res_df.dropna(subset=["silhouette"])
    if len(valid) == 0:
        log.info("  No valid DBSCAN configuration found. Trying defaults.")
        best_eps, best_ms = 1.0, 10
    else:
        best_row = valid.loc[valid["silhouette"].idxmax()]
        best_eps = best_row["eps"]
        best_ms = int(best_row["min_samples"])
        log.info(f"\n  Best: eps={best_eps}, min_samples={best_ms}, "
                 f"sil={best_row['silhouette']:.4f}, "
                 f"clusters={int(best_row['n_clusters'])}")

    db_best = DBSCAN(eps=best_eps, min_samples=best_ms, n_jobs=-1)
    labels = db_best.fit_predict(X_db)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    log.info(f"\n  Final DBSCAN: {n_clusters} clusters, {n_noise:,} noise points "
             f"({n_noise/len(labels)*100:.1f}%)")

    # Scatter plot
    mask_valid = labels != -1
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(X_2d[~mask_valid, 0], X_2d[~mask_valid, 1],
                    c="lightgray", alpha=0.3, s=3, label="Noise")
    scatter = axes[0].scatter(X_2d[mask_valid, 0], X_2d[mask_valid, 1],
                               c=labels[mask_valid], cmap="tab10", alpha=0.5, s=5)
    axes[0].set_title(f"DBSCAN Clusters (eps={best_eps}, min_s={best_ms})")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].legend()

    # Cluster sizes
    unique_labels, counts = np.unique(labels[mask_valid], return_counts=True)
    axes[1].bar([str(l) for l in unique_labels], counts, color="#4C72B0", edgecolor="white")
    axes[1].set_title("DBSCAN Cluster Sizes (excl. noise)")
    axes[1].set_xlabel("Cluster ID")
    axes[1].set_ylabel("Size")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_dbscan_scatter.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    # Readmission in clusters vs noise
    df_temp = df.copy()
    df_temp["dbscan_label"] = labels
    if "early_readmit" in df_temp.columns:
        readmit = df_temp.groupby("dbscan_label")["early_readmit"].agg(["mean", "count"])
        readmit["mean_pct"] = readmit["mean"] * 100
        log.info(f"\n  Early readmission rate by DBSCAN label (-1=noise):")
        log.info(readmit.round(2).to_string())

    log.info(f"  [Saved] DBSCAN figures → outputs/figures/")


# ── Comparison ────────────────────────────────────────────────────────────────

def clustering_comparison(X: np.ndarray) -> None:
    log.info(f"\n{SEP2}")
    log.info("CLUSTERING METHOD COMPARISON SUMMARY")
    log.info(SEP2)

    methods = {
        "KMeans_k4": KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10),
        "KMeans_k6": KMeans(n_clusters=6, random_state=RANDOM_STATE, n_init=10),
        "Hierarchical_ward_k4": AgglomerativeClustering(n_clusters=4, linkage="ward"),
        "Hierarchical_complete_k4": AgglomerativeClustering(n_clusters=4, linkage="complete"),
    }

    rows = []
    for name, model in methods.items():
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=5000)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        rows.append(dict(method=name, silhouette=round(sil, 4),
                         davies_bouldin=round(db, 4), calinski_harabasz=round(ch, 1)))
        log.info(f"  {name:<35} sil={sil:.4f}  db={db:.4f}  ch={ch:.1f}")

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(RESULTS_DIR, "clustering_comparison.csv"), index=False)
    log.info(f"\n  [Saved] outputs/results/clustering_comparison.csv")


def main():
    df, X_scaled, features = load_data()
    X_pca, X_2d = pca_reduction(X_scaled)

    kmeans_analysis(X_scaled, X_2d, df)
    hierarchical_analysis(X_scaled, X_2d, df)
    dbscan_analysis(X_scaled, X_2d, df)
    clustering_comparison(X_scaled)

    log.info(f"\n{SEP}")
    log.info("CLUSTERING COMPLETE")
    log.info(SEP)


if __name__ == "__main__":
    main()
