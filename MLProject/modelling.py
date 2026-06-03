import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (aman di server)
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

import mlflow
import mlflow.sklearn

# ── Argument Parser (untuk MLflow Project) ───────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--k_optimal",    type=int,   default=4)
parser.add_argument("--random_state", type=int,   default=42)
parser.add_argument("--data_path",    type=str,   default="data/Womens_Shoes_Clean.csv")
args = parser.parse_args()

# ── Konfigurasi MLflow ────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME     = "KMeans-Womens-Shoes"
K_OPTIMAL           = args.k_optimal
RANDOM_STATE        = args.random_state
DATA_PATH           = args.data_path

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("Membaca dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")


# ── 2. Preprocessing ──────────────────────────────────────────────────────────
print("Preprocessing...")
fitur_kmeans = ["prices.amountMin", "prices.amountMax", "price_range", "brand"]
X = df[fitur_kmeans].copy()
X["brand"] = X["brand"].str.strip().str.lower()

le = LabelEncoder()
X["brand_enc"] = le.fit_transform(X["brand"])

X_num = X[["prices.amountMin", "prices.amountMax", "price_range", "brand_enc"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)

n_brands   = X["brand"].nunique()
n_rows     = len(df)
n_missing  = df[fitur_kmeans[:-1]].isnull().sum().sum()


# ── 3. Elbow Curve (K 2-10) ───────────────────────────────────────────────────
print("Menghitung elbow curve...")
inertia     = []
sil_scores  = {}
k_range     = range(2, 11)

for k in k_range:
    km_tmp = KMeans(n_clusters=k, init="k-means++",
                    n_init=10, random_state=RANDOM_STATE)
    lbl_tmp = km_tmp.fit_predict(X_scaled)
    inertia.append(km_tmp.inertia_)
    sil_scores[k] = silhouette_score(X_scaled, lbl_tmp)

# Elbow plot
fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
ax_elbow.plot(list(k_range), inertia, "o-", color="#1D9E75", linewidth=2)
ax_elbow.axvline(x=K_OPTIMAL, color="#D85A30", linestyle="--",
                 label=f"K optimal = {K_OPTIMAL}")
ax_elbow.set_xlabel("Jumlah Cluster (K)")
ax_elbow.set_ylabel("Inertia (WCSS)")
ax_elbow.set_title("Elbow Method — Sepatu Wanita")
ax_elbow.legend()
plt.tight_layout()
elbow_path = "elbow_plot.png"
fig_elbow.savefig(elbow_path, dpi=150)
plt.close(fig_elbow)


# ── 4. KMeans Final (K=4) ────────────────────────────────────────────────────
print(f"Melatih KMeans K={K_OPTIMAL}...")
km = KMeans(n_clusters=K_OPTIMAL, init="k-means++",
            n_init=10, max_iter=300, random_state=RANDOM_STATE)
km.fit(X_scaled)
labels = km.labels_

df["cluster"] = labels
cluster_names = {0: "Mid-range Stabil", 1: "Premium",
                 2: "Promo Aktif",      3: "Budget"}
df["segment"] = df["cluster"].map(cluster_names)

sil_final   = silhouette_score(X_scaled, labels)
inertia_final = km.inertia_
n_iter      = km.n_iter_

sample_scores = silhouette_samples(X_scaled, labels)
per_cluster_sil = {
    f"silhouette_cluster_{c}": float(sample_scores[labels == c].mean())
    for c in range(K_OPTIMAL)
}

summary = df.groupby("cluster")[
    ["prices.amountMin", "prices.amountMax", "price_range"]
].mean().round(2)


# ── 5. PCA 2D Plot ────────────────────────────────────────────────────────────
print("Membuat plot PCA 2D...")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]
centroids_pca = pca.transform(km.cluster_centers_)
var1, var2 = pca.explained_variance_ratio_ * 100

COLORS = ["#1D9E75", "#7F77DD", "#BA7517", "#D85A30"]
LABELS = ["Mid-range Stabil", "Premium", "Promo Aktif", "Budget"]

fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
for c in range(K_OPTIMAL):
    mask = df["cluster"] == c
    ax_pca.scatter(df.loc[mask, "PC1"], df.loc[mask, "PC2"],
                   c=COLORS[c], label=LABELS[c],
                   alpha=0.4, s=10, edgecolors="none")
    ax_pca.scatter(*centroids_pca[c],
                   marker="*", s=250, c=COLORS[c],
                   edgecolors="white", linewidths=0.8, zorder=5)

ax_pca.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=11)
ax_pca.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=11)
ax_pca.set_title(f"K-Means (K={K_OPTIMAL}) — PCA 2D\n"
                 f"({var1+var2:.1f}% total variance explained)", fontsize=12)
ax_pca.legend(fontsize=10)
ax_pca.grid(True, alpha=0.15)
plt.tight_layout()
pca_path = "kmeans_pca.png"
fig_pca.savefig(pca_path, dpi=150, bbox_inches="tight")
plt.close(fig_pca)


# ── 6. CSV Hasil Clustering ───────────────────────────────────────────────────
csv_path = "hasil_clustering.csv"
df[["name", "brand", "prices.amountMin", "prices.amountMax",
    "price_range", "cluster", "segment"]].to_csv(csv_path, index=False)


# ── 7. MLflow Run ────────────────────────────────────────────────────────────
print("Logging ke MLflow...")
with mlflow.start_run(run_name=f"kmeans_k{K_OPTIMAL}") as run:

    # ── Tags
    mlflow.set_tags({
        "model_type"  : "KMeans",
        "dataset"     : "Womens_Shoes_Clean.csv",
        "framework"   : "sklearn",
        "author"      : "notebook_converted",
    })

    # ── Params
    mlflow.log_params({
        "k"                 : K_OPTIMAL,
        "init"              : "k-means++",
        "n_init"            : 10,
        "max_iter"          : 300,
        "random_state"      : RANDOM_STATE,
        "features"          : "amountMin,amountMax,price_range,brand_enc",
        "scaler"            : "StandardScaler",
        "n_rows"            : n_rows,
        "n_brands_unique"   : n_brands,
        "n_missing_values"  : int(n_missing),
    })

    # ── Metrics utama
    mlflow.log_metrics({
        "silhouette_score"  : round(sil_final, 6),
        "inertia"           : round(inertia_final, 2),
        "n_iter_converge"   : int(n_iter),
        "pca_var_pc1"       : round(float(var1), 2),
        "pca_var_pc2"       : round(float(var2), 2),
        "pca_var_total"     : round(float(var1 + var2), 2),
    })

    # ── Metrics per cluster
    mlflow.log_metrics(per_cluster_sil)

    # ── Silhouette untuk berbagai K (perbandingan)
    for k, s in sil_scores.items():
        mlflow.log_metric("sil_by_k", round(s, 6), step=k)

    # ── Cluster size & mean price
    for c in range(K_OPTIMAL):
        size = int((labels == c).sum())
        mean_min = float(summary.loc[c, "prices.amountMin"])
        mean_max = float(summary.loc[c, "prices.amountMax"])
        mlflow.log_metrics({
            f"cluster_{c}_size"        : size,
            f"cluster_{c}_mean_min_price": mean_min,
            f"cluster_{c}_mean_max_price": mean_max,
        })

    # ── Artifacts (plot & CSV)
    mlflow.log_artifact(elbow_path,  artifact_path="plots")
    mlflow.log_artifact(pca_path,    artifact_path="plots")
    mlflow.log_artifact(csv_path,    artifact_path="outputs")

    # ── Log model sklearn
    mlflow.sklearn.log_model(
        sk_model       = km,
        artifact_path  = "model",
        registered_model_name = "KMeans-Womens-Shoes",
        input_example  = X_scaled[:5],
    )

    run_id = run.info.run_id
    print(f"\nRun selesai!")
    print(f"   Run ID  : {run_id}")
    print(f"   UI      : {MLFLOW_TRACKING_URI}/#/experiments/")
    print(f"\nRingkasan metrics:")
    print(f"   Silhouette Score : {sil_final:.4f}")
    print(f"   Inertia          : {inertia_final:.2f}")
    print(f"   Iterasi konvergen: {n_iter}")
    print(f"   PCA variance     : {var1:.1f}% + {var2:.1f}% = {var1+var2:.1f}%")
    print(f"\nCluster distribution:")
    print(df["segment"].value_counts().to_string())