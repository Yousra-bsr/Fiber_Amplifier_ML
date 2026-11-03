""" 
KMeans performance-region clustering and local inverse-design analysis.


Compare two datasets, cluster the selected dataset in
standardized [NF, gain] space, and select nearby inverse-design candidates using
standardized NF/gain distance.


# Main workflow:
#   1. Load two different datasets.
#   2. Compare the two NF/Gain performance spaces. Load and clean the dataset.
#      Optionally, compare two different datasets(this section is commented out in the code; uncomment it if needed).
#   3. Scale [NF_dB, max_gain_dB]
#   4. Run KMeans diagnostics: elbow + silhouette
#   5. Fit the final KMeans model
#   6. Assign the desired target to a KMeans region
#   7. Inside that region, rank samples by  standardized target distance
#   8. Keep a configurable number of local inverse-design candidates
#   9. Summarize and plot the input distributions of those local candidates
#  10. Save PNG figures, MATLAB .mat figure data, and CSV outputs
# ============================================================


"""


# ============================================================
# Imports
# ============================================================


import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import savemat


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score




# ============================================================
# 1. Configuration
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FOLDER = PROJECT_ROOT / "data"


MODEL_I_DATA_PATH = DATA_FOLDER / "dataset_300000.csv"  
MODEL_II_DATA_PATH = DATA_FOLDER / "dataset_300000.csv" #

# Dataset used for the KMeans inverse-design analysis below.
# Choose either "model-i" or "model-ii".
CLUSTERING_DATASET = "model-i"
MODEL_NAMES = {"model-i": "Model I", "model-ii": "Model II"}
MODEL_FILE_NAMES = {"model-i": "model_i", "model-ii": "model_ii"}




# Input parameters used for inverse-design recommendation
INPUT_COLS = [
    "pump_signal_sep_nm",
    "pump_power_dBm",
    "Dcc_ps",
    "zero_disp_wavelength_nm"
]


# Target/output parameters used for clustering
TARGET_COLS = [
    "NF_dB",
    "max_gain_dB"
]


NF_COL = "NF_dB"
GAIN_COL = "max_gain_dB"


# Final number of KMeans performance regions
N_CLUSTERS = 3


# Random seed for reproducibility
RANDOM_SEED = 42


# Desired inverse-design target
DESIRED_NF_DB = 0.3
DESIRED_GAIN_DB = 20.0


# Number of very closest existing samples printed/saved separately
NUM_CLOSEST_SAMPLES = 10


# These are selected ONLY inside the KMeans region containing the desired target,
# and are ranked by standardized NF/Gain distance to that target.
N_LOCAL_CANDIDATES = 500


LOCAL_QUANTILE_LOW = 0.10
LOCAL_QUANTILE_HIGH = 0.90

# Range of K values tested for diagnostics.
# range(2, 8) means k = 2, 3, 4, 5, 6, 7.
CLUSTER_RANGE = range(2, 8)

# Optional silhouette subsampling.
# Set to None to use the entire clustering dataset.
# For very large datasets, e.g. 300000 samples, 30000 is much faster.
SILHOUETTE_SAMPLE_SIZE = 30000


# Output folder
OUTPUT_FOLDER = Path(__file__).resolve().parents[1] / "outputs"
PNG_FOLDER = OUTPUT_FOLDER / "png"
MAT_FOLDER = OUTPUT_FOLDER / "mat"
CSV_FOLDER = OUTPUT_FOLDER / "csv"
for folder in (PNG_FOLDER, MAT_FOLDER, CSV_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# Figure-saving options
# ============================================================
# Main figures are saved as PNG with matching MATLAB data (.mat).
SAVE_INDIVIDUAL_COMPARISON_PANELS = True


# ============================================================
# Helper: save numerical data behind each figure to MATLAB
# ============================================================

def save_figure_mat(png_path, **figure_data):
    """
    Save the numerical data used to create a Matplotlib figure into
    a MATLAB .mat file with the same base filename as the PNG.


    The .mat file stores numerical variables, not a native MATLAB .fig.
    """


    mat_path = MAT_FOLDER / (Path(png_path).stem + ".mat")
    clean_data = {}


    for key, value in figure_data.items():
        if isinstance(value, (pd.Series, pd.Index)):
            value = value.to_numpy()
        elif isinstance(value, pd.DataFrame):
            value = value.to_numpy()
        elif isinstance(value, (list, tuple)):
            value = np.asarray(value)
        elif np.isscalar(value):
            value = np.asarray(value)


        clean_data[key] = value


    savemat(mat_path, clean_data, do_compression=True)

    print(f"MATLAB data saved to: {mat_path}")
    return mat_path


# ============================================================
# 2. Load and clean both datasets
# ============================================================

print("--- Loading Model I and Model II datasets ---")

def load_and_clean_dataset(data_path, dataset_name):
    """
    Load one dataset, validate required columns, handle infinities,
    drop invalid rows, force gain to be non-negative, and add a source label.
    """


    dataset = pd.read_csv(data_path)
    required_cols = INPUT_COLS + TARGET_COLS


    missing_cols = [col for col in required_cols if col not in dataset.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns in {dataset_name} dataset: {missing_cols}"
        )


    cleaned_dataset = dataset[required_cols].copy()
    cleaned_dataset = cleaned_dataset.replace([np.inf, -np.inf], np.nan)
    cleaned_dataset = cleaned_dataset.dropna().reset_index(drop=True)

    cleaned_dataset[GAIN_COL] = np.maximum(
        cleaned_dataset[GAIN_COL].to_numpy(),
        0.0
    )

    cleaned_dataset["dataset"] = dataset_name

    print(
        f"{dataset_name} dataset shape after cleaning: "
        f"{cleaned_dataset.shape}"
    )

    return cleaned_dataset


# Load both datasets
df_model_i = load_and_clean_dataset(MODEL_I_DATA_PATH, "Model I")
df_model_ii = load_and_clean_dataset(MODEL_II_DATA_PATH, "Model II")


# Select dataset used for clustering
if CLUSTERING_DATASET.lower() == "model-i":
    df_model = df_model_i.copy()
elif CLUSTERING_DATASET.lower() == "model-ii":
    df_model = df_model_ii.copy()
else:
    raise ValueError("CLUSTERING_DATASET must be either 'model-i' or 'model-ii'.")

print(f"\nDataset selected for clustering: {MODEL_NAMES[CLUSTERING_DATASET.lower()]}")
print(f"Clustering dataset shape: {df_model.shape}")


# ============================================================
# 2B. Print NF/Gain ranges for both datasets
# ============================================================
print("\n--- Model I and Model II NF/Gain ranges ---")

for dataset_name, dataset in [("Model I", df_model_i), ("Model II", df_model_ii)]:
    print(f"\n{dataset_name}:")
    print(
        f"  NF range   : {dataset[NF_COL].min():.6f} to "
        f"{dataset[NF_COL].max():.6f}"
    )
    print(
        f"  Gain range : {dataset[GAIN_COL].min():.6f} to "
        f"{dataset[GAIN_COL].max():.6f}"
    )

# ============================================================
# 2C. Combined Model I-Model II scatter plot in NF/Gain space
# ===========================================================
# print("\n--- Plotting Model I and Model II in the same NF-Gain figure ---")

# plt.figure(figsize=(10, 7))

# plt.scatter(
#     df_model_i[NF_COL],
#     df_model_i[GAIN_COL],
#     s=10,
#     alpha=0.5,
#     marker="o",
#     edgecolors="none",
#     label=f"Model I dataset, n={len(df_model_i):,}"
# )


# plt.scatter(
#     df_model_ii[NF_COL],
#     df_model_ii[GAIN_COL],
#     s=10,
#     alpha=0.5,
#     marker="^",
#     edgecolors="none",
#     label=f"Model II dataset, n={len(df_model_ii):,}"
# )


# plt.scatter(
#     DESIRED_NF_DB,
#     DESIRED_GAIN_DB,
#     s=250,
#     marker="*",
#     edgecolors="black",
#     linewidths=1.2,
#     label=(
#         f"Desired target: NF={DESIRED_NF_DB:.2f} dB, "
#         f"Gain={DESIRED_GAIN_DB:.2f} dB"
#     ),
#     zorder=10
# )


# plt.xlabel("NF (dB)")
# plt.ylabel("Maximum gain (dB)")
# plt.title("Comparison of Model I and Model II Datasets\nin NF-Gain Performance Space")
# plt.grid(True, linestyle=":", alpha=0.6)
# plt.legend(loc="best", frameon=True)
# plt.tight_layout()

# comparison_plot_path = PNG_FOLDER / "model_i_vs_model_ii_NF_gain_comparison.png"

# plt.savefig(comparison_plot_path, dpi=300, bbox_inches="tight")


# save_figure_mat(
#     comparison_plot_path,
#     model_i_NF_dB=df_model_i[NF_COL],
#     model_i_max_gain_dB=df_model_i[GAIN_COL],
#     model_ii_NF_dB=df_model_ii[NF_COL],
#     model_ii_max_gain_dB=df_model_ii[GAIN_COL],
#     desired_NF_dB=DESIRED_NF_DB,
#     desired_max_gain_dB=DESIRED_GAIN_DB
# )


# plt.show()
# print(f"Combined comparison figure saved to: {comparison_plot_path}")

# ============================================================
# 2D. Detailed three-panel Model I-Model II comparison
# ============================================================

# print("\n--- Creating detailed Model I-Model II comparison figure ---")


# fig, axes = plt.subplots(
#     nrows=1,
#     ncols=3,
#     figsize=(20, 6),
#     sharex=True,
#     sharey=True
# )

# # Panel 1: Model I
# axes[0].scatter(
#     df_model_i[NF_COL],
#     df_model_i[GAIN_COL],
#     s=8,
#     alpha=0.30,
#     edgecolors="none"
# )
# axes[0].scatter(
#     DESIRED_NF_DB,
#     DESIRED_GAIN_DB,
#     s=180,
#     marker="*",
#     edgecolors="black",
#     zorder=10
# )
# axes[0].set_title(f"Model I Dataset\nn={len(df_model_i):,}")
# axes[0].set_xlabel("NF (dB)")
# axes[0].set_ylabel("Maximum gain (dB)")
# axes[0].grid(True, linestyle=":", alpha=0.6)

# # Panel 2: Model II
# axes[1].scatter(
#     df_model_ii[NF_COL],
#     df_model_ii[GAIN_COL],
#     s=8,
#     alpha=0.30,
#     edgecolors="none"
# )
# axes[1].scatter(
#     DESIRED_NF_DB,
#     DESIRED_GAIN_DB,
#     s=180,
#     marker="*",
#     edgecolors="black",
#     zorder=10
# )
# axes[1].set_title(f"Model II Dataset\nn={len(df_model_ii):,}")
# axes[1].set_xlabel("NF (dB)")
# axes[1].grid(True, linestyle=":", alpha=0.6)


# # Panel 3: Overlay
# axes[2].scatter(
#     df_model_i[NF_COL],
#     df_model_i[GAIN_COL],
#     s=8,
#     alpha=0.25,
#     marker="o",
#     edgecolors="none",
#     label="Model I"
# )
# axes[2].scatter(
#     df_model_ii[NF_COL],
#     df_model_ii[GAIN_COL],
#     s=8,
#     alpha=0.25,
#     marker="^",
#     edgecolors="none",
#     label="Model II"
# )
# axes[2].scatter(
#     DESIRED_NF_DB,
#     DESIRED_GAIN_DB,
#     s=180,
#     marker="*",
#     edgecolors="black",
#     label="Desired target",
#     zorder=10
# )
# axes[2].set_title("Model I and Model II Overlay")
# axes[2].set_xlabel("NF (dB)")
# axes[2].grid(True, linestyle=":", alpha=0.6)
# axes[2].legend()


# fig.suptitle("Comparison of Model I and Model II Performance Spaces", fontsize=16)
# plt.tight_layout()


# detailed_comparison_path = PNG_FOLDER / "model_i_vs_model_ii_detailed_NF_gain_comparison.png"


# plt.savefig(detailed_comparison_path, dpi=300, bbox_inches="tight")


# save_figure_mat(
#     detailed_comparison_path,
#     model_i_NF_dB=df_model_i[NF_COL],
#     model_i_max_gain_dB=df_model_i[GAIN_COL],
#     model_ii_NF_dB=df_model_ii[NF_COL],
#     model_ii_max_gain_dB=df_model_ii[GAIN_COL],
#     desired_NF_dB=DESIRED_NF_DB,
#     desired_max_gain_dB=DESIRED_GAIN_DB
# )


# plt.show()
# print(f"Detailed comparison figure saved to: {detailed_comparison_path}")


# # ------------------------------------------------------------
# # ------------------------------------------------------------
# if SAVE_INDIVIDUAL_COMPARISON_PANELS:


#     # Model I only
#     fig_model_i, ax_model_i = plt.subplots(figsize=(8, 6))


#     ax_model_i.scatter(
#         df_model_i[NF_COL],
#         df_model_i[GAIN_COL],
#         s=8,
#         alpha=0.30,
#         edgecolors="none"
#     )


#     ax_model_i.scatter(
#         DESIRED_NF_DB,
#         DESIRED_GAIN_DB,
#         s=180,
#         marker="*",
#         edgecolors="black",
#         label="Desired target",
#         zorder=10
#     )


#     ax_model_i.set_xlabel("NF (dB)")
#     ax_model_i.set_ylabel("Maximum gain (dB)")
#     ax_model_i.set_title(f"Model I Dataset, n={len(df_model_i):,}")
#     ax_model_i.grid(True, linestyle=":", alpha=0.6)
#     ax_model_i.legend()
#     fig_model_i.tight_layout()


#     model_i_only_path = PNG_FOLDER / "model_i_NF_gain_performance_space.png"


#     fig_model_i.savefig(
#         model_i_only_path,
#         dpi=300,
#         bbox_inches="tight"
#     )


#     save_figure_mat(
#         model_i_only_path,
#         model_i_NF_dB=df_model_i[NF_COL],
#         model_i_max_gain_dB=df_model_i[GAIN_COL],
#         desired_NF_dB=DESIRED_NF_DB,
#         desired_max_gain_dB=DESIRED_GAIN_DB
#     )


#     plt.show()


#     # Model II only
#     fig_model_ii, ax_model_ii = plt.subplots(figsize=(8, 6))


#     ax_model_ii.scatter(
#         df_model_ii[NF_COL],
#         df_model_ii[GAIN_COL],
#         s=8,
#         alpha=0.30,
#         edgecolors="none"
#     )


#     ax_model_ii.scatter(
#         DESIRED_NF_DB,
#         DESIRED_GAIN_DB,
#         s=180,
#         marker="*",
#         edgecolors="black",
#         label="Desired target",
#         zorder=10
#     )


#     ax_model_ii.set_xlabel("NF (dB)")
#     ax_model_ii.set_ylabel("Maximum gain (dB)")
#     ax_model_ii.set_title(f"Model II Dataset, n={len(df_model_ii):,}")
#     ax_model_ii.grid(True, linestyle=":", alpha=0.6)
#     ax_model_ii.legend()
#     fig_model_ii.tight_layout()


#     model_ii_only_path = PNG_FOLDER / "model_ii_NF_gain_performance_space.png"


#     fig_model_ii.savefig(
#         model_ii_only_path,
#         dpi=300,
#         bbox_inches="tight"
#     )


#     save_figure_mat(
#         model_ii_only_path,
#         model_ii_NF_dB=df_model_ii[NF_COL],
#         model_ii_max_gain_dB=df_model_ii[GAIN_COL],
#         desired_NF_dB=DESIRED_NF_DB,
#         desired_max_gain_dB=DESIRED_GAIN_DB
#     )


#     plt.show()


#     # Overlay only
#     fig_overlay, ax_overlay = plt.subplots(figsize=(9, 7))


#     ax_overlay.scatter(
#         df_model_i[NF_COL],
#         df_model_i[GAIN_COL],
#         s=8,
#         alpha=0.25,
#         marker="o",
#         edgecolors="none",
#         label="Model I"
#     )


#     ax_overlay.scatter(
#         df_model_ii[NF_COL],
#         df_model_ii[GAIN_COL],
#         s=8,
#         alpha=0.25,
#         marker="^",
#         edgecolors="none",
#         label="Model II"
#     )


#     ax_overlay.scatter(
#         DESIRED_NF_DB,
#         DESIRED_GAIN_DB,
#         s=180,
#         marker="*",
#         edgecolors="black",
#         label="Desired target",
#         zorder=10
#     )


#     ax_overlay.set_xlabel("NF (dB)")
#     ax_overlay.set_ylabel("Maximum gain (dB)")
#     ax_overlay.set_title("Model I and Model II Overlay")
#     ax_overlay.grid(True, linestyle=":", alpha=0.6)
#     ax_overlay.legend()
#     fig_overlay.tight_layout()


#     overlay_only_path = PNG_FOLDER / "model_i_vs_model_ii_overlay_only.png"


#     fig_overlay.savefig(
#         overlay_only_path,
#         dpi=300,
#         bbox_inches="tight"
#     )


#     save_figure_mat(
#         overlay_only_path,
#         model_i_NF_dB=df_model_i[NF_COL],
#         model_i_max_gain_dB=df_model_i[GAIN_COL],
#         model_ii_NF_dB=df_model_ii[NF_COL],
#         model_ii_max_gain_dB=df_model_ii[GAIN_COL],
#         desired_NF_dB=DESIRED_NF_DB,
#         desired_max_gain_dB=DESIRED_GAIN_DB
#     )


#     plt.show()




# # Save combined dataset
# df_both_datasets = pd.concat([df_model_i, df_model_ii], axis=0, ignore_index=True)
# df_both_datasets.to_csv(
#     CSV_FOLDER / "combined_model_i_model_ii_dataset.csv",
#     index=False
# )

# # ============================================================
# # 3. Check if desired target is inside data range
# # ============================================================

# print("\n--- Target feasibility check ---")

# nf_min = df_model[NF_COL].min()
# nf_max = df_model[NF_COL].max()
# gain_min = df_model[GAIN_COL].min()
# gain_max = df_model[GAIN_COL].max()


# print(f"NF range   : {nf_min:.6f} to {nf_max:.6f}")
# print(f"Gain range : {gain_min:.6f} to {gain_max:.6f}")


# inside_nf = nf_min <= DESIRED_NF_DB <= nf_max
# inside_gain = gain_min <= DESIRED_GAIN_DB <= gain_max


# if inside_nf and inside_gain:
#     print("Desired target is inside both marginal data ranges (not a guarantee of feasibility).")
# else:
#     print("WARNING: Desired target is outside the available marginal data ranges.")
#     print("The code will still recommend the closest available KMeans region.")


# ============================================================
# 4. Scale NF/Gain target space
# ============================================================

print("\n--- Scaling NF/Gain space ---")


Y_targets = df_model[TARGET_COLS].to_numpy()
target_scaler = StandardScaler()
Y_targets_scaled = target_scaler.fit_transform(Y_targets)

# Desired target in the SAME standardized NF/Gain coordinates
desired_target = np.array([[DESIRED_NF_DB, DESIRED_GAIN_DB]])
desired_target_scaled = target_scaler.transform(desired_target)


# ============================================================
# 5. KMeans diagnostics: elbow and silhouette
# ============================================================

print("\n--- KMeans diagnostics ---")


inertias = []
kmeans_silhouette_scores = []


# Use one fixed random subset for silhouette so every k is compared
# with exactly the same sampled observations.
if (
    SILHOUETTE_SAMPLE_SIZE is not None
    and SILHOUETTE_SAMPLE_SIZE < len(Y_targets_scaled)
):
    rng = np.random.default_rng(RANDOM_SEED)
    silhouette_indices = rng.choice(
        len(Y_targets_scaled),
        size=SILHOUETTE_SAMPLE_SIZE,
        replace=False
    )
    Y_silhouette = Y_targets_scaled[silhouette_indices]
    print(
        f"Silhouette uses a fixed subset of {len(Y_silhouette):,} "
        f"out of {len(Y_targets_scaled):,} samples."
    )
else:
    silhouette_indices = np.arange(len(Y_targets_scaled))
    Y_silhouette = Y_targets_scaled
    print(f"Silhouette uses all {len(Y_targets_scaled):,} samples.")


for k in CLUSTER_RANGE:
    kmeans_temp = KMeans(
        n_clusters=k,
        random_state=RANDOM_SEED,
        n_init=10
    )


    # Fit KMeans using the full selected dataset
    labels_temp = kmeans_temp.fit_predict(Y_targets_scaled)
    inertia = kmeans_temp.inertia_


    # Silhouette only on the fixed subset
    labels_subset = labels_temp[silhouette_indices]
    sil_score = silhouette_score(Y_silhouette, labels_subset)


    inertias.append(inertia)
    kmeans_silhouette_scores.append(sil_score)


    print(
        f"k = {k}: inertia = {inertia:.4f}, "
        f"silhouette = {sil_score:.6f}"
    )


# ------------------------------------------------------------
# Plot 1: KMeans elbow method
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(list(CLUSTER_RANGE), inertias, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("KMeans Elbow Method for Performance-Region Clustering")
plt.grid(True, linestyle=":")
plt.tight_layout()


elbow_path = PNG_FOLDER / "kmeans_elbow_method_performance_clusters.png"
plt.savefig(elbow_path, dpi=300)


save_figure_mat(
    elbow_path,
    number_of_clusters=np.array(list(CLUSTER_RANGE)),
    inertia=np.array(inertias)
)


plt.show()

# ------------------------------------------------------------
# Plot 2: KMeans silhouette score
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(list(CLUSTER_RANGE), kmeans_silhouette_scores, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.title("KMeans Silhouette Score for Performance-Region Clustering")
plt.grid(True, linestyle=":")
plt.tight_layout()


kmeans_silhouette_path = PNG_FOLDER / "kmeans_silhouette_score_performance_clusters.png"
plt.savefig(kmeans_silhouette_path, dpi=300)


save_figure_mat(
    kmeans_silhouette_path,
    number_of_clusters=np.array(list(CLUSTER_RANGE)),
    silhouette_score=np.array(kmeans_silhouette_scores),
    silhouette_sample_size=len(Y_silhouette)
)


plt.show()


# ============================================================
# 6. Final KMeans clustering
# ============================================================

print("\n--- Final KMeans clustering ---")

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_SEED,
    n_init=10
)


df_model["kmeans_performance_region"] = kmeans.fit_predict(Y_targets_scaled)


print("\nKMeans performance region counts:")
print(df_model["kmeans_performance_region"].value_counts().sort_index())


# ============================================================
# 7. KMeans centers in original NF/Gain units
# ============================================================

centers_scaled = kmeans.cluster_centers_
centers_original = target_scaler.inverse_transform(centers_scaled)

df_kmeans_centers = pd.DataFrame(centers_original, columns=TARGET_COLS)
df_kmeans_centers["kmeans_performance_region"] = np.arange(N_CLUSTERS)


print("\nKMeans performance-region centers:")
print(df_kmeans_centers)

df_kmeans_centers.to_csv(
    CSV_FOLDER / "kmeans_performance_region_centers.csv",
    index=False
)


# ============================================================
# 8. Save KMeans performance-region summary
# ============================================================

kmeans_region_summary = df_model.groupby("kmeans_performance_region")[
    INPUT_COLS + TARGET_COLS
].agg(["min", "mean", "max", "std"])


kmeans_region_summary.to_csv(
    CSV_FOLDER / "kmeans_performance_region_summary.csv"
)


# ============================================================
# 9. Plot KMeans performance regions in NF/Gain space
# ============================================================

print("\n--- Plotting KMeans performance regions ---")

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    df_model[NF_COL],
    df_model[GAIN_COL],
    c=df_model["kmeans_performance_region"],
    s=10,
    alpha=0.45,
    edgecolors="none"
)


plt.scatter(
    df_kmeans_centers[NF_COL],
    df_kmeans_centers[GAIN_COL],
    s=180,
    marker="X",
    edgecolors="black",
    label="KMeans region centers"
)


plt.scatter(
    DESIRED_NF_DB,
    DESIRED_GAIN_DB,
    s=220,
    marker="*",
    edgecolors="black",
    label="Desired target"
)


plt.xlabel("NF_dB")
plt.ylabel("max_gain_dB")
plt.title("KMeans Performance Regions in NF-Gain Space")
plt.grid(True, linestyle=":")
plt.legend()


cbar = plt.colorbar(scatter)
cbar.set_label("KMeans Performance Region")


plt.tight_layout()


kmeans_regions_plot_path = PNG_FOLDER / "kmeans_performance_regions_NF_gain_space.png"
plt.savefig(kmeans_regions_plot_path, dpi=300)


save_figure_mat(
    kmeans_regions_plot_path,
    NF_dB=df_model[NF_COL],
    max_gain_dB=df_model[GAIN_COL],
    kmeans_performance_region=df_model["kmeans_performance_region"],
    kmeans_center_NF_dB=df_kmeans_centers[NF_COL],
    kmeans_center_max_gain_dB=df_kmeans_centers[GAIN_COL],
    desired_NF_dB=DESIRED_NF_DB,
    desired_max_gain_dB=DESIRED_GAIN_DB
)


plt.show()


# ============================================================
# 10. Assign desired target to KMeans performance region
# ============================================================

print("\n--- Finding recommended KMeans performance region ---")

kmeans_recommended_region = int(kmeans.predict(desired_target_scaled)[0])

print(f"Desired NF   = {DESIRED_NF_DB:.6f} dB")
print(f"Desired Gain = {DESIRED_GAIN_DB:.6f} dB")
print(f"Recommended KMeans performance region = {kmeans_recommended_region}")


# ============================================================
# 11. Distance from desired target to KMeans region centers
# ============================================================

# Distances are in standardized NF/Gain space.
distances_to_kmeans_centers = np.linalg.norm(
    centers_scaled - desired_target_scaled,
    axis=1
)


df_kmeans_region_distances = pd.DataFrame({
    "kmeans_performance_region": np.arange(N_CLUSTERS),
    "center_NF_dB": df_kmeans_centers[NF_COL].values,
    "center_max_gain_dB": df_kmeans_centers[GAIN_COL].values,
    "distance_to_desired_target_scaled": distances_to_kmeans_centers
}).sort_values("distance_to_desired_target_scaled")


print("\nKMeans regions sorted by closeness to desired target:")
print(df_kmeans_region_distances)


df_kmeans_region_distances.to_csv(
    CSV_FOLDER / "kmeans_regions_sorted_by_closeness_to_target.csv",
    index=False
)


# ============================================================
# 12. Extract samples from recommended KMeans region
# ============================================================

kmeans_recommended_samples = df_model[
    df_model["kmeans_performance_region"] == kmeans_recommended_region
].copy()


print(
    f"\nNumber of samples in recommended KMeans region: "
    f"{len(kmeans_recommended_samples):,}"
)


kmeans_recommended_samples.to_csv(
    CSV_FOLDER / "kmeans_samples_in_recommended_region.csv",
    index=False
)


# ============================================================
# 13. Local inverse-design candidates around desired target
# ============================================================

print("\n--- Selecting LOCAL inverse-design candidates ---")
#
# Here we calculate distance after StandardScaler, so NF and Gain are
# balanced according to the spread of the selected dataset.


recommended_targets_scaled = target_scaler.transform(
    kmeans_recommended_samples[TARGET_COLS].to_numpy()
)


kmeans_recommended_samples["distance_to_target_scaled"] = np.linalg.norm(
    recommended_targets_scaled - desired_target_scaled,
    axis=1
)


kmeans_recommended_samples["distance_to_target_original"] = np.sqrt(
    (kmeans_recommended_samples[NF_COL] - DESIRED_NF_DB) ** 2
    +
    (kmeans_recommended_samples[GAIN_COL] - DESIRED_GAIN_DB) ** 2
)


kmeans_recommended_samples = kmeans_recommended_samples.sort_values(
    "distance_to_target_scaled"
).reset_index(drop=True)


n_local_actual = min(N_LOCAL_CANDIDATES, len(kmeans_recommended_samples))
local_candidates = kmeans_recommended_samples.head(n_local_actual).copy()


print(
    f"Using the {n_local_actual:,} closest samples inside KMeans region "
    f"{kmeans_recommended_region} as local inverse-design candidates."
)


print(
    "Distance is calculated in standardized [NF, Gain] target space, "
    "not directly in raw dB coordinates."
)


local_candidates.to_csv(
    CSV_FOLDER / "kmeans_local_inverse_design_candidates.csv",
    index=False
)


# ============================================================
# 14. Robust local input-parameter recommendations
# ============================================================

print("\n--- Local input-parameter recommendations ---")

local_input_summary_rows = []


for col in INPUT_COLS:
    values = local_candidates[col]


    col_min = values.min()
    col_q_low = values.quantile(LOCAL_QUANTILE_LOW)
    col_mean = values.mean()
    col_median = values.median()
    col_q_high = values.quantile(LOCAL_QUANTILE_HIGH)
    col_max = values.max()
    col_std = values.std()


    local_input_summary_rows.append({
        "input_parameter": col,
        "min": col_min,
        f"q{int(100 * LOCAL_QUANTILE_LOW):02d}": col_q_low,
        "mean": col_mean,
        "median": col_median,
        f"q{int(100 * LOCAL_QUANTILE_HIGH):02d}": col_q_high,
        "max": col_max,
        "std": col_std
    })


    print(
        f"{col}: "
        f"median = {col_median:.6f}, "
        f"{100*LOCAL_QUANTILE_LOW:.0f}-{100*LOCAL_QUANTILE_HIGH:.0f}% range = "
        f"[{col_q_low:.6f}, {col_q_high:.6f}], "
        f"mean = {col_mean:.6f}, std = {col_std:.6f}"
    )


local_input_summary = pd.DataFrame(local_input_summary_rows)
local_input_summary.to_csv(
    CSV_FOLDER / "kmeans_local_input_parameter_summary.csv",
    index=False
)


# ============================================================
# 15. Target-space summaries: entire region vs local candidates
# ============================================================

print("\n--- NF/Gain summary: recommended region vs local candidates ---")


region_target_summary = kmeans_recommended_samples[TARGET_COLS].agg(
    ["min", "mean", "median", "max", "std"]
).T


local_target_summary = local_candidates[TARGET_COLS].agg(
    ["min", "mean", "median", "max", "std"]
).T


print("\nEntire recommended KMeans region:")
print(region_target_summary)


print("\nLocal inverse-design candidates:")
print(local_target_summary)


region_target_summary.to_csv(
    CSV_FOLDER / "kmeans_recommended_region_target_summary.csv"
)


local_target_summary.to_csv(
    CSV_FOLDER / "kmeans_local_candidates_target_summary.csv"
)


# ============================================================
# 16. Closest existing KMeans samples
# ============================================================

print("\n--- Closest existing samples to desired target ---")

kmeans_closest_samples = local_candidates.head(NUM_CLOSEST_SAMPLES).copy()

kmeans_cols_to_show = INPUT_COLS + TARGET_COLS + [
    "kmeans_performance_region",
    "distance_to_target_scaled",
    "distance_to_target_original"
]


print(kmeans_closest_samples[kmeans_cols_to_show])


kmeans_closest_samples.to_csv(
    CSV_FOLDER / "kmeans_closest_samples_inside_recommended_region.csv",
    index=False
)


# Save selected clustering dataset with KMeans label
df_model.to_csv(
    CSV_FOLDER / f"{MODEL_FILE_NAMES[CLUSTERING_DATASET.lower()]}_data_with_kmeans_performance_regions.csv",
    index=False
)


# ============================================================
# 17. Best existing KMeans candidate
# ============================================================

print("\n--- Best existing KMeans candidate ---")

best_kmeans_candidate = kmeans_closest_samples.iloc[0]

for col in INPUT_COLS:
    print(f"{col}: {best_kmeans_candidate[col]:.6f}")


print(f"{NF_COL}: {best_kmeans_candidate[NF_COL]:.6f}")
print(f"{GAIN_COL}: {best_kmeans_candidate[GAIN_COL]:.6f}")
print(
    "Standardized distance to desired target: "
    f"{best_kmeans_candidate['distance_to_target_scaled']:.6f}"
)
print(
    "Original-unit NF/Gain distance: "
    f"{best_kmeans_candidate['distance_to_target_original']:.6f}"
)


# ============================================================
# 18. Plot target, region, local candidates, closest samples
# ============================================================

print("\n--- Plotting desired target and LOCAL inverse-design candidates ---")


plt.figure(figsize=(9, 7))


# All KMeans regions in the background
scatter = plt.scatter(
    df_model[NF_COL],
    df_model[GAIN_COL],
    c=df_model["kmeans_performance_region"],
    s=8,
    alpha=0.15,
    edgecolors="none"
)


# Entire recommended region
plt.scatter(
    kmeans_recommended_samples[NF_COL],
    kmeans_recommended_samples[GAIN_COL],
    s=12,
    alpha=0.20,
    label=f"Entire KMeans region {kmeans_recommended_region}"
)


plt.scatter(
    local_candidates[NF_COL],
    local_candidates[GAIN_COL],
    s=25,
    alpha=0.65,
    label=f"{n_local_actual:,} local inverse candidates"
)


# Closest few samples
plt.scatter(
    kmeans_closest_samples[NF_COL],
    kmeans_closest_samples[GAIN_COL],
    s=80,
    marker="x",
    linewidths=2,
    label=f"{len(kmeans_closest_samples)} closest samples"
)


# Desired target
plt.scatter(
    DESIRED_NF_DB,
    DESIRED_GAIN_DB,
    s=260,
    marker="*",
    edgecolors="black",
    linewidths=1.2,
    label="Desired target",
    zorder=10
)


plt.xlabel("NF_dB")
plt.ylabel("max_gain_dB")
plt.title("Desired Target and Local KMeans Inverse-Design Candidates")
plt.grid(True, linestyle=":")
plt.legend(loc="best")


cbar = plt.colorbar(scatter)
cbar.set_label("KMeans Performance Region")


plt.tight_layout()


kmeans_target_plot_path = PNG_FOLDER / "kmeans_desired_target_local_inverse_candidates.png"


plt.savefig(kmeans_target_plot_path, dpi=300, bbox_inches="tight")


save_figure_mat(
    kmeans_target_plot_path,
    all_NF_dB=df_model[NF_COL],
    all_max_gain_dB=df_model[GAIN_COL],
    all_kmeans_region=df_model["kmeans_performance_region"],
    recommended_region_id=kmeans_recommended_region,
    recommended_region_NF_dB=kmeans_recommended_samples[NF_COL],
    recommended_region_max_gain_dB=kmeans_recommended_samples[GAIN_COL],
    local_candidate_NF_dB=local_candidates[NF_COL],
    local_candidate_max_gain_dB=local_candidates[GAIN_COL],
    local_candidate_distance_scaled=local_candidates["distance_to_target_scaled"],
    closest_NF_dB=kmeans_closest_samples[NF_COL],
    closest_max_gain_dB=kmeans_closest_samples[GAIN_COL],
    closest_distance_scaled=kmeans_closest_samples["distance_to_target_scaled"],
    desired_NF_dB=DESIRED_NF_DB,
    desired_max_gain_dB=DESIRED_GAIN_DB
)


plt.show()


# ============================================================
# 19. Input-parameter distribution figure
# ============================================================

print("\n--- Plotting normalized LOCAL input-parameter distributions ---")


# Compare the local candidates with the complete clustering dataset.
# The new figure compares probability densities and highlights the robust
# 10-90% local interval and local median for each input parameter.


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()


for ax, col in zip(axes, INPUT_COLS):
    all_values = df_model[col].to_numpy()
    local_values = local_candidates[col].to_numpy()


    # Shared bin edges make the two densities directly comparable.
    bins = np.linspace(np.min(all_values), np.max(all_values), 41)


    ax.hist(
        all_values,
        bins=bins,
        density=True,
        alpha=0.30,
        edgecolor="black",
        label="All clustering data"
    )


    ax.hist(
        local_values,
        bins=bins,
        density=True,
        alpha=0.65,
        edgecolor="black",
        label=f"{n_local_actual:,} local candidates"
    )


    median = np.median(local_values)
    q_low = np.quantile(local_values, LOCAL_QUANTILE_LOW)
    q_high = np.quantile(local_values, LOCAL_QUANTILE_HIGH)


    ax.axvline(
        median,
        linestyle="--",
        linewidth=2,
        label="Local median"
    )


    ax.axvspan(
        q_low,
        q_high,
        alpha=0.15,
        label=(
            f"{100*LOCAL_QUANTILE_LOW:.0f}-"
            f"{100*LOCAL_QUANTILE_HIGH:.0f}% local range"
        )
    )


    ax.set_xlabel(col)
    ax.set_ylabel("Probability density")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8)


fig.suptitle(
    "Input Parameters of Local Inverse-Design Candidates\n"
    f"Desired target: NF={DESIRED_NF_DB:.2f} dB, "
    f"Gain={DESIRED_GAIN_DB:.2f} dB",
    fontsize=15
)


plt.tight_layout()


local_input_distribution_path = PNG_FOLDER / "kmeans_local_input_parameter_distributions.png"


plt.savefig(
    local_input_distribution_path,
    dpi=300,
    bbox_inches="tight"
)


save_figure_mat(
    local_input_distribution_path,
    local_candidate_count=n_local_actual,
    quantile_low=LOCAL_QUANTILE_LOW,
    quantile_high=LOCAL_QUANTILE_HIGH,
    all_pump_signal_sep_nm=df_model["pump_signal_sep_nm"],
    local_pump_signal_sep_nm=local_candidates["pump_signal_sep_nm"],
    all_pump_power_dBm=df_model["pump_power_dBm"],
    local_pump_power_dBm=local_candidates["pump_power_dBm"],
    all_Dcc_ps=df_model["Dcc_ps"],
    local_Dcc_ps=local_candidates["Dcc_ps"],
    all_zero_disp_wavelength_nm=df_model["zero_disp_wavelength_nm"],
    local_zero_disp_wavelength_nm=local_candidates["zero_disp_wavelength_nm"]
)

plt.show()

# ============================================================
# 20. Pairwise input-parameter relationships for local candidates
# ============================================================

print("\n--- Plotting pairwise relationships among local input candidates ---")

# One-dimensional histograms cannot reveal correlations between input
# parameters. This 4x4 matrix shows the local candidate family:
# diagonal = local distributions; lower triangle = pairwise relationships.

n_inputs = len(INPUT_COLS)
fig, axes = plt.subplots(n_inputs, n_inputs, figsize=(14, 14))


for row in range(n_inputs):
    for col_idx in range(n_inputs):
        ax = axes[row, col_idx]
        y_col = INPUT_COLS[row]
        x_col = INPUT_COLS[col_idx]


        if row == col_idx:
            ax.hist(
                local_candidates[x_col],
                bins=30,
                density=True,
                alpha=0.70,
                edgecolor="black"
            )
            ax.grid(True, linestyle=":", alpha=0.4)


        elif row > col_idx:
            ax.scatter(
                local_candidates[x_col],
                local_candidates[y_col],
                s=10,
                alpha=0.35,
                edgecolors="none"
            )
            ax.grid(True, linestyle=":", alpha=0.4)


        else:
            # Keep upper triangle uncluttered.
            ax.axis("off")
            continue


        if row == n_inputs - 1:
            ax.set_xlabel(x_col, fontsize=9)
        else:
            ax.set_xticklabels([])


        if col_idx == 0:
            ax.set_ylabel(y_col, fontsize=9)
        else:
            ax.set_yticklabels([])


fig.suptitle(
    "Pairwise Input-Parameter Relationships of Local Inverse Candidates\n"
    f"NF={DESIRED_NF_DB:.2f} dB, Gain={DESIRED_GAIN_DB:.2f} dB",
    fontsize=15
)


plt.tight_layout()


pairwise_input_path = PNG_FOLDER / "kmeans_local_candidate_input_pairwise_matrix.png"


plt.savefig(pairwise_input_path, dpi=300, bbox_inches="tight")


save_figure_mat(
    pairwise_input_path,
    pump_signal_sep_nm=local_candidates["pump_signal_sep_nm"],
    pump_power_dBm=local_candidates["pump_power_dBm"],
    Dcc_ps=local_candidates["Dcc_ps"],
    zero_disp_wavelength_nm=local_candidates["zero_disp_wavelength_nm"],
    local_candidate_NF_dB=local_candidates[NF_COL],
    local_candidate_max_gain_dB=local_candidates[GAIN_COL],
    local_candidate_distance_scaled=local_candidates["distance_to_target_scaled"]
)

plt.show()

# ============================================================
# 21. Final summary
# ============================================================

print("\n============================================================")
print("KMeans inverse-design clustering completed.")
print(f"Dataset used for clustering: {MODEL_NAMES[CLUSTERING_DATASET.lower()]}")
print("Model I and Model II comparison plots were generated.")
print("------------------------------------------------------------")
print(f"Desired NF   : {DESIRED_NF_DB:.6f} dB")
print(f"Desired Gain : {DESIRED_GAIN_DB:.6f} dB")
print("------------------------------------------------------------")
print(f"KMeans recommended region : {kmeans_recommended_region}")
print(f"Samples in that region    : {len(kmeans_recommended_samples):,}")
print(f"Local inverse candidates  : {n_local_actual:,}")
print(
    f"Local robust interval     : "
    f"{100*LOCAL_QUANTILE_LOW:.0f}-"
    f"{100*LOCAL_QUANTILE_HIGH:.0f}% quantiles"
)
print("------------------------------------------------------------")


print(f"Results saved in folder: {OUTPUT_FOLDER}")
