"""

KMeans performance-region clustering for inverse design.

The workflow compares Model I (inoue) and Model II (nlse) NF/gain performance spaces, performs

KMeans diagnostics in standardized target space, assigns a desired target to

a performance region, and extracts local inverse-design candidates from that

region in the original dB coordinates.

Generated PNG, MATLAB .mat, and CSV outputs are saved in separate subfolders.

The analysis and output filenames are otherwise unchanged.

Examples

\--------

Run with the default project layout:

    python src/kmeans_inverse_design.py

Choose the dataset used for clustering and a different desired target:

    python src/kmeans_inverse_design.py \\\\

        --clustering-dataset model-i \\

        --desired-nf 1.7 \\\\

        --desired-gain 9.0

"""

from __future__ import annotations

import argparse

import os

import sys

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy.io import savemat

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from sklearn.preprocessing import StandardScaler

from matplotlib.colors import BoundaryNorm

# =============================================================================

# Project paths

# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_I_DATA_PATH = (

    PROJECT_ROOT / "data" / "dataset_200000.csv"

)

DEFAULT_MODEL_II_DATA_PATH = (

    PROJECT_ROOT / "data" / "dataset_nlse_300000.csv"

)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clustering"

# =============================================================================

# Analysis configuration

# =============================================================================

INPUT_COLS = [

    "pump_signal_sep_nm",

    "pump_power_dBm",

    "beta2_ps2_per_km",

    "zero_disp_wavelength_nm",

]

TARGET_COLS = [

    "NF_dB",

    "max_gain_dB",

]

NF_COL = "NF_dB"

GAIN_COL = "max_gain_dB"

MODEL_I_DATA_PATH = str(DEFAULT_MODEL_I_DATA_PATH)

MODEL_II_DATA_PATH = str(DEFAULT_MODEL_II_DATA_PATH)

CLUSTERING_DATASET = "model-ii"

N_CLUSTERS = 3

RANDOM_SEED = 42

DESIRED_NF_DB = 1.7

DESIRED_GAIN_DB = 9.0

NUM_CLOSEST_SAMPLES = 10

N_LOCAL_CANDIDATES = 500

LOCAL_QUANTILE_LOW = 0.10

LOCAL_QUANTILE_HIGH = 0.90

CLUSTER_MIN = 2

CLUSTER_MAX = 5

CLUSTER_RANGE = range(CLUSTER_MIN, CLUSTER_MAX + 1)

SILHOUETTE_SAMPLE_SIZE = 20000

OUTPUT_FOLDER = str(DEFAULT_OUTPUT_DIR)
PNG_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "png")
MAT_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "mat")
CSV_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "csv")

SAVE_INDIVIDUAL_COMPARISON_PANELS = True

SAVE_INDIVIDUAL_INPUT_DISTRIBUTIONS = False

SAVE_INDIVIDUAL_PAIRWISE_PLOTS = False

GENERATED_PLOT_PAIRS: list[dict[str, str]] = []

# =============================================================================

# Command-line interface

# =============================================================================

def model_label(value: str) -> str:

    """Return the publication-facing model label."""

    return "Model I" if value == "model-i" else "Model II"

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(

        description=(

            "KMeans performance-region clustering and local inverse-design "

            "candidate selection in NF/gain space."

        )

    )

    parser.add_argument(

        "--model-i-data-path",

        dest="model_i_data_path",

        type=Path,

        default=DEFAULT_MODEL_I_DATA_PATH,

        help=f"Model I CSV path. Default: {DEFAULT_MODEL_I_DATA_PATH}",

    )

    parser.add_argument(

        "--model-ii-data-path",

        dest="model_ii_data_path",

        type=Path,

        default=DEFAULT_MODEL_II_DATA_PATH,

        help=f"Model II CSV path. Default: {DEFAULT_MODEL_II_DATA_PATH}",

    )

    parser.add_argument(

        "--clustering-dataset",

        choices=("model-i", "model-ii"),

        metavar="{model-i,model-ii}",

        default=CLUSTERING_DATASET,

        help="Dataset used for KMeans clustering. Default: Model II",

    )

    parser.add_argument(

        "--desired-nf",

        type=float,

        default=DESIRED_NF_DB,

        help=f"Desired NF in dB. Default: {DESIRED_NF_DB}",

    )

    parser.add_argument(

        "--desired-gain",

        type=float,

        default=DESIRED_GAIN_DB,

        help=f"Desired maximum gain in dB. Default: {DESIRED_GAIN_DB}",

    )

    parser.add_argument(

        "--n-clusters",

        type=int,

        default=N_CLUSTERS,

        help=f"Final number of KMeans regions. Default: {N_CLUSTERS}",

    )

    parser.add_argument(

        "--cluster-min",

        type=int,

        default=CLUSTER_MIN,

        help=f"Smallest k used for diagnostics. Default: {CLUSTER_MIN}",

    )

    parser.add_argument(

        "--cluster-max",

        type=int,

        default=CLUSTER_MAX,

        help=f"Largest k used for diagnostics. Default: {CLUSTER_MAX}",

    )

    parser.add_argument(

        "--local-candidates",

        type=int,

        default=N_LOCAL_CANDIDATES,

        help=(

            "Number of closest samples retained inside the recommended "

            f"region. Default: {N_LOCAL_CANDIDATES}"

        ),

    )

    parser.add_argument(

        "--closest-samples",

        type=int,

        default=NUM_CLOSEST_SAMPLES,

        help=(

            "Number of very closest existing samples saved separately. "

            f"Default: {NUM_CLOSEST_SAMPLES}"

        ),

    )

    parser.add_argument(

        "--quantile-low",

        type=float,

        default=LOCAL_QUANTILE_LOW,

        help=f"Lower robust-range quantile. Default: {LOCAL_QUANTILE_LOW}",

    )

    parser.add_argument(

        "--quantile-high",

        type=float,

        default=LOCAL_QUANTILE_HIGH,

        help=f"Upper robust-range quantile. Default: {LOCAL_QUANTILE_HIGH}",

    )

    parser.add_argument(

        "--silhouette-sample-size",

        type=int,

        default=SILHOUETTE_SAMPLE_SIZE,

        help=(

            "Number of samples used for silhouette scoring. Use 0 for all "

            f"samples. Default: {SILHOUETTE_SAMPLE_SIZE}"

        ),

    )

    parser.add_argument(

        "--output-dir",

        type=Path,

        default=DEFAULT_OUTPUT_DIR,

        help=f"Generated-results directory. Default: {DEFAULT_OUTPUT_DIR}",

    )

    parser.add_argument(

        "--save-individual-comparison-panels",

        action=argparse.BooleanOptionalAction,

        default=SAVE_INDIVIDUAL_COMPARISON_PANELS,

        help="Save the three Model I/Model II comparison panels separately.",

    )

    parser.add_argument(

        "--save-individual-input-distributions",

        action=argparse.BooleanOptionalAction,

        default=SAVE_INDIVIDUAL_INPUT_DISTRIBUTIONS,

        help="Save each local input distribution separately.",

    )

    parser.add_argument(

        "--save-individual-pairwise-plots",

        action=argparse.BooleanOptionalAction,

        default=SAVE_INDIVIDUAL_PAIRWISE_PLOTS,

        help="Save each unique pairwise input relationship separately.",

    )

    return parser.parse_args()

def validate_configuration() -> None:

    """Validate analysis settings before loading large datasets."""

    if N_CLUSTERS < 2:

        raise ValueError("n_clusters must be at least 2.")

    if CLUSTER_MIN < 2:

        raise ValueError("cluster_min must be at least 2.")

    if CLUSTER_MAX < CLUSTER_MIN:

        raise ValueError("cluster_max must be >= cluster_min.")

    if N_LOCAL_CANDIDATES < 1:

        raise ValueError("local_candidates must be at least 1.")

    if NUM_CLOSEST_SAMPLES < 1:

        raise ValueError("closest_samples must be at least 1.")

    if not 0.0 <= LOCAL_QUANTILE_LOW < LOCAL_QUANTILE_HIGH <= 1.0:

        raise ValueError(

            "Quantiles must satisfy 0 <= quantile_low < quantile_high <= 1."

        )

    if SILHOUETTE_SAMPLE_SIZE is not None and SILHOUETTE_SAMPLE_SIZE < 1:

        raise ValueError(

            "silhouette_sample_size must be positive or None."

        )

# =============================================================================

# Figure/data helpers

# =============================================================================

def save_figure_mat(png_path, **figure_data):

    """

    Save the numerical data used to create a Matplotlib figure into

    a MATLAB .mat file with the same base filename as the PNG.

    Example:

        figure.png -> figure.mat

    The .mat file stores numerical variables, not a native MATLAB .fig.

    """

    mat_filename = Path(png_path).stem + ".mat"
    mat_path = os.path.join(MAT_OUTPUT_FOLDER, mat_filename)

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

    GENERATED_PLOT_PAIRS.append({

        "png_file": os.path.abspath(png_path),

        "mat_file": os.path.abspath(mat_path)

    })

    print(f"MATLAB data saved to: {mat_path}")

    return mat_path

def load_and_clean_dataset(data_path, dataset_name):

    """

    Load one dataset, validate required columns, replace infinities,

    drop invalid rows, force gain to be non-negative, and add a source label.

    """

    if not os.path.exists(data_path):

        raise FileNotFoundError(

            f"{dataset_name} dataset not found: {data_path}"

        )

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

# =============================================================================

# Main analysis

# =============================================================================

def main() -> int:

    global MODEL_I_DATA_PATH

    global MODEL_II_DATA_PATH

    global CLUSTERING_DATASET

    global N_CLUSTERS

    global DESIRED_NF_DB

    global DESIRED_GAIN_DB

    global NUM_CLOSEST_SAMPLES

    global N_LOCAL_CANDIDATES

    global LOCAL_QUANTILE_LOW

    global LOCAL_QUANTILE_HIGH

    global CLUSTER_RANGE

    global SILHOUETTE_SAMPLE_SIZE

    global OUTPUT_FOLDER
    global PNG_OUTPUT_FOLDER
    global MAT_OUTPUT_FOLDER
    global CSV_OUTPUT_FOLDER

    global SAVE_INDIVIDUAL_COMPARISON_PANELS

    global SAVE_INDIVIDUAL_INPUT_DISTRIBUTIONS

    global SAVE_INDIVIDUAL_PAIRWISE_PLOTS

    args = parse_args()

    MODEL_I_DATA_PATH = str(args.model_i_data_path.expanduser().resolve())

    MODEL_II_DATA_PATH = str(args.model_ii_data_path.expanduser().resolve())

    CLUSTERING_DATASET = args.clustering_dataset

    N_CLUSTERS = args.n_clusters

    DESIRED_NF_DB = args.desired_nf

    DESIRED_GAIN_DB = args.desired_gain

    NUM_CLOSEST_SAMPLES = args.closest_samples

    N_LOCAL_CANDIDATES = args.local_candidates

    LOCAL_QUANTILE_LOW = args.quantile_low

    LOCAL_QUANTILE_HIGH = args.quantile_high

    CLUSTER_RANGE = range(args.cluster_min, args.cluster_max + 1)

    SILHOUETTE_SAMPLE_SIZE = (

        None

        if args.silhouette_sample_size == 0

        else args.silhouette_sample_size

    )

    OUTPUT_FOLDER = str(args.output_dir.expanduser().resolve())
    PNG_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "png")
    MAT_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "mat")
    CSV_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "csv")

    SAVE_INDIVIDUAL_COMPARISON_PANELS = (

        args.save_individual_comparison_panels

    )

    SAVE_INDIVIDUAL_INPUT_DISTRIBUTIONS = (

        args.save_individual_input_distributions

    )

    SAVE_INDIVIDUAL_PAIRWISE_PLOTS = (

        args.save_individual_pairwise_plots

    )

    try:

        validate_configuration()

    except ValueError as exc:

        print(f"ERROR: {exc}")

        return 1

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PNG_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(MAT_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(CSV_OUTPUT_FOLDER, exist_ok=True)

    GENERATED_PLOT_PAIRS.clear()

    print("--- Configuration ---")

    print(f"Model I data: {MODEL_I_DATA_PATH}")

    print(f"Model II data: {MODEL_II_DATA_PATH}")

    print(f"Clustering dataset: {model_label(CLUSTERING_DATASET)}")

    print(f"Desired NF: {DESIRED_NF_DB:.6f} dB")

    print(f"Desired gain: {DESIRED_GAIN_DB:.6f} dB")

    print(f"Final KMeans regions: {N_CLUSTERS}")

    print(f"Diagnostic k values: {list(CLUSTER_RANGE)}")

    print(f"Local candidates: {N_LOCAL_CANDIDATES}")

    print(

        "Robust local interval: "

        f"{100 * LOCAL_QUANTILE_LOW:.0f}-"

        f"{100 * LOCAL_QUANTILE_HIGH:.0f}%"

    )

    print(

        "Silhouette sample size: "

        f"{'all' if SILHOUETTE_SAMPLE_SIZE is None else SILHOUETTE_SAMPLE_SIZE}"

    )

    print(f"Output directory: {OUTPUT_FOLDER}")
    print(f"PNG files: {PNG_OUTPUT_FOLDER}")
    print(f"MAT files: {MAT_OUTPUT_FOLDER}")
    print(f"CSV files: {CSV_OUTPUT_FOLDER}")

    print("-" * 60)

    print("--- Loading Model I and Model II datasets ---")

    # Load both datasets

    try:

        df_model_i = load_and_clean_dataset(MODEL_I_DATA_PATH, "Model I")

        df_model_ii = load_and_clean_dataset(MODEL_II_DATA_PATH, "Model II")

    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:

        print(f"ERROR: {exc}")

        return 1

    # Select dataset used for clustering

    if CLUSTERING_DATASET == "model-i":

        df_model = df_model_i.copy()

    elif CLUSTERING_DATASET == "model-ii":

        df_model = df_model_ii.copy()

    else:

        raise ValueError("CLUSTERING_DATASET must be either Model I or Model II.")

    print(f"\nDataset selected for clustering: {model_label(CLUSTERING_DATASET)}")

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

    # 2C. Combined Model I / Model II scatter plot in NF/Gain space

    # ============================================================

    print("\n--- Plotting Model I and Model II in the same NF-Gain figure ---")

    plt.figure(figsize=(10, 7))

    plt.scatter(

        df_model_i[NF_COL],

        df_model_i[GAIN_COL],

        s=10,

        alpha=0.5,

        marker="o",

        edgecolors="none",

        label=f"Model I dataset, n={len(df_model_i):,}"

    )

    plt.scatter(

        df_model_ii[NF_COL],

        df_model_ii[GAIN_COL],

        s=10,

        alpha=0.5,

        marker="^",

        edgecolors="none",

        label=f"Model II dataset, n={len(df_model_ii):,}"

    )

    plt.scatter(

        DESIRED_NF_DB,

        DESIRED_GAIN_DB,

        s=250,

        marker="*",

        edgecolors="black",

        linewidths=1.2,

        label=(

            f"Desired target: NF={DESIRED_NF_DB:.2f} dB, "

            f"Gain={DESIRED_GAIN_DB:.2f} dB"

        ),

        zorder=10

    )

    plt.xlabel("NF (dB)")

    plt.ylabel("Maximum gain (dB)")

    plt.title("Comparison of Model I and Model II Datasets\nin NF-Gain Performance Space")

    plt.grid(True, linestyle=":", alpha=0.6)

    plt.legend(loc="best", frameon=True)

    plt.tight_layout()

    comparison_plot_path = os.path.join(

        PNG_OUTPUT_FOLDER,

        "model_i_vs_model_ii_NF_gain_comparison.png"

    )

    plt.savefig(comparison_plot_path, dpi=300, bbox_inches="tight")

    save_figure_mat(

        comparison_plot_path,

        model_i_NF_dB=df_model_i[NF_COL],

        model_i_max_gain_dB=df_model_i[GAIN_COL],

        model_ii_NF_dB=df_model_ii[NF_COL],

        model_ii_max_gain_dB=df_model_ii[GAIN_COL],

        desired_NF_dB=DESIRED_NF_DB,

        desired_max_gain_dB=DESIRED_GAIN_DB

    )

    plt.show()

    plt.close()

    print(f"Combined comparison figure saved to: {comparison_plot_path}")

    # ============================================================

    # 2D. Detailed three-panel Model I / Model II comparison

    # ============================================================

    print("\n--- Creating detailed Model I/Model II comparison figure ---")

    fig, axes = plt.subplots(

        nrows=1,

        ncols=3,

        figsize=(20, 6),

        sharex=True,

        sharey=True

    )

    # Panel 1: Model I

    axes[0].scatter(

        df_model_i[NF_COL],

        df_model_i[GAIN_COL],

        s=8,

        alpha=0.30,

        edgecolors="none"

    )

    axes[0].scatter(

        DESIRED_NF_DB,

        DESIRED_GAIN_DB,

        s=180,

        marker="*",

        edgecolors="black",

        zorder=10

    )

    axes[0].set_title(f"Model I Dataset\nn={len(df_model_i):,}")

    axes[0].set_xlabel("NF (dB)")

    axes[0].set_ylabel("Maximum gain (dB)")

    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Model II

    axes[1].scatter(

        df_model_ii[NF_COL],

        df_model_ii[GAIN_COL],

        s=8,

        alpha=0.30,

        edgecolors="none"

    )

    axes[1].scatter(

        DESIRED_NF_DB,

        DESIRED_GAIN_DB,

        s=180,

        marker="*",

        edgecolors="black",

        zorder=10

    )

    axes[1].set_title(f"Model II Dataset\nn={len(df_model_ii):,}")

    axes[1].set_xlabel("NF (dB)")

    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Overlay

    axes[2].scatter(

        df_model_i[NF_COL],

        df_model_i[GAIN_COL],

        s=8,

        alpha=0.25,

        marker="o",

        edgecolors="none",

        label="Model I"

    )

    axes[2].scatter(

        df_model_ii[NF_COL],

        df_model_ii[GAIN_COL],

        s=8,

        alpha=0.25,

        marker="^",

        edgecolors="none",

        label="Model II"

    )

    axes[2].scatter(

        DESIRED_NF_DB,

        DESIRED_GAIN_DB,

        s=180,

        marker="*",

        edgecolors="black",

        label="Desired target",

        zorder=10

    )

    axes[2].set_title("Model I and Model II Overlay")

    axes[2].set_xlabel("NF (dB)")

    axes[2].grid(True, linestyle=":", alpha=0.6)

    axes[2].legend()

    fig.suptitle("Comparison of Model I and Model II Performance Spaces", fontsize=16)

    plt.tight_layout()

    detailed_comparison_path = os.path.join(

        PNG_OUTPUT_FOLDER,

        "model_i_vs_model_ii_detailed_NF_gain_comparison.png"

    )

    plt.savefig(detailed_comparison_path, dpi=300, bbox_inches="tight")

    save_figure_mat(

        detailed_comparison_path,

        model_i_NF_dB=df_model_i[NF_COL],

        model_i_max_gain_dB=df_model_i[GAIN_COL],

        model_ii_NF_dB=df_model_ii[NF_COL],

        model_ii_max_gain_dB=df_model_ii[GAIN_COL],

        desired_NF_dB=DESIRED_NF_DB,

        desired_max_gain_dB=DESIRED_GAIN_DB

    )

    plt.show()

    plt.close()

    print(f"Detailed comparison figure saved to: {detailed_comparison_path}")

    # ------------------------------------------------------------

    # ALSO save the three comparison panels individually

    # ------------------------------------------------------------

    if SAVE_INDIVIDUAL_COMPARISON_PANELS:

        # Model I only

        fig_model_i, ax_model_i = plt.subplots(figsize=(8, 6))

        ax_model_i.scatter(

            df_model_i[NF_COL],

            df_model_i[GAIN_COL],

            s=8,

            alpha=0.30,

            edgecolors="none"

        )

        ax_model_i.scatter(

            DESIRED_NF_DB,

            DESIRED_GAIN_DB,

            s=180,

            marker="*",

            edgecolors="black",

            label="Desired target",

            zorder=10

        )

        ax_model_i.set_xlabel("NF (dB)")

        ax_model_i.set_ylabel("Maximum gain (dB)")

        ax_model_i.set_title(f"Model I Dataset, n={len(df_model_i):,}")

        ax_model_i.grid(True, linestyle=":", alpha=0.6)

        ax_model_i.legend()

        fig_model_i.tight_layout()

        model_i_only_path = os.path.join(

            PNG_OUTPUT_FOLDER,

            "model_i_NF_gain_performance_space.png"

        )

        fig_model_i.savefig(

            model_i_only_path,

            dpi=300,

            bbox_inches="tight"

        )

        save_figure_mat(

            model_i_only_path,

            model_i_NF_dB=df_model_i[NF_COL],

            model_i_max_gain_dB=df_model_i[GAIN_COL],

            desired_NF_dB=DESIRED_NF_DB,

            desired_max_gain_dB=DESIRED_GAIN_DB

        )

        plt.show()

        plt.close()

        # Model II only

        fig_model_ii, ax_model_ii = plt.subplots(figsize=(8, 6))

        ax_model_ii.scatter(

            df_model_ii[NF_COL],

            df_model_ii[GAIN_COL],

            s=8,

            alpha=0.30,

            edgecolors="none"

        )

        ax_model_ii.scatter(

            DESIRED_NF_DB,

            DESIRED_GAIN_DB,

            s=180,

            marker="*",

            edgecolors="black",

            label="Desired target",

            zorder=10

        )

        ax_model_ii.set_xlabel("NF (dB)")

        ax_model_ii.set_ylabel("Maximum gain (dB)")

        ax_model_ii.set_title(f"Model II Dataset, n={len(df_model_ii):,}")

        ax_model_ii.grid(True, linestyle=":", alpha=0.6)

        ax_model_ii.legend()

        fig_model_ii.tight_layout()

        model_ii_only_path = os.path.join(

            PNG_OUTPUT_FOLDER,

            "model_ii_NF_gain_performance_space.png"

        )

        fig_model_ii.savefig(

            model_ii_only_path,

            dpi=300,

            bbox_inches="tight"

        )

        save_figure_mat(

            model_ii_only_path,

            model_ii_NF_dB=df_model_ii[NF_COL],

            model_ii_max_gain_dB=df_model_ii[GAIN_COL],

            desired_NF_dB=DESIRED_NF_DB,

            desired_max_gain_dB=DESIRED_GAIN_DB

        )

        plt.show()

        plt.close()

        # Overlay only

        fig_overlay, ax_overlay = plt.subplots(figsize=(9, 7))

        ax_overlay.scatter(

            df_model_i[NF_COL],

            df_model_i[GAIN_COL],

            s=8,

            alpha=0.25,

            marker="o",

            edgecolors="none",

            label="Model I"

        )

        ax_overlay.scatter(

            df_model_ii[NF_COL],

            df_model_ii[GAIN_COL],

            s=8,

            alpha=0.25,

            marker="^",

            edgecolors="none",

            label="Model II"

        )

        ax_overlay.scatter(

            DESIRED_NF_DB,

            DESIRED_GAIN_DB,

            s=180,

            marker="*",

            edgecolors="black",

            label="Desired target",

            zorder=10

        )

        ax_overlay.set_xlabel("NF (dB)")

        ax_overlay.set_ylabel("Maximum gain (dB)")

        ax_overlay.set_title("Model I and Model II Overlay")

        ax_overlay.grid(True, linestyle=":", alpha=0.6)

        ax_overlay.legend()

        fig_overlay.tight_layout()

        overlay_only_path = os.path.join(

            PNG_OUTPUT_FOLDER,

            "model_i_vs_model_ii_overlay_only.png"

        )

        fig_overlay.savefig(

            overlay_only_path,

            dpi=300,

            bbox_inches="tight"

        )

        save_figure_mat(

            overlay_only_path,

            model_i_NF_dB=df_model_i[NF_COL],

            model_i_max_gain_dB=df_model_i[GAIN_COL],

            model_ii_NF_dB=df_model_ii[NF_COL],

            model_ii_max_gain_dB=df_model_ii[GAIN_COL],

            desired_NF_dB=DESIRED_NF_DB,

            desired_max_gain_dB=DESIRED_GAIN_DB

        )

        plt.show()

        plt.close()

    # Save combined dataset

    df_both_datasets = pd.concat([df_model_i, df_model_ii], axis=0, ignore_index=True)

    df_both_datasets.to_csv(

        os.path.join(CSV_OUTPUT_FOLDER, "combined_model_i_model_ii_dataset.csv"),

        index=False

    )

    # ============================================================

    # 3. Check if desired target is inside data range

    # ============================================================

    print("\n--- Target feasibility check ---")

    nf_min = df_model[NF_COL].min()

    nf_max = df_model[NF_COL].max()

    gain_min = df_model[GAIN_COL].min()

    gain_max = df_model[GAIN_COL].max()

    print(f"NF range   : {nf_min:.6f} to {nf_max:.6f}")

    print(f"Gain range : {gain_min:.6f} to {gain_max:.6f}")

    inside_nf = nf_min <= DESIRED_NF_DB <= nf_max

    inside_gain = gain_min <= DESIRED_GAIN_DB <= gain_max

    if inside_nf and inside_gain:

        print("Desired target is inside the available marginal data ranges.")

    else:

        print("WARNING: Desired target is outside the available marginal data ranges.")

        print("The code will still recommend the closest available KMeans region.")

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

    elbow_path = os.path.join(

        PNG_OUTPUT_FOLDER,

        "kmeans_elbow_method_performance_clusters.png"

    )

    plt.savefig(elbow_path, dpi=300)

    save_figure_mat(

        elbow_path,

        number_of_clusters=np.array(list(CLUSTER_RANGE)),

        inertia=np.array(inertias)

    )

    plt.show()

    plt.close()

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

    kmeans_silhouette_path = os.path.join(

        PNG_OUTPUT_FOLDER,

        "kmeans_silhouette_score_performance_clusters.png"

    )

    plt.savefig(kmeans_silhouette_path, dpi=300)

    save_figure_mat(

        kmeans_silhouette_path,

        number_of_clusters=np.array(list(CLUSTER_RANGE)),

        silhouette_score=np.array(kmeans_silhouette_scores),

        silhouette_sample_size=len(Y_silhouette)

    )

    plt.show()

    plt.close()

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

        os.path.join(CSV_OUTPUT_FOLDER, "kmeans_performance_region_centers.csv"),

        index=False

    )

    # ============================================================

    # 8. Save KMeans performance-region summary

    # ============================================================

    kmeans_region_summary = df_model.groupby("kmeans_performance_region")[

        INPUT_COLS + TARGET_COLS

    ].agg(["min", "mean", "max", "std"])

    kmeans_region_summary.to_csv(

        os.path.join(CSV_OUTPUT_FOLDER, "kmeans_performance_region_summary.csv")

    )

    # ============================================================

    # 9. Plot KMeans performance regions in NF/Gain space

    # ============================================================

    discrete_cmap = plt.get_cmap("viridis", N_CLUSTERS)
    boundaries = np.arange(-0.5, N_CLUSTERS + 0.5, 1)
    norm = BoundaryNorm(boundaries, N_CLUSTERS)
    

    print("\n--- Plotting KMeans performance regions ---")


    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(

        df_model[NF_COL],

        df_model[GAIN_COL],

        c=df_model["kmeans_performance_region"],

        cmap=discrete_cmap,
       
        norm=norm,

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

    cbar = plt.colorbar(
    scatter,
    ticks=np.arange(N_CLUSTERS),
    boundaries=boundaries)
    cbar.set_label("KMeans Performance Region")
    cbar.set_ticklabels([str(i) for i in range(N_CLUSTERS)])
    cbar.set_label("KMeans Performance Region")

    plt.tight_layout()

    kmeans_regions_plot_path = os.path.join(

        PNG_OUTPUT_FOLDER,

        "kmeans_performance_regions_NF_gain_space.png"

    )

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

    plt.close()

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

        os.path.join(

            CSV_OUTPUT_FOLDER,

            "kmeans_regions_sorted_by_closeness_to_target.csv"

        ),

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

        os.path.join(CSV_OUTPUT_FOLDER, "kmeans_samples_in_recommended_region.csv"),

        index=False

    )

    # ============================================================

    # 13. NEW: Local inverse-design candidates around desired target

    # ============================================================

    print("\n--- Selecting LOCAL inverse-design candidates ---")

    # Euclidean distance in the original NF/Gain dB coordinates,

    # matching the distance definition used in the paper:

    #   sqrt((NF-NF*)^2 + (Gain-Gain*)^2)

    kmeans_recommended_samples["distance_to_target"] = np.sqrt(

        (kmeans_recommended_samples[NF_COL] - DESIRED_NF_DB) ** 2

        +

        (kmeans_recommended_samples[GAIN_COL] - DESIRED_GAIN_DB) ** 2

    )

    # Rank local inverse-design candidates by the raw Euclidean distance.

    kmeans_recommended_samples = kmeans_recommended_samples.sort_values(

        "distance_to_target"

    ).reset_index(drop=True)

    n_local_actual = min(N_LOCAL_CANDIDATES, len(kmeans_recommended_samples))

    local_candidates = kmeans_recommended_samples.head(n_local_actual).copy()

    print(

        f"Using the {n_local_actual:,} closest samples inside KMeans region "

        f"{kmeans_recommended_region} as local inverse-design candidates."

    )

    print(

        "Distance is calculated as Euclidean distance in the original "

        "[NF_dB, max_gain_dB] coordinates."

    )

    local_candidates.to_csv(

        os.path.join(CSV_OUTPUT_FOLDER, "kmeans_local_inverse_design_candidates.csv"),

        index=False

    )

    # ============================================================

    # 14. NEW: Robust local input-parameter recommendations

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

        os.path.join(CSV_OUTPUT_FOLDER, "kmeans_local_input_parameter_summary.csv"),

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

        os.path.join(CSV_OUTPUT_FOLDER, "kmeans_recommended_region_target_summary.csv")

    )

    local_target_summary.to_csv(

        os.path.join(CSV_OUTPUT_FOLDER, "kmeans_local_candidates_target_summary.csv")

    )

    # ============================================================

    # 16. Closest existing KMeans samples

    # ============================================================

    print("\n--- Closest existing samples to desired target ---")

    kmeans_closest_samples = local_candidates.head(NUM_CLOSEST_SAMPLES).copy()

    kmeans_cols_to_show = INPUT_COLS + TARGET_COLS + [

        "kmeans_performance_region",

        "distance_to_target"

    ]

    print(kmeans_closest_samples[kmeans_cols_to_show])

    kmeans_closest_samples.to_csv(

        os.path.join(

            CSV_OUTPUT_FOLDER,

            "kmeans_closest_samples_inside_recommended_region.csv"

        ),

        index=False

    )

    # Save selected clustering dataset with KMeans label

    df_model.to_csv(

        os.path.join(

            CSV_OUTPUT_FOLDER,

            f"{CLUSTERING_DATASET.replace('-', '_')}_data_with_kmeans_performance_regions.csv"

        ),

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

        "Euclidean distance to desired target: "

        f"{best_kmeans_candidate['distance_to_target']:.6f}"

    )

    # ============================================================

    # 18. NEW/UPDATED: Plot target, region, local candidates, closest samples

    # ============================================================
    discrete_cmap = plt.get_cmap("viridis", N_CLUSTERS)
    boundaries = np.arange(-0.5, N_CLUSTERS + 0.5, 1)
    norm = BoundaryNorm(boundaries, N_CLUSTERS)


    print("\n--- Plotting desired target and LOCAL inverse-design candidates ---")

    plt.figure(figsize=(9, 7))

    # All KMeans regions in the background

    scatter = plt.scatter(

        df_model[NF_COL],

        df_model[GAIN_COL],

        c=df_model["kmeans_performance_region"],

        cmap=discrete_cmap,

        norm=norm,

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

    # local candidates highlighted separately

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

    cbar = plt.colorbar(
    scatter,
    ticks=np.arange(N_CLUSTERS),
    boundaries=boundaries)
    cbar.set_label("KMeans Performance Region")
    cbar.set_ticklabels([str(i) for i in range(N_CLUSTERS)])
    cbar.set_label("KMeans Performance Region")

    plt.tight_layout()

    kmeans_target_plot_path = os.path.join(

        PNG_OUTPUT_FOLDER,

        "kmeans_desired_target_local_inverse_candidates.png"

    )

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

        local_candidate_distance_euclidean=local_candidates["distance_to_target"],

        closest_NF_dB=kmeans_closest_samples[NF_COL],

        closest_max_gain_dB=kmeans_closest_samples[GAIN_COL],

        closest_distance_euclidean=kmeans_closest_samples["distance_to_target"],

        desired_NF_dB=DESIRED_NF_DB,

        desired_max_gain_dB=DESIRED_GAIN_DB

    )

    plt.show()

    plt.close()

    # ============================================================

    # 19. NEW: Better input-parameter distribution figure

    # ============================================================

    print("\n--- Plotting normalized LOCAL input-parameter distributions ---")

    # This replaces the old raw-count histograms of the entire KMeans region.

    # The new figure compares probability densities and highlights the robust

    # 10-90% local interval and local median for each input parameter.

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes = axes.ravel()

    for ax, col in zip(axes, INPUT_COLS):

        all_values = df_model[col].to_numpy()

        local_values = local_candidates[col].to_numpy()

        # Shared bin edges make the two densities directly comparable.

        if np.min(all_values) == np.max(all_values):

            bins = 30

        else:

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

    local_input_distribution_path = os.path.join(

        PNG_OUTPUT_FOLDER,

        "kmeans_local_input_parameter_distributions.png"

    )

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

        all_beta2_ps2_per_km=df_model["beta2_ps2_per_km"],

        local_beta2_ps2_per_km=local_candidates["beta2_ps2_per_km"],

        all_zero_disp_wavelength_nm=df_model["zero_disp_wavelength_nm"],

        local_zero_disp_wavelength_nm=local_candidates["zero_disp_wavelength_nm"]

    )

    plt.show()

    plt.close()

    # ------------------------------------------------------------

    # ALSO save each of the four input distributions individually

    # ------------------------------------------------------------

    if SAVE_INDIVIDUAL_INPUT_DISTRIBUTIONS:

        for col in INPUT_COLS:

            all_values = df_model[col].to_numpy()

            local_values = local_candidates[col].to_numpy()

            # Shared bins for a fair density comparison

            if np.min(all_values) == np.max(all_values):

                bins = 30

            else:

                bins = np.linspace(

                    np.min(all_values),

                    np.max(all_values),

                    41

                )

            median = np.median(local_values)

            q_low = np.quantile(local_values, LOCAL_QUANTILE_LOW)

            q_high = np.quantile(local_values, LOCAL_QUANTILE_HIGH)

            fig_single, ax_single = plt.subplots(figsize=(8, 5))

            ax_single.hist(

                all_values,

                bins=bins,

                density=True,

                alpha=0.30,

                edgecolor="black",

                label="All clustering data"

            )

            ax_single.hist(

                local_values,

                bins=bins,

                density=True,

                alpha=0.65,

                edgecolor="black",

                label=f"{n_local_actual:,} local candidates"

            )

            ax_single.axvline(

                median,

                linestyle="--",

                linewidth=2,

                label="Local median"

            )

            ax_single.axvspan(

                q_low,

                q_high,

                alpha=0.15,

                label=(

                    f"{100*LOCAL_QUANTILE_LOW:.0f}-"

                    f"{100*LOCAL_QUANTILE_HIGH:.0f}% local range"

                )

            )

            ax_single.set_xlabel(col)

            ax_single.set_ylabel("Probability density")

            ax_single.set_title(

                f"KMeans: {col}\n"

                f"Local inverse candidates near "

                f"NF={DESIRED_NF_DB:.2f} dB, "

                f"Gain={DESIRED_GAIN_DB:.2f} dB"

            )

            ax_single.grid(True, linestyle=":", alpha=0.6)

            ax_single.legend()

            fig_single.tight_layout()

            single_path = os.path.join(

                PNG_OUTPUT_FOLDER,

                f"kmeans_local_input_distribution_{col}.png"

            )

            fig_single.savefig(

                single_path,

                dpi=300,

                bbox_inches="tight"

            )

            save_figure_mat(

                single_path,

                input_parameter_name=np.array([col], dtype="U"),

                all_values=all_values,

                local_values=local_values,

                histogram_bin_edges=(

                    np.asarray(bins)

                    if not np.isscalar(bins)

                    else np.asarray([bins])

                ),

                local_median=median,

                local_quantile_low=q_low,

                local_quantile_high=q_high,

                desired_NF_dB=DESIRED_NF_DB,

                desired_max_gain_dB=DESIRED_GAIN_DB,

                local_candidate_count=n_local_actual

            )

            plt.show()

            plt.close()

    # ============================================================

    # 20. NEW: Pairwise input-parameter relationships for local candidates

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

    pairwise_input_path = os.path.join(

        PNG_OUTPUT_FOLDER,

        "kmeans_local_candidate_input_pairwise_matrix.png"

    )

    plt.savefig(pairwise_input_path, dpi=300, bbox_inches="tight")

    save_figure_mat(

        pairwise_input_path,

        pump_signal_sep_nm=local_candidates["pump_signal_sep_nm"],

        pump_power_dBm=local_candidates["pump_power_dBm"],

        beta2_ps2_per_km=local_candidates["beta2_ps2_per_km"],

        zero_disp_wavelength_nm=local_candidates["zero_disp_wavelength_nm"],

        local_candidate_NF_dB=local_candidates[NF_COL],

        local_candidate_max_gain_dB=local_candidates[GAIN_COL],

        local_candidate_distance_euclidean=local_candidates["distance_to_target"]

    )

    plt.show()

    plt.close()

    # ------------------------------------------------------------

    # ALSO save every unique pairwise relationship separately

    # ------------------------------------------------------------

    if SAVE_INDIVIDUAL_PAIRWISE_PLOTS:

        # Four input parameters -> 6 unique pairs

        for x_idx in range(len(INPUT_COLS)):

            for y_idx in range(x_idx + 1, len(INPUT_COLS)):

                x_col = INPUT_COLS[x_idx]

                y_col = INPUT_COLS[y_idx]

                fig_pair, ax_pair = plt.subplots(figsize=(7, 6))

                ax_pair.scatter(

                    local_candidates[x_col],

                    local_candidates[y_col],

                    s=14,

                    alpha=0.40,

                    edgecolors="none"

                )

                ax_pair.set_xlabel(x_col)

                ax_pair.set_ylabel(y_col)

                ax_pair.set_title(

                    "KMeans Local Inverse Candidates\n"

                    f"{y_col} vs {x_col}"

                )

                ax_pair.grid(True, linestyle=":", alpha=0.5)

                fig_pair.tight_layout()

                pair_path = os.path.join(

                    PNG_OUTPUT_FOLDER,

                    f"kmeans_local_pairwise_{y_col}_vs_{x_col}.png"

                )

                fig_pair.savefig(

                    pair_path,

                    dpi=300,

                    bbox_inches="tight"

                )

                save_figure_mat(

                    pair_path,

                    x_parameter_name=np.array([x_col], dtype="U"),

                    y_parameter_name=np.array([y_col], dtype="U"),

                    x_values=local_candidates[x_col],

                    y_values=local_candidates[y_col],

                    local_candidate_NF_dB=local_candidates[NF_COL],

                    local_candidate_max_gain_dB=local_candidates[GAIN_COL],

                    local_candidate_distance_euclidean=local_candidates[

                        "distance_to_target"

                    ],

                    desired_NF_dB=DESIRED_NF_DB,

                    desired_max_gain_dB=DESIRED_GAIN_DB

                )

                plt.show()

                plt.close()

    # ============================================================

    # 21. Final summary

    # ============================================================

    print("\n============================================================")

    print("KMeans inverse-design clustering completed.")

    print(f"Dataset used for clustering: {model_label(CLUSTERING_DATASET)}")

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

    # ============================================================

    # Verify and save a manifest of ALL PNG/MAT figure pairs

    # ============================================================

    plot_manifest = pd.DataFrame(GENERATED_PLOT_PAIRS).drop_duplicates()

    if not plot_manifest.empty:

        plot_manifest["png_exists"] = plot_manifest["png_file"].apply(

            os.path.exists

        )

        plot_manifest["mat_exists"] = plot_manifest["mat_file"].apply(

            os.path.exists

        )

        manifest_path = os.path.join(

            CSV_OUTPUT_FOLDER,

            "generated_plot_png_mat_manifest.csv"

        )

        plot_manifest.to_csv(

            manifest_path,

            index=False

        )

        n_pairs = len(plot_manifest)

        n_complete = int(

            (

                plot_manifest["png_exists"]

                &

                plot_manifest["mat_exists"]

            ).sum()

        )

        print(

            f"Verified PNG/MAT figure pairs: "

            f"{n_complete}/{n_pairs}"

        )

        print(

            f"Figure manifest saved to: "

            f"{manifest_path}"

        )

        if n_complete != n_pairs:

            print(

                "WARNING: At least one generated figure is missing "

                "either its PNG or MAT file. Check the manifest."

            )

    print(f"Results saved in folder: {OUTPUT_FOLDER}")
    print(f"PNG files: {PNG_OUTPUT_FOLDER}")
    print(f"MAT files: {MAT_OUTPUT_FOLDER}")
    print(f"CSV files: {CSV_OUTPUT_FOLDER}")

    print("============================================================")

    return 0

if __name__ == "__main__":

    sys.exit(main())
