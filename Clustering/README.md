# KMeans Performance-Region Clustering for Inverse Design

This code uses **KMeans clustering** to explore inverse design in the two-dimensional performance space defined by noise figure (`NF_dB`) and maximum gain (`max_gain_dB`). It clusters our model dataset to identify existing parameter combinations near a desired NF/gain target.

The method recommends **candidate input-parameter ranges based on nearby samples in the dataset**. 

## Workflow

1. Load and clean the dataset. Optionally, compare two different datasets (this section is commented out in the code; uncomment it if needed).
2. Use KMeans analysis.
3. Standardize `NF_dB` and `max_gain_dB` using `StandardScaler` fitted on the selected dataset.
4. Evaluate candidate cluster counts using the elbow method (inertia) and silhouette scores.
5. Fit the final KMeans model on the complete selected dataset and identify the region containing the desired target.
6. Rank the existing samples within that region by their Euclidean distance to the target in standardized NF/gain space.
7. Select the nearest local candidates, summarize their input parameters, and plot their distributions and pairwise relationships.
8. Save the figures and their underlying MATLAB `.mat` data, along with CSV summaries and candidate tables.

## Repository layout and input data

The code expects this directory layout:

```text
GithubProject/
├── data/
│   ├── dataset_300000.csv            
│   └── dataset_50000.csv       
└── Clustering/
    ├── README.md
    ├── src/
    │   └── kmeans_inverse_design.py
    └── outputs/                      

The input CSV files must contain:

| Input parameters | Target parameters |
| --- | --- |
| `pump_signal_sep_nm` | `NF_dB` |
| `pump_power_dBm` | `max_gain_dB` |
| `Dcc_ps` | |
| `zero_disp_wavelength_nm` | |

Rows containing missing or infinite values are removed, and negative `max_gain_dB` values are clipped to zero, as implemented in the script.

## Default configuration

Edit the configuration section near the beginning of `src/kmeans_inverse_design.py`:

| Setting | Default |
| --- | --- |
| `CLUSTERING_DATASET` | `"model-i"` |
| `N_CLUSTERS` | `2` |
| `CLUSTER_RANGE` | `range(2, 8)` (k = 2, 3, 4, 5, 6, 7) |
| `RANDOM_SEED` | `42` |
| `SILHOUETTE_SAMPLE_SIZE` | `30000` |
| `DESIRED_NF_DB` | `0.3` dB |
| `DESIRED_GAIN_DB` | `20.0` dB |
| `N_LOCAL_CANDIDATES` | `500` |
| `NUM_CLOSEST_SAMPLES` | `10` |
| `LOCAL_QUANTILE_LOW`, `LOCAL_QUANTILE_HIGH` | `0.10`, `0.90` |
| `SAVE_INDIVIDUAL_COMPARISON_PANELS` | `True` |

Set `CLUSTERING_DATASET = "model-i"` to cluster Model I. 
Set `SILHOUETTE_SAMPLE_SIZE = None` to score all samples; this may require substantially more time and memory.

**Silhouette diagnostics:** Each KMeans model is fitted on the complete selected dataset (`n_init=10`). Its silhouette score is evaluated on the same fixed random subset of up to 20,000 samples for all tested `k` values. The silhouette scores are therefore subset estimates, whereas inertia is computed from the full fit.

## Installation and execution

Install the required Python packages in your environment:

```bash
python -m pip install numpy pandas matplotlib scipy scikit-learn
```

From the `Clustering` directory, run:

```bash
python src/kmeans_inverse_design.py
```

## Local candidate selection

Let the desired target be \((NF^*, G^*)\). The code fits a `StandardScaler` to the selected dataset's two outputs and calculates the distance of each sample in the recommended KMeans region:

\[
d_i = \sqrt{\left(\frac{NF_i-NF^*}{\sigma_{NF}}\right)^2 + \left(\frac{G_i-G^*}{\sigma_G}\right)^2}
\]

Here, \(\sigma_{NF}\) and \(\sigma_G\) are the standard deviations used by the fitted scaler. The nearest **500** samples form the local candidate set. The **10** closest are also saved separately.


For each of the four input parameters, the local candidate summary includes the minimum, mean, median, maximum, standard deviation, and the 10th and 90th percentiles. The 10th–90th percentile interval describes the selected candidates;


## Generated outputs

The script automatically creates three subdirectories under `Clustering/outputs/`:

```text
Clustering/
└── outputs/
    ├── png/
    ├── mat/
    └── csv/
```

- **`png/`** contains Model I/Model II comparison plots, KMeans elbow and silhouette diagnostics, performance-region figures, the desired-target and local-candidate plot, the combined 2×2 input-distribution figure, and the pairwise input-parameter matrix.
- **`mat/`** contains numerical data corresponding to the PNG figures. Each `.mat` file has the same basename as its PNG counterpart; it is a data file, not a native MATLAB `.fig` file.
- **`csv/`** contains the combined Model I/Model II dataset, labeled clustering dataset, region centers and summaries, region distances, selected samples, local candidates, closest samples, and input/target summary tables.

Examples:

```text
Clustering/outputs/png/kmeans_performance_regions_NF_gain_space.png
Clustering/outputs/mat/kmeans_performance_regions_NF_gain_space.mat
Clustering/outputs/csv/kmeans_local_inverse_design_candidates.csv
Clustering/outputs/csv/kmeans_local_input_parameter_summary.csv
```


## Reproducibility

KMeans uses `random_state=42` and `n_init=10`. The fixed silhouette subset uses the same random seed. Cluster labels are identifiers without intrinsic ranking; the selected cluster is determined by applying the fitted KMeans model to the standardized desired target.
