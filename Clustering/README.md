# KMeans Performance-Region Clustering for Inverse Design

This repository implements a KMeans-based inverse-design workflow in the
two-dimensional performance space defined by `NF_dB` and `max_gain_dB`.

The analysis compares the performance spaces of **Model I** and **Model II**,
identifies the KMeans performance region associated with a desired NF/gain
target, and then selects nearby samples **inside that region** to infer
physically relevant input-parameter distributions and robust ranges.

## Workflow

The script:

1. loads and cleans the Model I and Model II datasets;
2. compares their NF/gain performance spaces;
3. selects either Model I or Model II for clustering;
4. standardizes `[NF_dB, max_gain_dB]` for KMeans clustering;
5. evaluates KMeans cluster counts using inertia and silhouette score;
6. fits the final KMeans model;
7. assigns the desired target to a KMeans performance region;
8. ranks samples inside that region in the original NF/gain coordinates;
9. retains a configurable local inverse-design candidate set;
10. summarizes robust input-parameter ranges;
11. creates performance-space, distribution, and pairwise plots;
12. saves CSV, PNG, and MATLAB `.mat` outputs in separate folders.

## Euclidean distance for local candidate selection

The local inverse-design candidates are ranked using Euclidean distance in the
original NF/gain coordinates:

\[
d =
\sqrt{
(NF-NF^*)^2 +
(G_{\max}-G_{\max}^*)^2
}
\]

where:

- `NF` is the sample noise figure;
- `NF*` is the desired noise figure;
- `G_max` is the sample maximum gain;
- `G_max*` is the desired maximum gain.

This distance definition is applied directly in the original dB coordinates and
matches the inverse-design distance used in the associated analysis.


## Data

The repository is intended to include both project-owned CSV datasets:

```text
data/
├── dataset_200000.csv
└── dataset_nlse_300000.csv
```

The exact filenames can be specified through the command-line options.

Required columns are:

### Inputs

```text
pump_signal_sep_nm
pump_power_dBm
beta2_ps2_per_km
zero_disp_wavelength_nm
```

### Targets

```text
NF_dB
max_gain_dB
```

## Default configuration

| Parameter | Default |
| --- | ---: |
| Dataset used for clustering | Model II |
| Final KMeans regions | 3 |
| Random seed | 42 |
| Desired NF | 1.7 dB |
| Desired maximum gain | 9.0 dB |
| Diagnostic k values | 2, 3, 4, 5 |
| Silhouette sample size | 20,000 |
| Local inverse candidates | 500 |
| Closest samples saved separately | 10 |
| Robust local interval | 10th–90th percentile |

For silhouette diagnostics, one fixed random subset is reused for each tested
value of `k`, while KMeans itself is fitted using the complete selected
clustering dataset.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   ├── dataset_200000.csv
│   └── dataset_nlse_300000.csv
└── src/
    └── kmeans_inverse_design.py
```

Generated files are written under:

```text
outputs/
└── clustering/
    ├── png/
    ├── mat/
    └── csv/
```

The output folders contain:

```text
outputs/clustering/png/
    PNG figures

outputs/clustering/mat/
    MATLAB .mat files containing the numerical data used for the figures

outputs/clustering/csv/
    CSV tables, summaries, candidate sets, and the PNG/MAT manifest
```

The complete `outputs/` directory is ignored by Git because these results can
be regenerated from the shared source code and data.

## Installation

Python 3.10 or newer is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the default analysis

```bash
python src/kmeans_inverse_design.py
```

Both Model I and Model II datasets are loaded and compared.

By default, **Model II** is used for KMeans clustering.

## Choose which dataset is clustered

To cluster Model I:

```bash
python src/kmeans_inverse_design.py \
    --clustering-dataset model-i
```

To cluster Model II:

```bash
python src/kmeans_inverse_design.py \
    --clustering-dataset model-ii
```

## Change the desired inverse-design target

```bash
python src/kmeans_inverse_design.py \
    --desired-nf 1.7 \
    --desired-gain 9.0
```

## Specify the dataset paths

```bash
python src/kmeans_inverse_design.py \
    --model-i-data-path data/dataset_model_i.csv \
    --model-ii-data-path data/dataset_model_ii.csv
```

## Command-line options

Important options include:

```text
--model-i-data-path PATH
--model-ii-data-path PATH
--clustering-dataset {model-i,model-ii}
--desired-nf VALUE
--desired-gain VALUE
--n-clusters N
--cluster-min N
--cluster-max N
--local-candidates N
--closest-samples N
--quantile-low VALUE
--quantile-high VALUE
--silhouette-sample-size N
--output-dir PATH
```

Use:

```bash
python src/kmeans_inverse_design.py --help
```

to see every option.

Set:

```bash
--silhouette-sample-size 0
```

to use the entire selected clustering dataset for silhouette scoring.

## Main outputs

### CSV outputs

CSV files are written to:

```text
outputs/clustering/csv/
```

They include:

- the combined Model I / Model II dataset;
- KMeans performance-region centers;
- KMeans region-level summaries;
- distances from KMeans region centers to the desired target;
- samples in the recommended KMeans region;
- local inverse-design candidates;
- robust local input-parameter summaries;
- target-space summaries;
- closest existing samples;
- the selected clustering dataset with KMeans labels;
- the generated PNG/MAT manifest.

### PNG outputs

Figures are written to:

```text
outputs/clustering/png/
```

They include:

- Model I / Model II NF-gain comparisons;
- elbow diagnostics;
- silhouette-score diagnostics;
- KMeans performance regions;
- desired-target and local-candidate visualization;
- local input-parameter distributions;
- pairwise input-parameter relationships.

### MATLAB outputs

MATLAB `.mat` files are written to:

```text
outputs/clustering/mat/
```

For registered figures, the corresponding `.mat` file contains the numerical
data used to generate the PNG figure.

For example:

```text
outputs/clustering/png/kmeans_performance_regions_NF_gain_space.png
outputs/clustering/mat/kmeans_performance_regions_NF_gain_space.mat
```

The script also creates a CSV manifest that checks whether each generated
PNG/MAT pair exists.

## Local inverse-design

After the desired target is assigned to a KMeans performance region, all samples
inside that region are ranked using Euclidean distance in the original
`[NF_dB, max_gain_dB]` coordinates:

\[
d =
\sqrt{
(NF-NF^*)^2 +
(G_{\max}-G_{\max}^*)^2
}
\]

The nearest samples form the local inverse-design candidate family.

For each input parameter, the code reports:

- minimum;
- lower quantile;
- mean;
- median;
- upper quantile;
- maximum;
- standard deviation.

By default, the robust recommendation interval is the **10th to 90th
percentile** of the local inverse-design candidates.

The input parameters analyzed are:

```text
pump_signal_sep_nm
pump_power_dBm
beta2_ps2_per_km
zero_disp_wavelength_nm
```

## KMeans clustering

KMeans clustering is performed in standardized NF/gain space.

The selected dataset is transformed using `StandardScaler` before KMeans is
fitted. This prevents the relative numerical spread of NF and maximum gain from
controlling the clustering geometry.

The cluster centers are subsequently transformed back to the original NF/gain
units for interpretation and reporting.

This standardization applies to the **KMeans clustering stage only**.

The local inverse-design candidate ranking uses the original-unit Euclidean
distance described above.

## Reproducibility

The default random seed is:

```text
42
```

KMeans uses:

```text
n_init = 10
```

For silhouette diagnostics, the same fixed random subset is used for every
tested number of clusters so that the silhouette scores can be compared
consistently.

## Version-control policy

The repository is configured to share:

- source code;
- documentation;
- Model I data;
- Model II data.

Generated outputs are excluded through `.gitignore`, including:

```text
outputs/
```

This prevents generated PNG, MAT, and CSV result files from being committed
while keeping the source datasets available in the repository.

Python caches, virtual environments, operating-system files, and editor-specific
files are also excluded.


## Citation
Paper citation, DOI, or BibTeX entry here.