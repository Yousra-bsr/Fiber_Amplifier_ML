# Dataset Generation for Dual-Pump Fiber-Optic Phase-Sensitive Amplifiers

This folder contains a Python script that generates a numerical dataset for predicting the **noise figure (NF)** and **maximum gain** of a dual-pump fiber-optic phase-sensitive amplifier (PSA). The computation follows a **semi-quantum, Inoue-style approach** and uses Numba-accelerated simulations to produce samples suitable for machine-learning workflows.


## Files

```text
.
├── generate_Train_test_datasets.py  # Numerical dataset generator
├── README.md                        # Documentation
├── requirements.txt                 # Required Python packages
└── .gitignore                       # Local Python files and temporary output
```

Generated CSV datasets can be stored in this folder or in a `Data/` subdirectory. The `.gitignore` **does not exclude CSV datasets**, so datasets intended for the repository can be committed to GitHub.

## What the script does

For each sample, the script:

1. Uniformly samples four physical input parameters from configurable ranges.
2. Converts the dispersion slope and zero-dispersion wavelength into the dispersion coefficients used by the model.
3. Integrates the coupled pump equations along the fiber with a fourth-order Runge–Kutta (RK4) method.
4. Evaluates the signal/idler transfer calculation with a memory-efficient row update equivalent to a product of 6 × 6 segment matrices.
5. Computes the noise figure and maximum gain in decibels.
6. Writes the results to a CSV file in batches, with Numba parallelizing the simulations across CPU threads.

**Important:** To generate separate datasets, run the script twice: once to create the training/validation dataset and once to create the test dataset, specifying a different output filename for each run.

## Dataset format

Each CSV row contains four inputs followed by two prediction targets, in this exact order:

| Column | Type | Description | Default sampling range |
| --- | --- | --- | --- |
| `pump_signal_sep_nm` | Input | Approximate signal–pump wavelength separation, in nm | 2–25 nm |
| `pump_power_dBm` | Input | Optical power **per pump**, in dBm | 19–25 dBm |
| `Dcc_ps` | Input | Dispersion slope, in ps/(nm²·km) | 0.01–0.07 |
| `zero_disp_wavelength_nm` | Input | Zero-dispersion wavelength, in nm | 1500–1600 nm |
| `NF_dB` | Target | Computed noise figure, in dB | Model output |
| `max_gain_dB` | Target | Computed maximum gain, in dB | Model output |

The signal wavelength is fixed at **1550 nm** in the code. The input sampling is uniform over the specified ranges.

CSV header:

```csv
pump_signal_sep_nm,pump_power_dBm,Dcc_ps,zero_disp_wavelength_nm,NF_dB,max_gain_dB
```

## Requirements

- Python 3.10 or newer is recommended.
- NumPy and Numba (see `requirements.txt`).
- A CPU; no GPU is required. Numba can parallelize batches across multiple CPU threads.

All other imports in the script (`argparse`, `csv`, `math`, `os`, `pathlib`, `time`) are part of Python's standard library.

## Installation

From the folder containing the script:

```bash
python -m venv .venv
```

Activate the environment:

- **Windows PowerShell:** `.\.venv\Scripts\Activate.ps1`
- **Linux / macOS:** `source .venv/bin/activate`

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

### 1. Verify the implementation

```bash
python generate_Train_test_datasets.py --self-test
```

The self-test checks the sparse transfer-matrix row update, dispersion signs, and finite simulation outputs for both phase conventions. It does **not** generate a dataset.

### 2. Generate the default dataset

```bash
python generate_Train_test_datasets.py
```

The default command generates **300,000 samples** in `dataset_300000.csv`, using a random seed of `19`, a batch size of `500`, and the `original` phase mode. The output path is relative to the terminal's current working directory unless an absolute path is supplied.

### 3. Generate a small dataset for testing

```bash
python generate_Train_test_datasets.py --n 100 --out data/example_100.csv --threads 4
```

The script creates the output directory if necessary.

### 4. Create datasets with different phase conventions

```bash
python generate_Train_test_datasets.py --n 10000 --phase-mode original --out data/original_10000.csv
python generate_Train_test_datasets.py --n 10000 --phase-mode accumulated --out data/accumulated_10000.csv
```

- **`original` (default):** uses the local phase-mismatch expression `Δβ(z) × z`.
- **`accumulated`:** integrates the phase mismatch along the fiber as an alternative convention.

Use different output filenames to avoid overwriting an earlier completed dataset.

### 5. Customize the sampling ranges

```bash
python generate_Train_test_datasets.py \
  --n 5000 \
  --pump-min 19 --pump-max 25 \
  --sep-min 2 --sep-max 25 \
  --dcc-min 0.01 --dcc-max 0.07 \
  --lambda0-min 1500 --lambda0-max 1600 \
  --out data/custom_5000.csv
```

The multiline example above uses Unix-style `\` line continuation. In Windows PowerShell, enter the command on one line or use PowerShell's backtick (`` ` ``) for line continuation.

## Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--n` | `300000` | Number of generated samples |
| `--out` | `dataset_300000.csv` | Output CSV path |
| `--chunk-size` | `500` | Samples simulated and written per batch |
| `--seed` | `19` | Random-number generator seed |
| `--threads` | Numba default | Number of CPU threads |
| `--phase-mode` | `original` | `original` or `accumulated` |
| `--pump-min`, `--pump-max` | `19`, `25` | Power range per pump, dBm |
| `--sep-min`, `--sep-max` | `2`, `25` | Signal–pump separation range, nm |
| `--dcc-min`, `--dcc-max` | `0.01`, `0.07` | Dispersion slope range, ps/(nm²·km) |
| `--lambda0-min`, `--lambda0-max` | `1500`, `1600` | Zero-dispersion wavelength range, nm |
| `--length-m` | `300` | Fiber length, m |
| `--gamma-km` | `12` | Nonlinear coefficient, W⁻¹·km⁻¹ |
| `--max-step-m` | `0.1` | Maximum spatial step, m |
| `--self-test` | Off | Run internal numerical checks and exit |

Use `python generate_Train_test_datasets.py --help` to see the script's built-in option descriptions.

### Default numerical configuration

- Fiber length: **300 m**.
- Nonlinearity: **12 W⁻¹·km⁻¹**, converted internally to units per meter.
- Maximum requested step: **0.1 m**.
- Integration steps per sample: `max(1000, ceil(length_m / max_step_m))`, which is **3,000 steps** at the defaults.
- Both pumps start with equal optical power specified by `pump_power_dBm`.

## Output and reproducibility

- Samples are processed in chunks, rather than requiring the entire dataset in memory.
- The script writes to a temporary file such as `dataset_300000.csv.partial` and replaces the final output CSV only after successful completion. If execution fails, the `.partial` file may remain for inspection.
- Non-finite NF or gain values cause the generation to stop instead of silently exporting invalid rows.
- The default seed is `19`. For reproducible sampling, also keep the **chunk size**, parameter ranges, and script version fixed. Changing the chunk size can change how random draws are grouped into rows, even with the same seed.
- The first simulation call includes Numba JIT compilation, so initial execution can be slower. Large datasets may require substantial CPU processing time.

## Using the data for machine learning

The four input columns can be used as model features and `NF_dB` / `max_gain_dB` as regression targets. Create training, validation, and test partitions in a separate preprocessing step, keeping those partitions distinct during model evaluation. 

## GitHub notes

The accompanying `.gitignore` excludes Python caches, local virtual environments, editor settings, and unfinished `.csv.partial` outputs. **It intentionally allows final `.csv` datasets to be tracked**, in keeping with a repository that includes the datasets used in the project. 