# Ray Tune + Optuna + ASHA Regression Search

This repository contains a GPU-aware hyperparameter-search workflow for a four-input, two-target PyTorch MLP.
The code can run on either a GPU or a CPU, although GPU acceleration is recommended for faster hyperparameter search.

The regression targets are `NF_dB` and `max_gain_dB`.

Hyperparameter search is performed with **Ray Tune**, **Optuna**, and **ASHA**.

The purpose of this script is only to identify and save the best hyperparameter configuration.

It does **not** train a final DNN model after the search.

The selected hyperparameters are intended to be used later in a separate DNN training code.

## What is searched

Ray Tune always searches:

- learning rate: log-uniform from `1e-4` to `1e-3`;
- batch size: controlled by `BATCH_CHOICES`;
- 4-, 5-, and 6-hidden-layer MLP architectures;
- AdamW weight decay: log-uniform from `1e-6` to `1e-4`.

The two ASINH-scale hyperparameters are **searched only when `USE_ASINH_TARGET = True`**:

- NF ASINH scale: `2e-4`, `5e-4`, `1e-3`, `2e-3`;
- maximum-gain ASINH scale: `0.05`, `0.1`, `0.2`, `0.5`.

When `USE_ASINH_TARGET = False` (the default), ASINH scaling parameters are **not included** in the Ray Tune search space.

Architecture candidates:

```text
4 layers
[512, 512, 256, 256]
[512, 256, 256, 256]
[512, 512, 512, 256]
5 layers
[512, 512, 256, 256, 128]
[512, 512, 512, 256, 256]
[512, 512, 256, 256, 256]
6 layers
[512, 512, 256, 256, 128, 64]
[512, 512, 256, 256, 256, 128]
[512, 512, 256, 256, 128, 128]
```

## Data

Put the project-owned CSV files in:

```text
Data/
├── dataset_300000.csv
├── dataset_50000.csv
```

The hyperparameter-search script uses the training dataset only.
The external datasets are used in separate DNN training workflows.

Required inputs:

```text
pump_signal_sep_nm
pump_power_dBm
Dcc_ps
zero_disp_wavelength_nm
```

Required targets:

```text
NF_dB
max_gain_dB
```

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    └── ray_tune_hyperparameter_search.py
```

Generated search results are organized under:

```text
outputs/
├── ray_results/
└── hyperparameter_search/
    ├── best_hyperparameters.json
    ├── best_hyperparameters.csv
    └── ray_tune_trial_results.csv
```

The complete `outputs/` folder is ignored by Git.

### Output files

`best_hyperparameters.json` contains the selected Ray Tune configuration, the ASINH mode (use_asinh_target and target_preprocessing), and search metadata such as the best validation loss, reported epoch, random seed, validation split, and learning-rate scheduler information.

`best_hyperparameters.csv` contains the best configuration in compact tabular form, including use_asinh_target; NF/gain ASINH scale columns appear only when ASINH was enabled. It can be read by Python, MATLAB, or spreadsheet software.

`ray_tune_trial_results.csv` contains the Ray Tune results for all completed trials.

These files allow the selected hyperparameters and preprocessing mode to be transferred to the separate DNN training code.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\Activate.ps1    # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA execution, install the PyTorch build that matches the CUDA environment on the target machine.

## Run

With the repository layout above:

```bash
python src/ray_tune_hyperparameter_search.py
```

## Cloud / Runpod compatibility

The workflow can also use a cloud workspace in which the data files are stored directly in a specified base directory.

Set `BASE_DIR`, for example:

```bash
BASE_DIR=/workspace/labs \
python src/ray_tune_hyperparameter_search.py
```

When `BASE_DIR` is set, the script expects the training dataset directly under that directory, for example:

```text
$BASE_DIR/dataset_300000.csv
```

The script writes search summaries to `OUTPUT_DIR / "hyperparameter_search"` and stores Ray experiment results under `RAY_STORAGE_PATH` (default: `OUTPUT_DIR / "ray_results"`). Here `OUTPUT_DIR` is constructed relative to the script repository; setting `BASE_DIR` changes the dataset location but does not automatically relocate the outputs. Use `RAY_STORAGE_PATH` to change where Ray stores its experiment data.

## Important environment variables

| Variable | Default | Purpose |
| --- | ---: | --- |
| `NUM_TRIALS` | `100` | Number of Ray Tune trials |
| `TUNE_EPOCHS` | `500` | Maximum epochs per tuning trial |
| `TUNE_EARLY_STOPPING_PATIENCE` | `80` | Trial-local early-stopping patience |
| `TRIALS_PER_GPU` | `4` | Fractional-GPU sharing factor |
| `CPUS_PER_TRIAL` | `2` | CPU resources requested per trial |
| `SYSTEM_RESERVED_CPUS` | `4` on CUDA systems | CPUs left for OS/driver overhead |
| `TARGET_CONCURRENT_TRIALS` | auto | Optional concurrency cap |
| `BATCH_CHOICES` | `64,128,256` | Batch sizes searched |
| `TUNE_COMPILE` | `0` | Try `torch.compile` during tuning |
| `ASHA_GRACE_PERIOD` | `75` | Minimum training period before ASHA can stop a trial |
| `ASHA_REDUCTION_FACTOR` | `3` | ASHA reduction factor |
| `RAY_STORAGE_PATH` | `outputs/ray_results` | Ray Tune result storage |
| `BASE_DIR` | unset | Optional cloud-compatible project/data root |

The Boolean **environment variables listed above** accept forms such as `1/0`, `true/false`, and `yes/no`. `USE_ASINH_TARGET` is configured directly in the Python script.


## B300-oriented example

A reasonable starting point for a multi-GPU cloud machine such as the environment targeted by the original code is:

```bash
TRIALS_PER_GPU=4 \
CPUS_PER_TRIAL=2 \
SYSTEM_RESERVED_CPUS=4 \
USE_BF16=1 \
NUM_TRIALS=100 \
python src/ray_tune_hyperparameter_search.py
```

The repository default keeps:

```text
BATCH_CHOICES=64,128,256
```

## Target preprocessing

All four input features are standardized with `StandardScaler`, fitted only on training samples. The same fitted input scaler is applied to the validation samples. Input preprocessing is independent of the tuned hyperparameters and is performed once before the Ray Tune trials.

The target transformation is controlled by a **Python configuration constant** in `ray_tune_hyperparameter_search.py`:

```python
USE_ASINH_TARGET = False  # Default: disable ASINH
NF_ASINH_CHOICES = [2e-4, 5e-4, 1e-3, 2e-3]
GAIN_ASINH_CHOICES = [0.05, 0.1, 0.2, 0.5]
```

Change the value of `USE_ASINH_TARGET` to switch modes before launching the hyperparameter search. This is a code setting, **not an environment variable**.

### Mode 1 — `USE_ASINH_TARGET = False` (default)

- No ASINH transformation is applied to either `NF_dB` or `max_gain_dB`.
- Each target is standardized directly with a training-fitted `StandardScaler`:

  `y_scaled = (y - mean_train) / std_train`

- Ray Tune searches only learning rate, batch size, hidden-layer architecture, and weight decay.
- `nf_asinh_scale` and `gain_asinh_scale` are absent from trial configurations and the selected best configuration.
- Only one standardized target dataset is prepared and cached for use by the Ray workers.

### Mode 2 — `USE_ASINH_TARGET = True`

For each target, the code first applies an inverse hyperbolic sine transformation with its own tunable scale:

```text
NF_transformed   = asinh(NF_dB / nf_asinh_scale)
Gain_transformed = asinh(max_gain_dB / gain_asinh_scale)
```

The transformed NF and gain values are then each standardized using `StandardScaler` fitted on their respective training values; the training-fitted scalers are used on validation values as well.

Ray Tune additionally searches the following hyperparameters:

```python
NF_ASINH_CHOICES = [2e-4, 5e-4, 1e-3, 2e-3]
GAIN_ASINH_CHOICES = [0.05, 0.1, 0.2, 0.5]
```

These produce **4 × 4 = 16 possible NF/gain scale pairs**. The transformed and standardized training/validation targets for all 16 pairs are precomputed once and cached for reuse by Ray workers. Each trial selects the cached arrays corresponding to its own proposed scales.

| Configuration | Target preprocessing | ASINH-scale tuning | Cached target configurations |
| --- | --- | --- | ---: |
| `USE_ASINH_TARGET = False` | `StandardScaler` only | No | 1 |
| `USE_ASINH_TARGET = True` | `ASINH` → `StandardScaler` | Yes: NF and gain scales | 16 |

**Important:** the training/validation L1 loss is calculated in the *standardized target space*. If ASINH is enabled, different ASINH scales change the transformed target space, so trial validation losses do not directly represent errors in original dB units. To compare the physical prediction accuracy of selected models or preprocessing strategies, inverse-transform predictions and calculate NF/gain MAE, RMSE, and R² on the same held-out dataset in dB. The hyperparameter-search script itself does not perform this final evaluation.

## Hyperparameter search process

Each Ray Tune trial temporarily trains an MLP using one proposed hyperparameter configuration.

This trial-level training is necessary because Ray Tune must measure the validation performance of each candidate configuration. It is separate from the definitive DNN training that is performed later in another script.

During each trial:

1. the target arrays for the selected preprocessing mode (and ASINH scales, if enabled) are selected;
2. the candidate MLP architecture is created;
3. AdamW is configured with the candidate learning rate and weight decay;
4. the model is trained for up to `TUNE_EPOCHS`;
5. validation L1 loss is reported to Ray Tune after each epoch;
6. ASHA can terminate poorly performing trials early;
7. the best-performing configurations are favored by Optuna.

The learning-rate scheduler used inside each trial is:

```text
CosineAnnealingLR
```

with:

```text
eta_min = 1e-6
T_max = TUNE_EPOCHS
```

## Optuna and ASHA

Optuna proposes promising hyperparameter configurations using information obtained from previous trials.

ASHA is responsible for trial-level early stopping. It prevents computational resources from being spent unnecessarily on configurations that perform poorly during the early stages of training.

The optimization metric is:

```text
val_loss
```

with:

```text
mode = min
```

Therefore, the configuration with the lowest reported validation L1 loss is selected as the best configuration. This loss is measured on standardized targets (ASINH-transformed first if enabled), rather than in physical dB units.

## Best hyperparameters

After all Ray Tune trials have completed, the best configuration is selected using:

```text
metric = val_loss
mode = min
scope = all
```

The selected configuration always includes:

- learning rate;
- batch size;
- hidden-layer architecture;
- AdamW weight decay.

The saved CSV also records the number of hidden layers. When `USE_ASINH_TARGET = True`, the best configuration additionally includes `nf_asinh_scale` and `gain_asinh_scale`. **When it is `False`, neither scale appears in the best configuration**, because neither was tuned.

The results are saved to:

```text
outputs/hyperparameter_search/best_hyperparameters.json
outputs/hyperparameter_search/best_hyperparameters.csv
```

Both output formats record `use_asinh_target`. The JSON additionally records `target_preprocessing` as `standard_only` or `asinh_then_standard`. The CSV includes the two scale columns only when ASINH was enabled.

No final DNN model is retrained, evaluated, or checkpointed by this script. The saved hyperparameters should be used in the separate DNN training code. 

**Note:** the summary files use the same filenames in either mode, so running the search again can overwrite earlier summaries. 

## Ray Tune trial results

The complete Ray Tune result table is saved as:

```text
outputs/hyperparameter_search/ray_tune_trial_results.csv
```

This file can be used to inspect and compare all tested configurations, including their validation performance and Ray Tune metadata.

The full Ray Tune experiment data are also stored under:

```text
outputs/ray_results/
```

## Reproducibility

The default random seed is:

```text
42
```

The train/validation split is fixed using the same random seed.

Exact GPU results can nevertheless depend on:

- PyTorch version;
- CUDA version;
- cuDNN version;
- GPU hardware;
- BF16 mixed precision;
- TF32 behavior;
- Ray scheduling;
- parallel trial execution.

Therefore, small numerical differences between runs can occur even when the same search configuration is used.

## Version-control policy

The repository shares:

- source code;
- documentation;
- project-owned CSV datasets.

Generated search outputs are excluded from version control, including:

- Ray Tune experiment results;
- saved hyperparameter-search outputs;
- caches;
- virtual environments;
- temporary files;
- editor files.

The complete `outputs/` directory is ignored through `.gitignore`.

## Citation

Paper citation, DOI, or BibTeX entry here.
