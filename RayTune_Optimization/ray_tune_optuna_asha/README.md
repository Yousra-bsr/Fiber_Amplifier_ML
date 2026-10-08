# Ray Tune + Optuna + ASHA Regression Search

This repository contains a GPU-aware hyperparameter-search workflow for a four-input, two-target PyTorch MLP.
The code can run on either a GPU or a CPU, although GPU acceleration is recommended for faster hyperparameter search.

The regression targets are `NF_dB` and `max_gain_dB`. 
Hyperparameter search is performed with **Ray Tune**, **Optuna**, and **ASHA**.

The purpose of this script is only to identify and save the best hyperparameter configuration.
It does **not** retrain a final DNN model after the search.
The selected hyperparameters are intended to be used later in a separate DNN training code.

## What is searched

The search space contains:

- learning rate: log-uniform from `1e-4` to `1e-3`;
- batch size: controlled by `BATCH_CHOICES`;
- 4-, 5-, and 6-hidden-layer MLP architectures;
- AdamW weight decay: log-uniform from `1e-6` to `1e-4`;
- NF ASINH scale: `2e-4`, `5e-4`, `1e-3`, `2e-3`;
- gain ASINH scale: `0.05`, `0.1`, `0.2`, `0.5`.

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
├── dataset_nlse_300000.csv
├── dataset_ext_nlse_20000.csv
├── dataset_200000.csv
└── dataset_ext_20000.csv
```

The hyperparameter-search script uses the training dataset only.
The external datasets are not used by this script because final model evaluation has been moved to the separate DNN training workflow.

Required inputs:

```text
pump_signal_sep_nm
pump_power_dBm
beta2_ps2_per_km
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

`best_hyperparameters.json` contains the selected Ray Tune configuration together with search metadata such as the best validation loss, reported epoch, random seed, validation split, and learning-rate scheduler information.

`best_hyperparameters.csv` contains the best configuration in a compact tabular form that can easily be read by Python, MATLAB, or spreadsheet software.

`ray_tune_trial_results.csv` contains the Ray Tune results for all completed trials.

These files allow the selected hyperparameters to be transferred directly to the separate DNN training code.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\Activate.ps1    # Windows PowerShell

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
$BASE_DIR/dataset_nlse_300000.csv
```

or, after changing the selected dataset in the script:

```text
$BASE_DIR/dataset_200000.csv
```

Generated files are written to:

```text
$BASE_DIR/outputs/
```

## Important environment variables

| Variable | Default | Purpose |
| --- | ---: | --- |
| `NUM_TRIALS` | `200` | Number of Ray Tune trials |
| `TUNE_EPOCHS` | `500` | Maximum epochs per tuning trial |
| `TUNE_EARLY_STOPPING_PATIENCE` | `80` | Trial-local early-stopping patience |
| `TRIALS_PER_GPU` | `4` | Fractional-GPU sharing factor |
| `CPUS_PER_TRIAL` | `2` | CPU resources requested per trial |
| `SYSTEM_RESERVED_CPUS` | `4` on CUDA systems | CPUs left for OS/driver overhead |
| `TARGET_CONCURRENT_TRIALS` | auto | Optional concurrency cap |
| `BATCH_CHOICES` | `64,128,256` | Batch sizes searched |
| `USE_BF16` | `1` | BF16 autocast when supported |
| `USE_FUSED_ADAMW` | `1` | Fused AdamW when supported |
| `TUNE_COMPILE` | `0` | Try `torch.compile` during tuning |
| `ASHA_GRACE_PERIOD` | `75` | Minimum training period before ASHA can stop a trial |
| `ASHA_REDUCTION_FACTOR` | `3` | ASHA reduction factor |
| `RAY_STORAGE_PATH` | `outputs/ray_results` | Ray Tune result storage |
| `BASE_DIR` | unset | Optional cloud-compatible project/data root |

Boolean values accept forms such as `1/0`, `true/false`, and `yes/no`.

The final-training-related variables from the previous version, such as `FINAL_EPOCHS`, `FINAL_EARLY_STOPPING_PATIENCE`, `FINAL_COMPILE`, `EVAL_USE_BF16`, and `SAVE_OPTIMIZER_STATE`, are no longer required because this script performs hyperparameter search only.

## B300-oriented example

A reasonable starting point for a multi-GPU cloud machine such as the environment targeted by the original code is:

```bash
TRIALS_PER_GPU=4 \
CPUS_PER_TRIAL=2 \
SYSTEM_RESERVED_CPUS=4 \
USE_BF16=1 \
NUM_TRIALS=200 \
python src/ray_tune_hyperparameter_search.py
```

The repository default keeps:

```text
BATCH_CHOICES=64,128,256
```

## Performance design retained

The search-only code keeps the important GPU-oriented performance choices from the original research script:

- fractional Ray GPU resources so multiple trials can share each physical GPU;
- one host-to-device transfer of the trial data;
- device-side shuffling;
- GPU-resident validation;
- no per-minibatch `.item()` synchronization;
- BF16 autocast when supported;
- fused AdamW with a compatibility fallback;
- TF32/high-throughput CUDA matrix paths;
- precomputed ASINH target transformations and target scalers;
- Ray object-store sharing of preprocessed trial data;
- optional `torch.compile`;
- automatic CPU/GPU resource detection;
- configurable trial concurrency.

These optimizations are particularly useful when many Ray Tune trials are executed on multi-GPU systems.

## Target preprocessing

Input features are standardized using `StandardScaler`.

The input scaling is independent of the tuned hyperparameters and is therefore computed once before the Ray Tune trials begin.

The two targets are transformed using candidate ASINH scales:

```text
asinh(y / scale)
```

For every candidate combination of NF and gain ASINH scales, the transformed targets are standardized before training.

These transformed and standardized target arrays are precomputed once and shared with Ray workers instead of being recalculated independently inside every trial.

The searched ASINH scales are:

```text
NF:
2e-4
5e-4
1e-3
2e-3

Max gain:
0.05
0.1
0.2
0.5
```

## Hyperparameter search process

Each Ray Tune trial temporarily trains an MLP using one proposed hyperparameter configuration.

This trial-level training is necessary because Ray Tune must measure the validation performance of each candidate configuration. It is separate from the definitive DNN training that is performed later in another script.

During each trial:

1. the corresponding preprocessed target data are selected;
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

Therefore, the configuration with the lowest reported validation L1 loss is selected as the best configuration.

## Best hyperparameters

After all Ray Tune trials have completed, the best configuration is selected using:

```text
metric = val_loss
mode = min
scope = all
```

The selected configuration includes:

- learning rate;
- batch size;
- hidden-layer architecture;
- number of hidden layers;
- AdamW weight decay;
- NF ASINH scale;
- maximum-gain ASINH scale.

The script saves this configuration to:

```text
outputs/hyperparameter_search/best_hyperparameters.json
```

and:

```text
outputs/hyperparameter_search/best_hyperparameters.csv
```

No final DNN model is retrained, evaluated, or checkpointed by this script.

The saved hyperparameters should instead be loaded into the separate DNN training code used for the final model.

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
