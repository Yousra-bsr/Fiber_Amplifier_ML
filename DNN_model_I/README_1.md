# Multi-Target Neural-Network Regression for NF and Maximum Gain (Model I)

This repository contains a PyTorch regression workflow for predicting two
targets simultaneously:

- **Noise figure:** `NF_dB`
- **Maximum gain:** `max_gain_dB`

The model uses four physical/input features and the currently selected
Ray-Tune-based architecture. The training script includes preprocessing,
target transformation, training with early stopping, validation, 
external-test evaluation, inference timing, and figure/data export.

> **Note**
> Ray Tune itself is not required to run this repository. The script uses the
> selected configuration as fixed training parameters; it does not run a new
> hyperparameter search.

## Current model configuration

The active configuration in `src/DNN_model_I.py` is:

| Parameter | Value |
| --- | --- |
| Input features | 4 |
| Targets | 2 |
| Hidden layers | `[512, 512, 512, 256]` |
| Hidden-layer count | 4 |
| Activation | LeakyReLU (`0.01`) |
| Dropout | `0.0` |
| Optimizer | AdamW |
| Learning rate | `0.001` |
| Weight decay | `1e-5` |
| Batch size | `128` |
| Maximum epochs | `500` |
| Validation split | `0.2` |
| Random seed | `42` |
| Early-stopping patience | `60` |
| LR scheduler | CosineAnnealingLR |
| Minimum learning rate | `1e-6` |
| Target scaler | StandardScaler |
| NF ASINH scale | `0.002` |
| Gain ASINH scale | `0.5` |

The MLP implementation is dynamic: changing `HIDDEN_SIZES` changes the number
and size of the hidden layers without rewriting the model class.

## Input data

Place the CSV files in the `Data/` directory:

```text
Data/
├── dataset_200000.csv
└── dataset_ext_20000.csv
```

The first file is used for training and validation. The second is used as an external test dataset.

### Required input columns

```text
pump_signal_sep_nm
pump_power_dBm
beta2_ps2_per_km
zero_disp_wavelength_nm
```

### Required target columns

```text
NF_dB
max_gain_dB
```

Rows containing missing values in the required columns are dropped. The script
also enforces non-negative values for `max_gain_dB` before training/evaluation.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    └── DNN_model_I.py
```

When the script runs, it creates an `outputs/` directory automatically:

```text
outputs/
├── checkpoints/
├── figures/
│   ├── pdf/
│   └── png/
├── matlab/
├── predictions/
└── timing/
```

Generated outputs are ignored by Git by default.

## Installation

Python 3.10 or newer is recommended.

Create and activate a virtual environment.

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

Install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA/GPU use, install the PyTorch build appropriate for your CUDA
environment if the default `pip install torch` build is not suitable.

## Running the model

With the default folder structure:

```bash
python src/DNN_model_I.py
```

## Preprocessing

The workflow:

1. checks that all required columns exist;
2. drops rows with missing required values;
3. enforces non-negative `max_gain_dB`;
4. creates a reproducible train/validation split;
5. standardizes the four input features;
6. applies the column-wise ASINH target transform;
7. standardizes the transformed targets.

The target transform is

```text
asinh(y / scale)
```

with separate scales for `NF_dB` and `max_gain_dB`.

The fitted input and target scaling information is saved with the best model
checkpoint so predictions can be returned to the original physical scale.

## Training

The model is trained with:

- L1 loss;
- AdamW;
- CosineAnnealingLR;
- early stopping based on validation loss.

The best validation-loss checkpoint is saved rather than only retaining the
final epoch.


## Evaluation metrics

Metrics are reported separately for `NF_dB` and `max_gain_dB` on the original
target scale:

- MSE
- RMSE
- MAE
- median absolute error
- R²
- MAPE

Validation MAE and R² on the scaled targets are also recorded during training.

## Figures and MATLAB data

Figures are saved as:

- PDF
- PNG
- MATLAB `.mat` data

The script generates training/validation loss curves, validation MAE and R²
curves, predicted-vs-actual plots, and absolute-error-vs-actual plots.

Target-distribution and prediction-error histograms are displayed but are not
saved.

## External test evaluation

If the external-test CSV exists, the script:

1. validates its required columns;
2. predicts both targets;
3. reports the regression metrics;
4. saves a CSV containing predictions and errors;
5. generates external-test plots;
6. measures single-sample inference time.

If the file is absent, external evaluation is skipped without stopping the
training/validation workflow.

## Inference timing

The timing output separates two measurements:

- **end-to-end inference**, including scaling, tensor conversion/device
  transfer, model execution, inverse target scaling, and inverse ASINH;
- **model-only forward-pass time**, using an already prepared input tensor.

This distinction is useful when reporting computational performance.

## Reproducibility

The default random seed is `42` for NumPy and PyTorch. The train/validation
split also uses this seed.

Exact numerical reproducibility can still depend on hardware, PyTorch/CUDA
versions, and GPU operations.


## Citation

Citation, DOI, or BibTeX entry here once available.
