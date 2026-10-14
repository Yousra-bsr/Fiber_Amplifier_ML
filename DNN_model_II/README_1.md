# Multi-Target Neural-Network Regression for NF and Maximum Gain (Model II)

This repository contains a PyTorch regression workflow for predicting
`NF_dB` and `max_gain_dB` simultaneously from four input parameters using an
NLSE dataset.

The script uses the selected Ray Tune configuration as fixed model settings.

## Active model configuration

| Parameter | Value |
| --- | --- |
| Input features | 4 |
| Targets | 2 |
| Hidden layers | `[512, 512, 256, 256, 128, 128]` |
| Hidden-layer count | 6 |
| Activation | LeakyReLU (`0.01`) |
| Dropout | `0.0` |
| Optimizer | AdamW |
| Learning rate | `0.0008` |
| Weight decay | `1e-5` |
| Batch size | `256` |
| Maximum epochs | `500` |
| Validation split | `0.2` |
| Random seed | `42` |
| Early-stopping patience | `60` |
| LR scheduler | CosineAnnealingLR |
| Minimum learning rate | `1e-6` |
| Target scaler | StandardScaler |
| NF ASINH scale | `0.002` |
| Gain ASINH scale | `0.5` |

The network implementation is dynamic, so changing `HIDDEN_SIZES` changes the
number and size of hidden layers without rewriting the model class.

## Data

Place the two project-owned CSV files in:

```text
Data/
├── inputdata_nlse_300000.csv
└── dataTest_nlse_20000.csv
```

The first file is used for training and validation. The second is used for
external testing on unseen data.

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

Rows with missing required values are removed. The workflow also enforces
non-negative `max_gain_dB`.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    └── DNN_model_I.py
```

Running the script creates:

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

Generated outputs are ignored by Git.

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

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU execution, install a PyTorch build compatible with your CUDA setup if
needed.

## Run the model

With the default layout:

```bash
python src/DNN_model_II.py
```


## Preprocessing

The workflow:

1. validates required columns;
2. removes rows with missing required values;
3. enforces non-negative maximum gain;
4. creates a reproducible train/validation split;
5. standardizes the four input features;
6. applies the ASINH transformation to both targets;
7. standardizes the transformed targets.

The target transform is:

```text
asinh(y / scale)
```

with separate scales for `NF_dB` and `max_gain_dB`.

The fitted preprocessing state is stored with the best model checkpoint.

## Training

Training uses:

- L1 loss;
- AdamW;
- CosineAnnealingLR;
- early stopping based on validation loss.

The best validation-loss checkpoint is saved.


## Evaluation metrics

Metrics are reported separately for both targets on the original physical
scale:

- MSE
- RMSE
- MAE
- median absolute error
- R²
- MAPE

Validation MAE and R² on scaled targets are also tracked during training.

## External-test evaluation

If `dataTest_nlse_20000.csv` exists, the script:

1. validates its columns;
2. predicts both targets;
3. reports external-test metrics;
4. saves prediction/error CSV output;
5. generates external-test figures;
6. measures single-sample inference time.

If the file is absent, external evaluation is skipped without stopping the
training/validation workflow.

## Inference timing

Two measurements are saved:

- **end-to-end inference time**, including preprocessing, device transfer,
  model execution, inverse scaling, and inverse ASINH;
- **model-only forward-pass time**, using a prepared tensor.

## Figures and MATLAB data

Figures are saved as PDF and PNG, with corresponding MATLAB
`.mat` numerical data. Histogram diagnostics are displayed but are not saved.

## Reproducibility

The default random seed is `42` for NumPy and PyTorch, and the same seed is
used for the train/validation split.

Exact reproducibility can still depend on hardware, CUDA, and PyTorch versions.

## Version-control policy

This repository is configured to share:

- source code;
- documentation;
- the NLSE training dataset;
- the NLSE external-test dataset.

Generated outputs, checkpoints, MATLAB files, caches, virtual environments,
and editor files are excluded through `.gitignore`.

## Citation

Citation, DOI, or BibTeX entry here.

