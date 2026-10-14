"""
Train and evaluate a four-input, two-target neural-network regressor
using the selected Ray Tune configuration.

Targets
-------
- NF_dB
- max_gain_dB

The script:
- loads a training CSV;
- preprocesses and scales four input features and two targets;
- Applies an ASINH transform to the targets;
- trains a configurable fully connected PyTorch MLP;
- saves the best checkpoint;
- evaluates a validation split;
- Evaluates an external CSV;
- saves plots as PDF and PNG;
- saves the data behind plots as MATLAB .mat files.

"""

import os
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import savemat

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    mean_absolute_percentage_error
)

# ------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------
print("--- 0. Configuration ---")

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "dataset_200000.csv"
NEW_DATA_PATH = PROJECT_ROOT / "data" / "dataset_ext_20000.csv"

INPUT_COLS = [
    "pump_signal_sep_nm",
    "pump_power_dBm",
    "beta2_ps2_per_km",
    "zero_disp_wavelength_nm"
]

TARGET_COLS = [
    "NF_dB",
    "max_gain_dB"
]

INPUT_SIZE = len(INPUT_COLS)
OUTPUT_SIZE = len(TARGET_COLS)

# ------------------------------------------------------------
# Best hyperparameters found by Ray Tune Optimization
# ------------------------------------------------------------
HIDDEN_SIZES = [512, 512, 512, 256]

DROP_OUT_RATE = 0.0

LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS        = 500
VALIDATION_SPLIT = 0.2
RANDOM_SEED      = 42
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 60
NUM_SAMPLES_TO_SHOW     = 5

Y_SCALER_TYPE   = "standard"

USE_ASINH_TARGET = True

NF_ASINH_SCALE = 0.002
GAIN_ASINH_SCALE = 0.5

ASINH_SCALES     = np.array(
    [NF_ASINH_SCALE, GAIN_ASINH_SCALE],
    dtype=np.float32
).reshape(1, -1)

# CosineAnnealingLR settings
# The learning rate decreases smoothly from LEARNING_RATE
# toward MIN_LEARNING_RATE over EPOCHS epochs.
MIN_LEARNING_RATE = 1e-6

# ------------------------------------------------------------
# Output folders for plots
# ------------------------------------------------------------

PLOT_OUTPUT_FOLDER = PROJECT_DIR / "Outputs_model_I"

# Model checkpoint
CHECKPOINT_PATH = os.path.join(
    PLOT_OUTPUT_FOLDER,
    "best_model_NF_and_gain_model_I.pth"
)

PDF_FOLDER = os.path.join(PLOT_OUTPUT_FOLDER, "pdf_figures")
PNG_FOLDER = os.path.join(PLOT_OUTPUT_FOLDER, "png_figures")
MAT_FOLDER = os.path.join(PLOT_OUTPUT_FOLDER, "matlab_figure_data")

os.makedirs(PLOT_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(PNG_FOLDER, exist_ok=True)
os.makedirs(MAT_FOLDER, exist_ok=True)

def save_current_figure(figure_name, mat_data=None):
    """
    Save the current matplotlib figure as:
      - PDF
      - PNG
      - MAT file containing the data needed to reproduce the figure in MATLAB
    """

    pdf_path = os.path.join(PDF_FOLDER, figure_name + ".pdf")
    png_path = os.path.join(PNG_FOLDER, figure_name + ".png")
    mat_path = os.path.join(MAT_FOLDER, figure_name + ".mat")

    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    if mat_data is not None:
        savemat(mat_path, mat_data)

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")

    if mat_data is not None:
        print(f"Saved MAT: {mat_path}")

print("Configuration set:")
print(f"  DATA_PATH={DATA_PATH}")
print(f"  NEW_DATA_PATH={NEW_DATA_PATH}")
print(f"  INPUT_COLS={INPUT_COLS}")
print(f"  TARGET_COLS={TARGET_COLS}")
print(f"  INPUT_SIZE={INPUT_SIZE}, OUTPUT_SIZE={OUTPUT_SIZE}")
print(f"  HIDDEN_SIZES={HIDDEN_SIZES}")
print(f"  LEARNING_RATE={LEARNING_RATE}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}")
print(f"  DROP_OUT_RATE={DROP_OUT_RATE}, WEIGHT_DECAY={WEIGHT_DECAY}")
print(f"  USE_ASINH_TARGET={USE_ASINH_TARGET}")
print(f"  ASINH_SCALES={ASINH_SCALES.flatten().tolist()}")
print(f"  CHECKPOINT_PATH={CHECKPOINT_PATH}")
print("  LR_SCHEDULER=CosineAnnealingLR")
print(f"  COSINE_T_MAX={EPOCHS}, MIN_LEARNING_RATE={MIN_LEARNING_RATE}")
print(f"  PLOT_OUTPUT_FOLDER={PLOT_OUTPUT_FOLDER}")
print("-" * 30)

# ------------------------------------------------------------
# Device and seed setup
# ------------------------------------------------------------
print("--- Device Setup ---")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print(f"Seeds set. Seed={RANDOM_SEED}")
print("-" * 30)

# ------------------------------------------------------------
# Target transform helpers for two targets
# ------------------------------------------------------------
def transform_targets(y_np, use_asinh=True, scales=None):
    """
    Applies ASINH transform column-wise.
    Expected shape: [N, 2]
    Column 0: NF_dB
    Column 1: max_gain_dB
    """

    y_np = y_np.astype(np.float32).copy()

    if not use_asinh:
        return y_np

    if y_np.ndim != 2:
        raise ValueError(f"Expected target shape [N, num_targets], got {y_np.shape}")

    if scales is None:
        raise ValueError("scales must be provided when using ASINH transform.")

    return np.arcsinh(y_np / scales).astype(np.float32)

def inverse_transform_targets(y_trans_np, use_asinh=True, scales=None):
    """
    Inverts ASINH transform column-wise.
    Expected shape: [N, 2]
    """

    y_trans_np = y_trans_np.astype(np.float32).copy()

    if not use_asinh:
        return y_trans_np

    if y_trans_np.ndim != 2:
        raise ValueError(f"Expected target shape [N, num_targets], got {y_trans_np.shape}")

    if scales is None:
        raise ValueError("scales must be provided when using inverse ASINH transform.")

    return (scales * np.sinh(y_trans_np)).astype(np.float32)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
print("--- 1. Loading Data ---")
start_load_time = time.time()

try:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")
    print("Columns found:")
    print(df.columns.tolist())

except FileNotFoundError as e:
    print("\n*** ERROR: File Not Found ***")
    print(e)
    raise

except Exception as e:
    print("\n*** Unexpected error during data loading ***")
    print(e)
    raise

print(f"Data loading finished in {time.time() - start_load_time:.2f}s.")
print("-" * 30)

# ------------------------------------------------------------
# 2. Preprocessing
# ------------------------------------------------------------
print("--- 2. Preprocessing Data ---")
start_preprocess_time = time.time()

required_cols = INPUT_COLS + TARGET_COLS
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print("*** ERROR: Missing required columns ***")
    print(missing_cols)
    raise

df_model = df[required_cols].copy()

n_missing_rows = df_model.isnull().any(axis=1).sum()

if n_missing_rows > 0:
    print(f"Warning: {n_missing_rows} rows contain missing values and will be dropped.")
    df_model = df_model.dropna()

x_data_np = df_model[INPUT_COLS].values.astype(np.float32)
y_data_np = df_model[TARGET_COLS].values.astype(np.float32)

# Physically enforce non-negative gain
y_data_np[:, TARGET_COLS.index("max_gain_dB")] = np.maximum(
    y_data_np[:, TARGET_COLS.index("max_gain_dB")],
    0.0
)

num_samples = len(df_model)

print(f"Number of usable samples: {num_samples}")
print(f"X shape: {x_data_np.shape}")
print(f"Y shape: {y_data_np.shape}")

print(f"\n--- Sample Raw Data: first {NUM_SAMPLES_TO_SHOW} samples ---")

for i in range(min(NUM_SAMPLES_TO_SHOW, len(x_data_np))):
    input_values = ", ".join([
        f"{INPUT_COLS[j]}={x_data_np[i, j]:.6f}"
        for j in range(INPUT_SIZE)
    ])

    target_values = ", ".join([
        f"{TARGET_COLS[k]}={y_data_np[i, k]:.6f}"
        for k in range(OUTPUT_SIZE)
    ])

    print(f"Sample {i + 1}: {input_values}, {target_values}")

# ============================================================
# Target distributions
# Histogram plots are shown but NOT saved
# ============================================================
print("\n--- Diagnostic: Target Distributions ---")

for target_col in TARGET_COLS:
    print(f"\nTarget: {target_col}")
    print(f"  Min   : {df_model[target_col].min():.6f}")
    print(f"  Max   : {df_model[target_col].max():.6f}")
    print(f"  Mean  : {df_model[target_col].mean():.6f}")
    print(f"  Median: {df_model[target_col].median():.6f}")
    print(f"  Std   : {df_model[target_col].std():.6f}")

    plt.figure(figsize=(8, 5))
    plt.hist(
        df_model[target_col].values,
        bins=40,
        edgecolor="black",
        alpha=0.7
    )
    plt.title(f"Histogram of {target_col}")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()

if USE_ASINH_TARGET:
    y_asinh_all = transform_targets(
        y_data_np,
        use_asinh=True,
        scales=ASINH_SCALES
    )

    for k, target_col in enumerate(TARGET_COLS):
        plt.figure(figsize=(8, 5))
        plt.hist(
            y_asinh_all[:, k],
            bins=40,
            edgecolor="black",
            alpha=0.7
        )
        plt.title(f"Histogram of ASINH-transformed {target_col}")
        plt.xlabel(f"asinh({target_col} / {ASINH_SCALES[0, k]})")
        plt.ylabel("Count")
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.show()

# ------------------------------------------------------------
# Train/validation split
# ------------------------------------------------------------
print("\nSplitting data...")

all_indices = np.arange(len(df_model))

train_indices, val_indices = train_test_split(
    all_indices,
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_SEED,
    shuffle=True
)

x_train_np = x_data_np[train_indices]
x_val_np = x_data_np[val_indices]

y_train_np = y_data_np[train_indices]
y_val_np = y_data_np[val_indices]

print(f"Training samples: {len(train_indices)}")
print(f"Validation samples: {len(val_indices)}")

# ------------------------------------------------------------
# Scale X
# ------------------------------------------------------------
x_scaler = StandardScaler()

x_train_scaled_np = x_scaler.fit_transform(x_train_np)
x_val_scaled_np = x_scaler.transform(x_val_np)

if np.isnan(x_train_scaled_np).any() or np.isnan(x_val_scaled_np).any():
    raise ValueError("NaNs detected after X scaling.")

# ------------------------------------------------------------
# Transform and scale Y
# ------------------------------------------------------------
y_train_orig_np = y_train_np.copy()
y_val_orig_np = y_val_np.copy()

y_train_trans_np = transform_targets(
    y_train_np,
    use_asinh=USE_ASINH_TARGET,
    scales=ASINH_SCALES
)

y_val_trans_np = transform_targets(
    y_val_np,
    use_asinh=USE_ASINH_TARGET,
    scales=ASINH_SCALES
)

y_scaler = StandardScaler()
y_train_scaled_np = y_scaler.fit_transform(y_train_trans_np)
y_val_scaled_np = y_scaler.transform(y_val_trans_np)

if np.isnan(y_train_scaled_np).any() or np.isnan(y_val_scaled_np).any():
    raise ValueError("NaNs detected after Y scaling.")

# ------------------------------------------------------------
# DataLoaders
# ------------------------------------------------------------
x_train_tensor = torch.from_numpy(x_train_scaled_np).float()
x_val_tensor = torch.from_numpy(x_val_scaled_np).float()

y_train_tensor = torch.from_numpy(y_train_scaled_np).float()
y_val_tensor = torch.from_numpy(y_val_scaled_np).float()

train_dataset = TensorDataset(
    x_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    x_val_tensor,
    y_val_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True if device.type == "cuda" else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=0,
    pin_memory=True if device.type == "cuda" else False
)

print(f"Preprocessing finished in {time.time() - start_preprocess_time:.2f}s.")
print("-" * 30)

# ============================================================
# 3. Neural network
# ============================================================
print("--- 3. Defining Neural Network ---")

class AdvancedNN(nn.Module):
    """
    Fully-connected MLP for two-target regression.

    The layer sizes are set by HIDDEN_SIZES.
    """

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        dropout_rate=0.0
    ):
        super().__init__()

        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must contain at least one hidden layer.")

        layers = []
        previous_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.LeakyReLU(0.01))

            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))

            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

model = AdvancedNN(
    INPUT_SIZE,
    HIDDEN_SIZES,
    OUTPUT_SIZE,
    dropout_rate=DROP_OUT_RATE
).to(device)

print(model)
print(f"Hidden layers: {HIDDEN_SIZES}")
print(f"Number of hidden layers: {len(HIDDEN_SIZES)}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print("-" * 30)

# ------------------------------------------------------------
# 4. Loss, optimizer, scheduler
# ------------------------------------------------------------
criterion = nn.L1Loss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=MIN_LEARNING_RATE
)

# ------------------------------------------------------------
# 5. Training
# ------------------------------------------------------------
print("--- 5. Starting Training ---")

history = {
    "train_loss": [],
    "val_loss": [],
    "val_mae_scaled_mean": [],
    "val_r2_scaled_mean": [],
    "val_mae_scaled_NF_dB": [],
    "val_mae_scaled_max_gain_dB": [],
    "val_r2_scaled_NF_dB": [],
    "val_r2_scaled_max_gain_dB": [],
    "lr": []
}

best_val_loss = float("inf")
epochs_without_improvement = 0
effective_epochs = EPOCHS
start_train_time = time.time()

for epoch in range(EPOCHS):

    model.train()

    running_train_loss = 0.0
    epoch_start_time = time.time()

    for inputs, targets in train_loader:

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n*** ERROR: NaN/Inf loss at epoch {epoch + 1}. ***")
            raise RuntimeError("Training loss became NaN or Inf")

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * inputs.size(0)

    epoch_train_loss = running_train_loss / len(train_dataset)

    history["train_loss"].append(epoch_train_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    history["lr"].append(current_lr)

    model.eval()

    running_val_loss = 0.0
    val_targets_scaled_list = []
    val_predictions_scaled_list = []

    with torch.no_grad():

        for inputs_val, targets_val in val_loader:

            inputs_val = inputs_val.to(device, non_blocking=True)
            targets_val = targets_val.to(device, non_blocking=True)

            outputs_val = model(inputs_val)

            val_loss = criterion(outputs_val, targets_val)

            running_val_loss += val_loss.item() * inputs_val.size(0)

            val_targets_scaled_list.append(targets_val.cpu().numpy())
            val_predictions_scaled_list.append(outputs_val.cpu().numpy())

    epoch_val_loss = running_val_loss / len(val_dataset)

    history["val_loss"].append(epoch_val_loss)

    val_targets_scaled_np = np.concatenate(val_targets_scaled_list, axis=0)
    val_predictions_scaled_np = np.concatenate(val_predictions_scaled_list, axis=0)

    mae_each_scaled = mean_absolute_error(
        val_targets_scaled_np,
        val_predictions_scaled_np,
        multioutput="raw_values"
    )

    r2_each_scaled = r2_score(
        val_targets_scaled_np,
        val_predictions_scaled_np,
        multioutput="raw_values"
    )

    epoch_val_mae_scaled_mean = float(np.mean(mae_each_scaled))
    epoch_val_r2_scaled_mean = float(np.mean(r2_each_scaled))

    history["val_mae_scaled_mean"].append(epoch_val_mae_scaled_mean)
    history["val_r2_scaled_mean"].append(epoch_val_r2_scaled_mean)

    history["val_mae_scaled_NF_dB"].append(mae_each_scaled[0])
    history["val_mae_scaled_max_gain_dB"].append(mae_each_scaled[1])

    history["val_r2_scaled_NF_dB"].append(r2_each_scaled[0])
    history["val_r2_scaled_max_gain_dB"].append(r2_each_scaled[1])

    epoch_duration = time.time() - epoch_start_time

    scheduler.step()

    print(
        f"Epoch [{epoch + 1:0{len(str(EPOCHS))}}/{EPOCHS}] | "
        f"Train Loss: {epoch_train_loss:.6f} | "
        f"Val Loss: {epoch_val_loss:.6f} | "
        f"Val MAE mean scaled: {epoch_val_mae_scaled_mean:.6f} | "
        f"Val R2 mean scaled: {epoch_val_r2_scaled_mean:.4f} | "
        f"LR: {current_lr:.1e} | "
        f"Duration: {epoch_duration:.2f}s",
        end=""
    )

    if epoch_val_loss < best_val_loss:

        print(" | Improved. Checkpoint saved.")

        best_val_loss = epoch_val_loss
        epochs_without_improvement = 0

        y_scaler_state = {
            "type": "standard",
            "mean": y_scaler.mean_,
            "scale": y_scaler.scale_
        }

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scheduler_type": "CosineAnnealingLR",

                "x_scaler_mean": x_scaler.mean_,
                "x_scaler_scale": x_scaler.scale_,

                "y_scaler_state": y_scaler_state,

                "loss": best_val_loss,

                "input_size": INPUT_SIZE,
                "output_size": OUTPUT_SIZE,
                "input_cols": INPUT_COLS,
                "target_cols": TARGET_COLS,

                "hidden_sizes": HIDDEN_SIZES,

                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "weight_decay": WEIGHT_DECAY,

                "nf_asinh_scale": NF_ASINH_SCALE,
                "gain_asinh_scale": GAIN_ASINH_SCALE,

                "use_asinh_target": USE_ASINH_TARGET,
                "asinh_scales": ASINH_SCALES,
                "y_scaler_type_used": Y_SCALER_TYPE
            },
            CHECKPOINT_PATH
        )

    else:

        epochs_without_improvement += 1

        print(f" | No improvement for {epochs_without_improvement} epochs.")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("\n--- Early Stopping Triggered ---")
            effective_epochs = epoch + 1
            break

print("\n--- Training Finished ---")
print(f"Total training duration: {time.time() - start_train_time:.2f}s")
print(f"Best validation loss: {best_val_loss:.6f}")
print("-" * 30)

# ------------------------------------------------------------
# 6. Load best checkpoint
# ------------------------------------------------------------
print("--- 6. Loading Best Model ---")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

loaded_hidden_sizes = checkpoint.get("hidden_sizes", HIDDEN_SIZES)

eval_model = AdvancedNN(
    checkpoint.get("input_size", INPUT_SIZE),
    loaded_hidden_sizes,
    checkpoint.get("output_size", OUTPUT_SIZE),
    dropout_rate=DROP_OUT_RATE
).to(device)

eval_model.load_state_dict(checkpoint["model_state_dict"])
eval_model.eval()

eval_x_scaler = StandardScaler()
eval_x_scaler.mean_ = checkpoint["x_scaler_mean"]
eval_x_scaler.scale_ = checkpoint["x_scaler_scale"]
eval_x_scaler.n_features_in_ = len(eval_x_scaler.mean_)

eval_y_scaler = None
y_scaler_state = checkpoint.get("y_scaler_state", None)

if y_scaler_state is not None:

    if y_scaler_state["type"] == "standard":

        eval_y_scaler = StandardScaler()
        eval_y_scaler.mean_ = y_scaler_state["mean"]
        eval_y_scaler.scale_ = y_scaler_state["scale"]
        eval_y_scaler.n_features_in_ = len(eval_y_scaler.mean_)

use_asinh_loaded = checkpoint.get("use_asinh_target", False)
asinh_scales_loaded = checkpoint.get("asinh_scales", ASINH_SCALES)

print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
print(f"Loaded hidden sizes     : {loaded_hidden_sizes}")
print(f"Learning rate           : {checkpoint.get('learning_rate', LEARNING_RATE)}")
print(f"Batch size              : {checkpoint.get('batch_size', BATCH_SIZE)}")
print(f"Weight decay            : {checkpoint.get('weight_decay', WEIGHT_DECAY)}")
print(f"NF ASINH scale          : {checkpoint.get('nf_asinh_scale', NF_ASINH_SCALE)}")
print(f"Gain ASINH scale        : {checkpoint.get('gain_asinh_scale', GAIN_ASINH_SCALE)}")
print(f"LR scheduler            : {checkpoint.get('scheduler_type', 'CosineAnnealingLR')}")
print("-" * 30)

# ------------------------------------------------------------
# Prediction helper
# ------------------------------------------------------------
def predict_original_scale(model, x_np, batch_size=512):

    x_scaled_np = eval_x_scaler.transform(x_np)

    x_tensor = torch.from_numpy(x_scaled_np).float().to(device)

    dataset = TensorDataset(x_tensor)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    pred_scaled_list = []

    model.eval()

    with torch.no_grad():

        for (inputs_batch,) in loader:

            outputs_scaled = model(inputs_batch)

            pred_scaled_list.append(outputs_scaled.cpu().numpy())

    pred_scaled_np = np.concatenate(pred_scaled_list, axis=0)

    if eval_y_scaler is not None:
        pred_trans_np = eval_y_scaler.inverse_transform(pred_scaled_np)
    else:
        pred_trans_np = pred_scaled_np

    pred_np = inverse_transform_targets(
        pred_trans_np,
        use_asinh=use_asinh_loaded,
        scales=asinh_scales_loaded
    )

    return pred_np

# ------------------------------------------------------------
# Metrics helper
# ------------------------------------------------------------
def print_metrics(y_true_np, y_pred_np, dataset_name):

    print(f"\n--- {dataset_name} Metrics: Original Scale ---")

    for k, target_col in enumerate(TARGET_COLS):

        y_true = y_true_np[:, k]
        y_pred = y_pred_np[:, k]

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        medae = median_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        safe_y_true = np.where(y_true == 0, 1e-9, y_true)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            mape = mean_absolute_percentage_error(safe_y_true, y_pred) * 100

        print(f"\nTarget: {target_col}")
        print(f"  MSE:   {mse:.6f}")
        print(f"  RMSE:  {rmse:.6f}")
        print(f"  MAE:   {mae:.6f}")
        print(f"  MedAE: {medae:.6f}")
        print(f"  R2:    {r2:.4f}")
        print(f"  MAPE:  {mape:.4f}%")

# ------------------------------------------------------------
# 7. Evaluate validation set
# ------------------------------------------------------------
print("--- 7. Evaluating Validation Set ---")

preds_val_np = predict_original_scale(
    eval_model,
    x_val_np,
    batch_size=BATCH_SIZE * 2
)

print_metrics(y_val_orig_np, preds_val_np, "Validation")

# ============================================================
# 8. Evaluate external test CSV
# ============================================================

external_test_available = False
x_new_np = None
y_new_np = None
y_pred_new_np = None
df_results_new = None

if os.path.exists(NEW_DATA_PATH):

    external_test_available = True

    print("\n--- External Test Evaluation on New CSV ---")

    df_new = pd.read_csv(NEW_DATA_PATH)

    print(f"New dataset shape: {df_new.shape}")

    required_cols_new = INPUT_COLS + TARGET_COLS

    missing_cols_new = [
        col for col in required_cols_new
        if col not in df_new.columns
    ]

    if missing_cols_new:

        print("*** ERROR: Missing columns in new CSV ***")
        print(missing_cols_new)

        external_test_available = False

    else:

        df_new_model = df_new[required_cols_new].copy().dropna()

        print(f"Usable new samples: {len(df_new_model)}")
        if df_new_model.empty:
            raise ValueError("No usable rows in external CSV")

        x_new_np = df_new_model[INPUT_COLS].values.astype(np.float32)
        y_new_np = df_new_model[TARGET_COLS].values.astype(np.float32)

        # ============================================================
        # Fast single-sample inference time
        # ============================================================

        sample_index = 282

        if sample_index >= len(x_new_np):
            sample_index = 0
            print("Warning: sample_index was larger than external data size. Using sample_index = 0.")

        x_one_np = x_new_np[sample_index:sample_index + 1].astype(np.float32)

        def predict_one_sample_fast(x_one_np):

            x_scaled_np = eval_x_scaler.transform(x_one_np)

            x_tensor = torch.from_numpy(x_scaled_np).float().to(device)

            with torch.no_grad():
                y_scaled_tensor = eval_model(x_tensor)

            y_scaled_np = y_scaled_tensor.cpu().numpy()

            if eval_y_scaler is not None:
                y_trans_np = eval_y_scaler.inverse_transform(y_scaled_np)
            else:
                y_trans_np = y_scaled_np

            y_pred_np = inverse_transform_targets(
                y_trans_np,
                use_asinh=use_asinh_loaded,
                scales=asinh_scales_loaded
            )

            return y_pred_np

        for _ in range(20):
            _ = predict_one_sample_fast(x_one_np)

        if device.type == "cuda":
            torch.cuda.synchronize()

        num_repeats = 1000

        start_time = time.perf_counter()

        for _ in range(num_repeats):
            y_one_pred_np = predict_one_sample_fast(x_one_np)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        avg_time_s = (end_time - start_time) / num_repeats

        pred_nf = y_one_pred_np[0, TARGET_COLS.index("NF_dB")]
        pred_gain = y_one_pred_np[0, TARGET_COLS.index("max_gain_dB")]

        print("\n--- Fast Single-Sample ML Prediction Time ---")
        print(f"Device used        : {device}")
        print(f"Input sample index : {sample_index}")
        print(f"Predicted NF       : {pred_nf:.6f} dB")
        print(f"Predicted Max gain : {pred_gain:.6f} dB")
        print(f"Average time/sample: {avg_time_s:.9f} s")
        print(f"Average time/sample: {avg_time_s * 1e3:.6f} ms")
        print(f"Average time/sample: {avg_time_s * 1e6:.3f} µs")

        with open("inference_time_results.txt", "w") as f:
            f.write(f"Device used        : {device}\n")
            f.write(f"Input sample index : {sample_index}\n")
            f.write(f"Predicted NF       : {pred_nf:.6f} dB\n")
            f.write(f"Predicted Max gain : {pred_gain:.6f} dB\n")
            f.write(f"Average time/sample: {avg_time_s:.9f} s\n")
            f.write(f"Average time/sample: {avg_time_s * 1e3:.6f} ms\n")
            f.write(f"Average time/sample: {avg_time_s * 1e6:.3f} µs\n")

        gain_idx = TARGET_COLS.index("max_gain_dB")

        y_new_np[:, gain_idx] = np.maximum(
            y_new_np[:, gain_idx],
            0.0
        )

        y_pred_new_np = predict_original_scale(
            eval_model,
            x_new_np,
            batch_size=BATCH_SIZE * 2
        )

        print("\n--- External Test Metrics: Original Scale ---")

        df_results_new = df_new_model.copy()

        for k, target_col in enumerate(TARGET_COLS):

            y_true_new = y_new_np[:, k]
            y_pred_new = y_pred_new_np[:, k]

            mse_new = mean_squared_error(y_true_new, y_pred_new)
            rmse_new = np.sqrt(mse_new)
            mae_new = mean_absolute_error(y_true_new, y_pred_new)
            medae_new = median_absolute_error(y_true_new, y_pred_new)
            r2_new = r2_score(y_true_new, y_pred_new)

            safe_y_true_new = np.where(y_true_new == 0, 1e-9, y_true_new)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                mape_new = mean_absolute_percentage_error(
                    safe_y_true_new,
                    y_pred_new
                ) * 100

            print(f"\nTarget: {target_col}")
            print(f"MSE:   {mse_new:.6f}")
            print(f"RMSE:  {rmse_new:.6f}")
            print(f"MAE:   {mae_new:.6f}")
            print(f"MedAE: {medae_new:.6f}")
            print(f"R2:    {r2_new:.4f}")
            print(f"MAPE:  {mape_new:.4f}%")

            df_results_new[f"predicted_{target_col}"] = y_pred_new

            df_results_new[f"absolute_error_{target_col}"] = np.abs(
                y_true_new - y_pred_new
            )

            df_results_new[f"relative_error_percent_{target_col}"] = (
                np.abs(y_true_new - y_pred_new)
                / np.maximum(np.abs(y_true_new), 1e-9)
            ) * 100

        OUTPUT_PREDICTIONS_PATH = "external_test_predictions_NF_and_gain.csv"

        df_results_new.to_csv(
            OUTPUT_PREDICTIONS_PATH,
            index=False
        )

        print(f"\nPredictions saved to: {OUTPUT_PREDICTIONS_PATH}")

else:

    print(f"\nExternal test file not found: {NEW_DATA_PATH}")

# ============================================================
# 9. Plots
# Save all plots as PDF, PNG, and MAT
# ============================================================

print("\n--- 9. Generating and Saving Plots ---")

epochs_range = range(1, effective_epochs + 1)
epochs_np = np.array(list(epochs_range)).reshape(-1, 1)

best_epoch_num = int(
    np.argmin(history["val_loss"][:effective_epochs]) + 1
)

train_loss_np = np.array(
    history["train_loss"][:effective_epochs]
).reshape(-1, 1)

val_loss_np = np.array(
    history["val_loss"][:effective_epochs]
).reshape(-1, 1)

val_mae_nf_np = np.array(
    history["val_mae_scaled_NF_dB"][:effective_epochs]
).reshape(-1, 1)

val_mae_gain_np = np.array(
    history["val_mae_scaled_max_gain_dB"][:effective_epochs]
).reshape(-1, 1)

val_r2_nf_np = np.array(
    history["val_r2_scaled_NF_dB"][:effective_epochs]
).reshape(-1, 1)

val_r2_gain_np = np.array(
    history["val_r2_scaled_max_gain_dB"][:effective_epochs]
).reshape(-1, 1)

# ------------------------------------------------------------
# Plot 1: Loss curves
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    epochs_range,
    history["train_loss"][:effective_epochs],
    label="Training Loss",
    marker="."
)

plt.plot(
    epochs_range,
    history["val_loss"][:effective_epochs],
    label="Validation Loss",
    marker="."
)

plt.axvline(
    best_epoch_num,
    linestyle="--",
    label=f"Best Epoch {best_epoch_num}"
)

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss for NF_dB and max_gain_dB")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

save_current_figure(
    figure_name="plot_01_training_validation_loss_curves",
    mat_data={
        "epochs": epochs_np,
        "train_loss": train_loss_np,
        "val_loss": val_loss_np,
        "best_epoch": np.array([[best_epoch_num]])
    }
)

plt.show()

# ------------------------------------------------------------
# Plot 2: Validation MAE on scaled targets
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    epochs_range,
    history["val_mae_scaled_NF_dB"][:effective_epochs],
    label="Validation MAE scaled NF_dB",
    marker="."
)

plt.plot(
    epochs_range,
    history["val_mae_scaled_max_gain_dB"][:effective_epochs],
    label="Validation MAE scaled max_gain_dB",
    marker="."
)

plt.axvline(
    best_epoch_num,
    linestyle="--",
    label=f"Best Epoch {best_epoch_num}"
)

plt.xlabel("Epoch")
plt.ylabel("MAE scaled")
plt.title("Validation MAE on Scaled Targets")
plt.grid(True)
plt.legend()
plt.tight_layout()

save_current_figure(
    figure_name="plot_02_validation_mae_scaled_targets",
    mat_data={
        "epochs": epochs_np,
        "val_mae_scaled_NF_dB": val_mae_nf_np,
        "val_mae_scaled_max_gain_dB": val_mae_gain_np,
        "best_epoch": np.array([[best_epoch_num]])
    }
)

plt.show()

# ------------------------------------------------------------
# Plot 3: Validation R2 on scaled targets
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    epochs_range,
    history["val_r2_scaled_NF_dB"][:effective_epochs],
    label="Validation R2 scaled NF_dB",
    marker="."
)

plt.plot(
    epochs_range,
    history["val_r2_scaled_max_gain_dB"][:effective_epochs],
    label="Validation R2 scaled max_gain_dB",
    marker="."
)

plt.axvline(
    best_epoch_num,
    linestyle="--",
    label=f"Best Epoch {best_epoch_num}"
)

plt.xlabel("Epoch")
plt.ylabel("R2 scaled")
plt.title("Validation R2 on Scaled Targets")
plt.grid(True)
plt.legend()
plt.tight_layout()

save_current_figure(
    figure_name="plot_03_validation_r2_scaled_targets",
    mat_data={
        "epochs": epochs_np,
        "val_r2_scaled_NF_dB": val_r2_nf_np,
        "val_r2_scaled_max_gain_dB": val_r2_gain_np,
        "best_epoch": np.array([[best_epoch_num]])
    }
)

plt.show()

# ------------------------------------------------------------
# Validation target-specific plots
# ------------------------------------------------------------

DISPLAY_NAMES = {
    "NF_dB": "NF [dB]",
    "max_gain_dB": "Max gain [dB]"
}

for k, target_col in enumerate(TARGET_COLS):

    display_name = DISPLAY_NAMES[target_col]

    safe_target = target_col.replace("-", "_").replace(" ", "_")

    y_true = y_val_orig_np[:, k]
    y_pred = preds_val_np[:, k]

    prediction_error = y_true - y_pred
    absolute_error = np.abs(prediction_error)

    mse_val = mean_squared_error(y_true, y_pred)
    rmse_val = np.sqrt(mse_val)
    mae_val = mean_absolute_error(y_true, y_pred)
    medae_val = median_absolute_error(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)

    safe_y_true = np.where(y_true == 0, 1e-9, y_true)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mape_val = mean_absolute_percentage_error(safe_y_true, y_pred) * 100

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    buffer = 0.05 * (max_val - min_val) + 1e-6

    plot_min = min_val - buffer
    plot_max = max_val + buffer

    # ========================================================
    # Plot 4: Validation Predicted vs Actual
    # ========================================================

    plt.figure(figsize=(7, 7))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.4,
        s=12,
        edgecolors="k",
        linewidths=0.3
    )

    plt.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        "r--",
        label="Ideal fit"
    )

    plt.xlabel(f"Actual {display_name}")
    plt.ylabel(f"Predicted {display_name}")
    plt.title(f"Validation: Predicted vs Actual {display_name}")

    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)

    plt.gca().set_aspect("equal", adjustable="box")

    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()

    save_current_figure(
        figure_name=f"plot_04_validation_pred_vs_actual_{safe_target}",
        mat_data={
            "y_true": y_true.reshape(-1, 1),
            "y_pred": y_pred.reshape(-1, 1),
            "prediction_error": prediction_error.reshape(-1, 1),
            "absolute_error": absolute_error.reshape(-1, 1),
            "plot_min": np.array([[plot_min]]),
            "plot_max": np.array([[plot_max]]),
            "mse": np.array([[mse_val]]),
            "rmse": np.array([[rmse_val]]),
            "mae": np.array([[mae_val]]),
            "medae": np.array([[medae_val]]),
            "r2": np.array([[r2_val]]),
            "mape": np.array([[mape_val]])
        }
    )

    plt.show()

    # ========================================================
    # Plot 5: Validation Absolute Error vs Actual
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.scatter(
        y_true,
        absolute_error,
        alpha=0.5,
        s=18,
        edgecolors="k",
        linewidths=0.3
    )

    plt.xlabel(f"Actual {display_name}")
    plt.ylabel("Absolute Error")
    plt.title(f"Validation: Absolute Error vs Actual {display_name}")
    plt.grid(True, linestyle=":")
    plt.tight_layout()

    save_current_figure(
        figure_name=f"plot_05_validation_abs_error_vs_actual_{safe_target}",
        mat_data={
            "y_true": y_true.reshape(-1, 1),
            "absolute_error": absolute_error.reshape(-1, 1),
            "prediction_error": prediction_error.reshape(-1, 1),
            "mse": np.array([[mse_val]]),
            "rmse": np.array([[rmse_val]]),
            "mae": np.array([[mae_val]]),
            "medae": np.array([[medae_val]]),
            "r2": np.array([[r2_val]]),
            "mape": np.array([[mape_val]])
        }
    )

    plt.show()

# ============================================================
# External test plots
# ============================================================

if external_test_available and y_new_np is not None and y_pred_new_np is not None:

    print("\n--- Saving External Test Plots ---")

    for k, target_col in enumerate(TARGET_COLS):

        display_name = DISPLAY_NAMES[target_col]

        safe_target = target_col.replace("-", "_").replace(" ", "_")

        y_true_new = y_new_np[:, k]
        y_pred_new = y_pred_new_np[:, k]

        prediction_error_new = y_true_new - y_pred_new
        absolute_error_new = np.abs(prediction_error_new)

        mse_new = mean_squared_error(y_true_new, y_pred_new)
        rmse_new = np.sqrt(mse_new)
        mae_new = mean_absolute_error(y_true_new, y_pred_new)
        medae_new = median_absolute_error(y_true_new, y_pred_new)
        r2_new = r2_score(y_true_new, y_pred_new)

        safe_y_true_new = np.where(y_true_new == 0, 1e-9, y_true_new)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            mape_new = mean_absolute_percentage_error(safe_y_true_new, y_pred_new) * 100

        min_val = min(y_true_new.min(), y_pred_new.min())
        max_val = max(y_true_new.max(), y_pred_new.max())

        buffer = 0.05 * (max_val - min_val) + 1e-6

        plot_min = min_val - buffer
        plot_max = max_val + buffer

        # ====================================================
        # External Plot 1: Predicted vs Actual
        # ====================================================

        plt.figure(figsize=(8, 5))

        plt.scatter(
            y_true_new,
            y_pred_new,
            alpha=0.5,
            s=18,
            edgecolors="k",
            linewidths=0.3
        )

        plt.plot(
            [plot_min, plot_max],
            [plot_min, plot_max],
            "r--",
            label="Ideal fit"
        )

        plt.xlabel(f"Actual {display_name}", fontsize=20)
        plt.ylabel(f"Predicted {display_name}", fontsize=20)

        plt.xlim(plot_min, plot_max)
        plt.ylim(plot_min, plot_max)

        plt.gca().set_aspect("equal", adjustable="box")

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.grid(True, linestyle=":")
        plt.legend(fontsize=18)
        plt.tight_layout()

        save_current_figure(
            figure_name=f"plot_06_external_pred_vs_actual_{safe_target}",
            mat_data={
                "y_true": y_true_new.reshape(-1, 1),
                "y_pred": y_pred_new.reshape(-1, 1),
                "prediction_error": prediction_error_new.reshape(-1, 1),
                "absolute_error": absolute_error_new.reshape(-1, 1),
                "plot_min": np.array([[plot_min]]),
                "plot_max": np.array([[plot_max]]),
                "mse": np.array([[mse_new]]),
                "rmse": np.array([[rmse_new]]),
                "mae": np.array([[mae_new]]),
                "medae": np.array([[medae_new]]),
                "r2": np.array([[r2_new]]),
                "mape": np.array([[mape_new]])
            }
        )

        plt.show()

        # ====================================================
        # External histogram plot
        # Shown only, NOT saved
        # ====================================================

        plt.figure(figsize=(8, 5))

        plt.hist(
            prediction_error_new,
            bins=40,
            edgecolor="black",
            alpha=0.7
        )

        plt.xlabel(f"Prediction Error: Actual - Predicted {target_col}")
        plt.ylabel("Count")
        plt.title(f"External Test Error Histogram for {target_col}")
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.show()

        # ====================================================
        # External Plot 2: Absolute Error vs Actual
        # ====================================================

        plt.figure(figsize=(8, 5))

        plt.scatter(
            y_true_new,
            absolute_error_new,
            alpha=0.5,
            s=18,
            edgecolors="k",
            linewidths=0.3
        )

        plt.xlabel(f"Actual {display_name}", fontsize=18)
        plt.ylabel("Absolute Error", fontsize=18)
        plt.title(f"External Test Absolute Error vs Actual {display_name}")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, linestyle=":")
        plt.tight_layout()

        save_current_figure(
            figure_name=f"plot_07_external_abs_error_vs_actual_{safe_target}",
            mat_data={
                "y_true": y_true_new.reshape(-1, 1),
                "absolute_error": absolute_error_new.reshape(-1, 1),
                "prediction_error": prediction_error_new.reshape(-1, 1),
                "mse": np.array([[mse_new]]),
                "rmse": np.array([[rmse_new]]),
                "mae": np.array([[mae_new]]),
                "medae": np.array([[medae_new]]),
                "r2": np.array([[r2_new]]),
                "mape": np.array([[mape_new]])
            }
        )

        plt.show()

# ------------------------------------------------------------
# Save complete MATLAB summary file
# ------------------------------------------------------------

summary_data = {
    "epochs": epochs_np,

    "train_loss": train_loss_np,
    "val_loss": val_loss_np,

    "val_mae_scaled_NF_dB": val_mae_nf_np,
    "val_mae_scaled_max_gain_dB": val_mae_gain_np,

    "val_r2_scaled_NF_dB": val_r2_nf_np,
    "val_r2_scaled_max_gain_dB": val_r2_gain_np,

    "best_epoch": np.array([[best_epoch_num]]),

    "x_val": x_val_np,
    "y_val_true": y_val_orig_np,
    "y_val_pred": preds_val_np
}

if external_test_available and x_new_np is not None and y_new_np is not None and y_pred_new_np is not None:
    summary_data["x_external"] = x_new_np
    summary_data["y_external_true"] = y_new_np
    summary_data["y_external_pred"] = y_pred_new_np

savemat(
    os.path.join(MAT_FOLDER, "summary_regression_non_histogram_plots.mat"),
    summary_data
)

print("\nSaved all plot PDFs in:", PDF_FOLDER)
print("Saved all plot PNGs in:", PNG_FOLDER)
print("Saved all plot MATLAB data in:", MAT_FOLDER)

print("\n--- Script Finished Successfully ---")
