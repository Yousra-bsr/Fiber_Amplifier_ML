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
- optionally applies an ASINH transform to the targets;
- trains a configurable fully connected PyTorch MLP;
- saves the best checkpoint;
- evaluates a validation split;
- optionally evaluates an external CSV;
- saves non-histogram plots as PDF and PNG;
- saves the data behind non-histogram plots as MATLAB .mat files.

Example
-------
python src/DNN_model_I.py

or

python src/DNN_model_I.py \
    --data-path data/dataset_200000.csv \
    --external-data-path data/dataset_ext_20000.csv
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import savemat
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "dataset_200000.csv"
)
DEFAULT_EXTERNAL_DATA_PATH = (
    PROJECT_ROOT / "data" / "dataset_ext_20000.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CHECKPOINT_NAME = "best_model_NF_and_gain_inoue.pth"


# =============================================================================
# Model/data configuration
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

DISPLAY_NAMES = {
    "NF_dB": "NF [dB]",
    "max_gain_dB": "Max gain [dB]",
}

INPUT_SIZE = len(INPUT_COLS)
OUTPUT_SIZE = len(TARGET_COLS)

HIDDEN_SIZES = [512, 512, 512, 256]
DROPOUT_RATE = 0.0

LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 500
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 60
NUM_SAMPLES_TO_SHOW = 5
MIN_LEARNING_RATE = 1e-6

Y_SCALER_TYPE = "standard"

USE_ASINH_TARGET = True
NF_ASINH_SCALE = 0.002
GAIN_ASINH_SCALE = 0.5

ASINH_SCALES = np.array(
    [NF_ASINH_SCALE, GAIN_ASINH_SCALE],
    dtype=np.float32,
).reshape(1, -1)


# =============================================================================
# Command-line arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the four-input / two-target regression model."
        )
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Training CSV path. Default: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--external-data-path",
        type=Path,
        default=DEFAULT_EXTERNAL_DATA_PATH,
        help=(
            "Optional external-test CSV path. "
            f"Default: {DEFAULT_EXTERNAL_DATA_PATH}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help=(
            "Path for the best model checkpoint. "
            "Default: <output-dir>/checkpoints/"
            f"{DEFAULT_CHECKPOINT_NAME}"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Maximum number of training epochs. Default: {EPOCHS}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Training batch size. Default: {BATCH_SIZE}",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=282,
        help="External-data row used for single-sample timing. Default: 282",
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=1000,
        help="Number of repetitions for inference timing. Default: 1000",
    )

    return parser.parse_args()


# =============================================================================
# Utility helpers
# =============================================================================

def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(
            "CUDA is available. "
            f"Using device: {torch.cuda.get_device_name(0)}"
        )
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    return device


def set_random_seed(seed: int) -> None:
    """Set NumPy and PyTorch random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_output_directories(output_dir: Path) -> dict[str, Path]:
    """Create output directories and return their paths."""
    folders = {
        "root": output_dir,
        "pdf": output_dir / "figures" / "pdf",
        "png": output_dir / "figures" / "png",
        "mat": output_dir / "matlab",
        "checkpoints": output_dir / "checkpoints",
        "predictions": output_dir / "predictions",
        "timing": output_dir / "timing",
    }

    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    return folders


def save_current_figure(
    figure_name: str,
    folders: dict[str, Path],
    mat_data: dict[str, np.ndarray] | None = None,
) -> None:
    """
    Save the current Matplotlib figure as PDF and PNG.

    If mat_data is provided, also save the numerical data required to
    reproduce the figure as a MATLAB .mat file.
    """
    pdf_path = folders["pdf"] / f"{figure_name}.pdf"
    png_path = folders["png"] / f"{figure_name}.png"

    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")

    if mat_data is not None:
        mat_path = folders["mat"] / f"{figure_name}.mat"
        savemat(mat_path, mat_data)
        print(f"Saved MAT: {mat_path}")


def transform_targets(
    y_np: np.ndarray,
    use_asinh: bool = True,
    scales: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the ASINH transform column-wise to the target array."""
    y_np = y_np.astype(np.float32).copy()

    if not use_asinh:
        return y_np

    if y_np.ndim != 2:
        raise ValueError(
            f"Expected target shape [N, num_targets], got {y_np.shape}"
        )

    if scales is None:
        raise ValueError(
            "scales must be provided when using the ASINH transform."
        )

    return np.arcsinh(y_np / scales).astype(np.float32)


def inverse_transform_targets(
    y_trans_np: np.ndarray,
    use_asinh: bool = True,
    scales: np.ndarray | None = None,
) -> np.ndarray:
    """Invert the column-wise ASINH transform."""
    y_trans_np = y_trans_np.astype(np.float32).copy()

    if not use_asinh:
        return y_trans_np

    if y_trans_np.ndim != 2:
        raise ValueError(
            f"Expected target shape [N, num_targets], got {y_trans_np.shape}"
        )

    if scales is None:
        raise ValueError(
            "scales must be provided when using the inverse ASINH transform."
        )

    return (scales * np.sinh(y_trans_np)).astype(np.float32)


def build_scaler_state(
    scaler: StandardScaler | MinMaxScaler | None,
) -> dict | None:
    """Convert a fitted target scaler into checkpoint-friendly state."""
    if scaler is None:
        return None

    if isinstance(scaler, StandardScaler):
        return {
            "type": "standard",
            "mean": scaler.mean_,
            "scale": scaler.scale_,
        }

    if isinstance(scaler, MinMaxScaler):
        return {
            "type": "minmax",
            "min_": scaler.min_,
            "scale_": scaler.scale_,
            "data_min_": scaler.data_min_,
            "data_max_": scaler.data_max_,
            "data_range_": scaler.data_range_,
            "feature_range": scaler.feature_range,
        }

    raise TypeError(f"Unsupported scaler type: {type(scaler).__name__}")


def restore_target_scaler(
    scaler_state: dict | None,
) -> StandardScaler | MinMaxScaler | None:
    """Restore a target scaler from checkpoint state."""
    if scaler_state is None:
        return None

    scaler_type = scaler_state["type"]

    if scaler_type == "standard":
        scaler = StandardScaler()
        scaler.mean_ = scaler_state["mean"]
        scaler.scale_ = scaler_state["scale"]
        scaler.n_features_in_ = len(scaler.mean_)
        return scaler

    if scaler_type == "minmax":
        scaler = MinMaxScaler(feature_range=scaler_state["feature_range"])
        scaler.min_ = scaler_state["min_"]
        scaler.scale_ = scaler_state["scale_"]
        scaler.data_min_ = scaler_state["data_min_"]
        scaler.data_max_ = scaler_state["data_max_"]
        scaler.data_range_ = scaler_state["data_range_"]
        scaler.n_features_in_ = len(scaler.min_)
        return scaler

    raise ValueError(f"Unsupported saved scaler type: {scaler_type}")


# =============================================================================
# Data loading and preprocessing
# =============================================================================

def load_csv(path: Path, label: str) -> pd.DataFrame:
    """Load a CSV file and report basic information."""
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")

    df = pd.read_csv(path)

    print(f"{label} loaded successfully.")
    print(f"{label} shape: {df.shape}")
    print("Columns found:")
    print(df.columns.tolist())

    return df


def select_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns, drop rows containing missing values."""
    required_cols = INPUT_COLS + TARGET_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df_model = df[required_cols].copy()

    n_missing_rows = int(df_model.isnull().any(axis=1).sum())
    if n_missing_rows > 0:
        print(
            f"Warning: {n_missing_rows} rows contain missing values "
            "and will be dropped."
        )
        df_model = df_model.dropna()

    if df_model.empty:
        raise ValueError("No usable rows remain after preprocessing.")

    return df_model


def dataframe_to_arrays(
    df_model: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the selected dataframe columns to float32 NumPy arrays."""
    x_np = df_model[INPUT_COLS].to_numpy(dtype=np.float32)
    y_np = df_model[TARGET_COLS].to_numpy(dtype=np.float32)

    gain_index = TARGET_COLS.index("max_gain_dB")
    y_np[:, gain_index] = np.maximum(y_np[:, gain_index], 0.0)

    return x_np, y_np


def show_sample_rows(
    x_np: np.ndarray,
    y_np: np.ndarray,
    n_samples: int = NUM_SAMPLES_TO_SHOW,
) -> None:
    """Print a few raw input/target samples."""
    print(f"\n--- Sample Raw Data: first {n_samples} samples ---")

    for i in range(min(n_samples, len(x_np))):
        input_values = ", ".join(
            f"{INPUT_COLS[j]}={x_np[i, j]:.6f}"
            for j in range(INPUT_SIZE)
        )
        target_values = ", ".join(
            f"{TARGET_COLS[k]}={y_np[i, k]:.6f}"
            for k in range(OUTPUT_SIZE)
        )
        print(f"Sample {i + 1}: {input_values}, {target_values}")


def show_target_distributions(
    df_model: pd.DataFrame,
    y_np: np.ndarray,
) -> None:
    """
    Show target histograms.

    Histograms are intentionally displayed only and are not saved.
    """
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
            alpha=0.7,
        )
        plt.title(f"Histogram of {target_col}")
        plt.xlabel(target_col)
        plt.ylabel("Count")
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.show()
        plt.close()

    if USE_ASINH_TARGET:
        y_asinh_all = transform_targets(
            y_np,
            use_asinh=True,
            scales=ASINH_SCALES,
        )

        for k, target_col in enumerate(TARGET_COLS):
            plt.figure(figsize=(8, 5))
            plt.hist(
                y_asinh_all[:, k],
                bins=40,
                edgecolor="black",
                alpha=0.7,
            )
            plt.title(f"Histogram of ASINH-transformed {target_col}")
            plt.xlabel(
                f"asinh({target_col} / {ASINH_SCALES[0, k]})"
            )
            plt.ylabel("Count")
            plt.grid(True, linestyle=":")
            plt.tight_layout()
            plt.show()
            plt.close()


def split_and_scale_data(
    x_np: np.ndarray,
    y_np: np.ndarray,
) -> dict:
    """Split data, transform targets, fit scalers, and return arrays/scalers."""
    all_indices = np.arange(len(x_np))

    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    x_train_np = x_np[train_indices]
    x_val_np = x_np[val_indices]
    y_train_np = y_np[train_indices]
    y_val_np = y_np[val_indices]

    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")

    x_scaler = StandardScaler()
    x_train_scaled_np = x_scaler.fit_transform(x_train_np).astype(np.float32)
    x_val_scaled_np = x_scaler.transform(x_val_np).astype(np.float32)

    if (
        np.isnan(x_train_scaled_np).any()
        or np.isnan(x_val_scaled_np).any()
    ):
        raise ValueError("NaNs detected after X scaling.")

    y_train_trans_np = transform_targets(
        y_train_np,
        use_asinh=USE_ASINH_TARGET,
        scales=ASINH_SCALES,
    )
    y_val_trans_np = transform_targets(
        y_val_np,
        use_asinh=USE_ASINH_TARGET,
        scales=ASINH_SCALES,
    )

    y_scaler: StandardScaler | MinMaxScaler | None

    if Y_SCALER_TYPE == "standard":
        y_scaler = StandardScaler()
        y_train_scaled_np = y_scaler.fit_transform(
            y_train_trans_np
        ).astype(np.float32)
        y_val_scaled_np = y_scaler.transform(
            y_val_trans_np
        ).astype(np.float32)

    elif Y_SCALER_TYPE == "minmax":
        y_scaler = MinMaxScaler(feature_range=(-1, 1))
        y_train_scaled_np = y_scaler.fit_transform(
            y_train_trans_np
        ).astype(np.float32)
        y_val_scaled_np = y_scaler.transform(
            y_val_trans_np
        ).astype(np.float32)

    elif Y_SCALER_TYPE in {"none", None}:
        y_scaler = None
        y_train_scaled_np = y_train_trans_np.astype(np.float32)
        y_val_scaled_np = y_val_trans_np.astype(np.float32)

    else:
        raise ValueError(
            "Y_SCALER_TYPE must be 'standard', 'minmax', or 'none'."
        )

    if (
        np.isnan(y_train_scaled_np).any()
        or np.isnan(y_val_scaled_np).any()
    ):
        raise ValueError("NaNs detected after Y scaling.")

    return {
        "x_train": x_train_np,
        "x_val": x_val_np,
        "y_train": y_train_np,
        "y_val": y_val_np,
        "x_train_scaled": x_train_scaled_np,
        "x_val_scaled": x_val_scaled_np,
        "y_train_scaled": y_train_scaled_np,
        "y_val_scaled": y_val_scaled_np,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }


def create_data_loaders(
    split_data: dict,
    batch_size: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    x_train_tensor = torch.from_numpy(
        split_data["x_train_scaled"]
    ).float()
    y_train_tensor = torch.from_numpy(
        split_data["y_train_scaled"]
    ).float()

    x_val_tensor = torch.from_numpy(
        split_data["x_val_scaled"]
    ).float()
    y_val_tensor = torch.from_numpy(
        split_data["y_val_scaled"]
    ).float()

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


# =============================================================================
# Neural network
# =============================================================================

class MultiTargetRegressor(nn.Module):
    """Fully connected MLP for NF and maximum-gain regression."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer.")

        layers: list[nn.Module] = []
        previous_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.LeakyReLU(0.01))

            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))

            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# =============================================================================
# Training and checkpointing
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    split_data: dict,
    device: torch.device,
    checkpoint_path: Path,
    epochs: int,
) -> tuple[dict[str, list], int, float]:
    """Train the network and save the best validation-loss checkpoint."""
    criterion = nn.L1Loss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=MIN_LEARNING_RATE,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae_scaled_mean": [],
        "val_r2_scaled_mean": [],
        "val_mae_scaled_NF_dB": [],
        "val_mae_scaled_max_gain_dB": [],
        "val_r2_scaled_NF_dB": [],
        "val_r2_scaled_max_gain_dB": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    effective_epochs = epochs

    print("\n--- Starting Training ---")
    start_train_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training
        model.train()
        running_train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(
                    f"NaN/Inf loss detected at epoch {epoch + 1}."
                )

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = (
            running_train_loss / len(train_loader.dataset)
        )
        history["train_loss"].append(epoch_train_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        # Validation
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

                running_val_loss += (
                    val_loss.item() * inputs_val.size(0)
                )
                val_targets_scaled_list.append(
                    targets_val.cpu().numpy()
                )
                val_predictions_scaled_list.append(
                    outputs_val.cpu().numpy()
                )

        epoch_val_loss = (
            running_val_loss / len(val_loader.dataset)
        )
        history["val_loss"].append(epoch_val_loss)

        val_targets_scaled_np = np.concatenate(
            val_targets_scaled_list,
            axis=0,
        )
        val_predictions_scaled_np = np.concatenate(
            val_predictions_scaled_list,
            axis=0,
        )

        mae_each_scaled = mean_absolute_error(
            val_targets_scaled_np,
            val_predictions_scaled_np,
            multioutput="raw_values",
        )
        r2_each_scaled = r2_score(
            val_targets_scaled_np,
            val_predictions_scaled_np,
            multioutput="raw_values",
        )

        epoch_val_mae_scaled_mean = float(np.mean(mae_each_scaled))
        epoch_val_r2_scaled_mean = float(np.mean(r2_each_scaled))

        history["val_mae_scaled_mean"].append(
            epoch_val_mae_scaled_mean
        )
        history["val_r2_scaled_mean"].append(
            epoch_val_r2_scaled_mean
        )
        history["val_mae_scaled_NF_dB"].append(
            float(mae_each_scaled[0])
        )
        history["val_mae_scaled_max_gain_dB"].append(
            float(mae_each_scaled[1])
        )
        history["val_r2_scaled_NF_dB"].append(
            float(r2_each_scaled[0])
        )
        history["val_r2_scaled_max_gain_dB"].append(
            float(r2_each_scaled[1])
        )

        epoch_duration = time.time() - epoch_start_time

        scheduler.step()

        print(
            f"Epoch [{epoch + 1:0{len(str(epochs))}}/{epochs}] | "
            f"Train Loss: {epoch_train_loss:.6f} | "
            f"Val Loss: {epoch_val_loss:.6f} | "
            f"Val MAE mean scaled: "
            f"{epoch_val_mae_scaled_mean:.6f} | "
            f"Val R2 mean scaled: "
            f"{epoch_val_r2_scaled_mean:.4f} | "
            f"LR: {current_lr:.1e} | "
            f"Duration: {epoch_duration:.2f}s",
            end="",
        )

        if epoch_val_loss < best_val_loss:
            print(" | Improved. Checkpoint saved.")

            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scheduler_type": "CosineAnnealingLR",
                "loss": best_val_loss,

                "x_scaler_mean": split_data["x_scaler"].mean_,
                "x_scaler_scale": split_data["x_scaler"].scale_,
                "y_scaler_state": build_scaler_state(
                    split_data["y_scaler"]
                ),

                "input_size": INPUT_SIZE,
                "output_size": OUTPUT_SIZE,
                "input_cols": INPUT_COLS,
                "target_cols": TARGET_COLS,

                "hidden_sizes": HIDDEN_SIZES,
                "dropout_rate": DROPOUT_RATE,
                "learning_rate": LEARNING_RATE,
                "batch_size": train_loader.batch_size,
                "weight_decay": WEIGHT_DECAY,
                "min_learning_rate": MIN_LEARNING_RATE,
                "validation_split": VALIDATION_SPLIT,
                "random_seed": RANDOM_SEED,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,

                "use_asinh_target": USE_ASINH_TARGET,
                "asinh_scales": ASINH_SCALES,
                "y_scaler_type_used": Y_SCALER_TYPE,
            }

            torch.save(checkpoint, checkpoint_path)

        else:
            epochs_without_improvement += 1
            print(
                " | No improvement for "
                f"{epochs_without_improvement} epochs."
            )

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):
                print("\n--- Early Stopping Triggered ---")
                effective_epochs = epoch + 1
                break

    print("\n--- Training Finished ---")
    print(
        "Total training duration: "
        f"{time.time() - start_train_time:.2f}s"
    )
    print(f"Best validation loss: {best_val_loss:.6f}")

    return history, effective_epochs, best_val_loss


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    nn.Module,
    StandardScaler,
    StandardScaler | MinMaxScaler | None,
    bool,
    np.ndarray,
]:
    """Load the best model and preprocessing state."""
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    hidden_sizes = checkpoint.get(
        "hidden_sizes",
        HIDDEN_SIZES,
    )

    model = MultiTargetRegressor(
        input_size=checkpoint.get("input_size", INPUT_SIZE),
        hidden_sizes=hidden_sizes,
        output_size=checkpoint.get("output_size", OUTPUT_SIZE),
        dropout_rate=checkpoint.get(
            "dropout_rate",
            DROPOUT_RATE,
        ),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x_scaler = StandardScaler()
    x_scaler.mean_ = checkpoint["x_scaler_mean"]
    x_scaler.scale_ = checkpoint["x_scaler_scale"]
    x_scaler.n_features_in_ = len(x_scaler.mean_)

    y_scaler = restore_target_scaler(
        checkpoint.get("y_scaler_state")
    )

    use_asinh = checkpoint.get("use_asinh_target", False)
    asinh_scales = checkpoint.get(
        "asinh_scales",
        ASINH_SCALES,
    )

    print(
        "Loaded checkpoint from epoch "
        f"{checkpoint.get('epoch', 'N/A')}"
    )
    print(f"Loaded hidden sizes: {hidden_sizes}")
    print(
        "Saved learning rate: "
        f"{checkpoint.get('learning_rate', LEARNING_RATE)}"
    )
    print(
        "Saved weight decay: "
        f"{checkpoint.get('weight_decay', WEIGHT_DECAY)}"
    )
    print(
        "Saved minimum learning rate: "
        f"{checkpoint.get('min_learning_rate', MIN_LEARNING_RATE)}"
    )
    print(
        "LR scheduler: "
        f"{checkpoint.get('scheduler_type', 'CosineAnnealingLR')}"
    )

    return model, x_scaler, y_scaler, use_asinh, asinh_scales


# =============================================================================
# Prediction and metrics
# =============================================================================

def predict_original_scale(
    model: nn.Module,
    x_np: np.ndarray,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler | MinMaxScaler | None,
    use_asinh: bool,
    asinh_scales: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Predict targets and return them on their original physical scale."""
    x_scaled_np = x_scaler.transform(x_np).astype(np.float32)
    x_tensor = torch.from_numpy(x_scaled_np).float()

    dataset = TensorDataset(x_tensor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    pred_scaled_list = []
    model.eval()

    with torch.no_grad():
        for (inputs_batch,) in loader:
            inputs_batch = inputs_batch.to(
                device,
                non_blocking=True,
            )
            outputs_scaled = model(inputs_batch)
            pred_scaled_list.append(
                outputs_scaled.cpu().numpy()
            )

    pred_scaled_np = np.concatenate(
        pred_scaled_list,
        axis=0,
    )

    if y_scaler is not None:
        pred_trans_np = y_scaler.inverse_transform(
            pred_scaled_np
        )
    else:
        pred_trans_np = pred_scaled_np

    pred_np = inverse_transform_targets(
        pred_trans_np,
        use_asinh=use_asinh,
        scales=asinh_scales,
    )

    return pred_np


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate regression metrics for one target."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    safe_y_true = np.where(y_true == 0, 1e-9, y_true)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mape = (
            mean_absolute_percentage_error(
                safe_y_true,
                y_pred,
            )
            * 100
        )

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "medae": float(medae),
        "r2": float(r2),
        "mape": float(mape),
    }


def print_metrics(
    y_true_np: np.ndarray,
    y_pred_np: np.ndarray,
    dataset_name: str,
) -> None:
    """Print target-specific metrics on the original target scale."""
    print(f"\n--- {dataset_name} Metrics: Original Scale ---")

    for k, target_col in enumerate(TARGET_COLS):
        metrics = calculate_metrics(
            y_true_np[:, k],
            y_pred_np[:, k],
        )

        print(f"\nTarget: {target_col}")
        print(f"  MSE:   {metrics['mse']:.6f}")
        print(f"  RMSE:  {metrics['rmse']:.6f}")
        print(f"  MAE:   {metrics['mae']:.6f}")
        print(f"  MedAE: {metrics['medae']:.6f}")
        print(f"  R2:    {metrics['r2']:.4f}")
        print(f"  MAPE:  {metrics['mape']:.4f}%")


# =============================================================================
# Inference timing
# =============================================================================

def predict_one_sample_end_to_end(
    model: nn.Module,
    x_one_np: np.ndarray,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler | MinMaxScaler | None,
    use_asinh: bool,
    asinh_scales: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Predict one sample including preprocessing and postprocessing.

    This timing includes:
    - scikit-learn input scaling;
    - NumPy-to-Torch conversion;
    - transfer to the model device;
    - model forward pass;
    - transfer back to CPU;
    - inverse target scaling;
    - inverse ASINH transform.
    """
    x_scaled_np = x_scaler.transform(x_one_np).astype(np.float32)
    x_tensor = torch.from_numpy(x_scaled_np).float().to(device)

    with torch.no_grad():
        y_scaled_tensor = model(x_tensor)

    y_scaled_np = y_scaled_tensor.cpu().numpy()

    if y_scaler is not None:
        y_trans_np = y_scaler.inverse_transform(y_scaled_np)
    else:
        y_trans_np = y_scaled_np

    return inverse_transform_targets(
        y_trans_np,
        use_asinh=use_asinh,
        scales=asinh_scales,
    )


def measure_single_sample_inference(
    model: nn.Module,
    x_one_np: np.ndarray,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler | MinMaxScaler | None,
    use_asinh: bool,
    asinh_scales: np.ndarray,
    device: torch.device,
    repeats: int,
) -> tuple[np.ndarray, float, float]:
    """
    Measure both end-to-end and model-only single-sample inference time.

    Returns
    -------
    prediction:
        Prediction on the original target scale.
    end_to_end_seconds:
        Average preprocessing + model + postprocessing time.
    model_only_seconds:
        Average model forward-pass time using a pre-scaled tensor.
    """
    if repeats <= 0:
        raise ValueError("timing repeats must be positive.")

    model.eval()

    # Warm-up for end-to-end timing.
    for _ in range(20):
        prediction = predict_one_sample_end_to_end(
            model,
            x_one_np,
            x_scaler,
            y_scaler,
            use_asinh,
            asinh_scales,
            device,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for _ in range(repeats):
        prediction = predict_one_sample_end_to_end(
            model,
            x_one_np,
            x_scaler,
            y_scaler,
            use_asinh,
            asinh_scales,
            device,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_to_end_seconds = (
        time.perf_counter() - start_time
    ) / repeats

    # Prepare a pre-scaled tensor once for model-only timing.
    x_scaled_np = x_scaler.transform(
        x_one_np
    ).astype(np.float32)
    x_tensor = torch.from_numpy(
        x_scaled_np
    ).float().to(device)

    with torch.no_grad():
        for _ in range(20):
            _ = model(x_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        for _ in range(repeats):
            _ = model(x_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()

    model_only_seconds = (
        time.perf_counter() - start_time
    ) / repeats

    return prediction, end_to_end_seconds, model_only_seconds


def save_inference_timing(
    output_path: Path,
    device: torch.device,
    sample_index: int,
    prediction: np.ndarray,
    end_to_end_seconds: float,
    model_only_seconds: float,
) -> None:
    """Save clearly labelled single-sample inference timing results."""
    pred_nf = prediction[0, TARGET_COLS.index("NF_dB")]
    pred_gain = prediction[
        0,
        TARGET_COLS.index("max_gain_dB"),
    ]

    lines = [
        f"Device used: {device}",
        f"Input sample index: {sample_index}",
        f"Predicted NF: {pred_nf:.6f} dB",
        f"Predicted Max gain: {pred_gain:.6f} dB",
        "",
        "End-to-end single-sample inference:",
        f"  {end_to_end_seconds:.9f} s",
        f"  {end_to_end_seconds * 1e3:.6f} ms",
        f"  {end_to_end_seconds * 1e6:.3f} us",
        "",
        "Model-only single-sample forward pass:",
        f"  {model_only_seconds:.9f} s",
        f"  {model_only_seconds * 1e3:.6f} ms",
        f"  {model_only_seconds * 1e6:.3f} us",
    ]

    print("\n--- Single-Sample Inference Time ---")
    for line in lines:
        print(line)

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# Plotting
# =============================================================================

def plot_training_history(
    history: dict[str, list],
    effective_epochs: int,
    folders: dict[str, Path],
) -> dict[str, np.ndarray | int]:
    """Create and save the training-history plots."""
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

    # Loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs_range,
        history["train_loss"][:effective_epochs],
        label="Training Loss",
        marker=".",
    )
    plt.plot(
        epochs_range,
        history["val_loss"][:effective_epochs],
        label="Validation Loss",
        marker=".",
    )
    plt.axvline(
        best_epoch_num,
        linestyle="--",
        label=f"Best Epoch {best_epoch_num}",
    )
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title(
        "Training and Validation Loss for NF_dB and max_gain_dB"
    )
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    save_current_figure(
        "plot_01_training_validation_loss_curves",
        folders,
        {
            "epochs": epochs_np,
            "train_loss": train_loss_np,
            "val_loss": val_loss_np,
            "best_epoch": np.array([[best_epoch_num]]),
        },
    )
    plt.show()
    plt.close()

    # Validation MAE
    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs_range,
        history["val_mae_scaled_NF_dB"][:effective_epochs],
        label="Validation MAE scaled NF_dB",
        marker=".",
    )
    plt.plot(
        epochs_range,
        history["val_mae_scaled_max_gain_dB"][:effective_epochs],
        label="Validation MAE scaled max_gain_dB",
        marker=".",
    )
    plt.axvline(
        best_epoch_num,
        linestyle="--",
        label=f"Best Epoch {best_epoch_num}",
    )
    plt.xlabel("Epoch")
    plt.ylabel("MAE scaled")
    plt.title("Validation MAE on Scaled Targets")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_current_figure(
        "plot_02_validation_mae_scaled_targets",
        folders,
        {
            "epochs": epochs_np,
            "val_mae_scaled_NF_dB": val_mae_nf_np,
            "val_mae_scaled_max_gain_dB": val_mae_gain_np,
            "best_epoch": np.array([[best_epoch_num]]),
        },
    )
    plt.show()
    plt.close()

    # Validation R2
    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs_range,
        history["val_r2_scaled_NF_dB"][:effective_epochs],
        label="Validation R2 scaled NF_dB",
        marker=".",
    )
    plt.plot(
        epochs_range,
        history["val_r2_scaled_max_gain_dB"][:effective_epochs],
        label="Validation R2 scaled max_gain_dB",
        marker=".",
    )
    plt.axvline(
        best_epoch_num,
        linestyle="--",
        label=f"Best Epoch {best_epoch_num}",
    )
    plt.xlabel("Epoch")
    plt.ylabel("R2 scaled")
    plt.title("Validation R2 on Scaled Targets")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_current_figure(
        "plot_03_validation_r2_scaled_targets",
        folders,
        {
            "epochs": epochs_np,
            "val_r2_scaled_NF_dB": val_r2_nf_np,
            "val_r2_scaled_max_gain_dB": val_r2_gain_np,
            "best_epoch": np.array([[best_epoch_num]]),
        },
    )
    plt.show()
    plt.close()

    return {
        "epochs": epochs_np,
        "train_loss": train_loss_np,
        "val_loss": val_loss_np,
        "val_mae_nf": val_mae_nf_np,
        "val_mae_gain": val_mae_gain_np,
        "val_r2_nf": val_r2_nf_np,
        "val_r2_gain": val_r2_gain_np,
        "best_epoch": best_epoch_num,
    }


def plot_dataset_predictions(
    y_true_np: np.ndarray,
    y_pred_np: np.ndarray,
    dataset_label: str,
    file_prefix: str,
    folders: dict[str, Path],
    large_font: bool = False,
) -> None:
    """
    Save predicted-vs-actual and absolute-error plots for both targets.

    For external data, prediction-error histograms are shown but not saved.
    """
    for k, target_col in enumerate(TARGET_COLS):
        display_name = DISPLAY_NAMES[target_col]
        safe_target = target_col.replace("-", "_").replace(" ", "_")

        y_true = y_true_np[:, k]
        y_pred = y_pred_np[:, k]

        prediction_error = y_true - y_pred
        absolute_error = np.abs(prediction_error)
        metrics = calculate_metrics(y_true, y_pred)

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        buffer = 0.05 * (max_val - min_val) + 1e-6

        plot_min = min_val - buffer
        plot_max = max_val + buffer

        label_fontsize = 20 if large_font else None
        tick_fontsize = 16 if large_font else None
        legend_fontsize = 18 if large_font else None

        # Predicted vs actual
        plt.figure(figsize=(7, 7))
        plt.scatter(
            y_true,
            y_pred,
            alpha=0.5 if large_font else 0.4,
            s=18 if large_font else 12,
            edgecolors="k",
            linewidths=0.3,
        )
        plt.plot(
            [plot_min, plot_max],
            [plot_min, plot_max],
            "r--",
            label="Ideal fit",
        )
        plt.xlabel(
            f"Actual {display_name}",
            fontsize=label_fontsize,
        )
        plt.ylabel(
            f"Predicted {display_name}",
            fontsize=label_fontsize,
        )
        plt.title(
            f"{dataset_label}: Predicted vs Actual {display_name}"
        )
        plt.xlim(plot_min, plot_max)
        plt.ylim(plot_min, plot_max)
        plt.gca().set_aspect("equal", adjustable="box")

        if tick_fontsize is not None:
            plt.xticks(fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)

        plt.grid(True, linestyle=":")
        plt.legend(fontsize=legend_fontsize)
        plt.tight_layout()

        save_current_figure(
            f"{file_prefix}_pred_vs_actual_{safe_target}",
            folders,
            {
                "y_true": y_true.reshape(-1, 1),
                "y_pred": y_pred.reshape(-1, 1),
                "prediction_error": prediction_error.reshape(-1, 1),
                "absolute_error": absolute_error.reshape(-1, 1),
                "plot_min": np.array([[plot_min]]),
                "plot_max": np.array([[plot_max]]),
                "mse": np.array([[metrics["mse"]]]),
                "rmse": np.array([[metrics["rmse"]]]),
                "mae": np.array([[metrics["mae"]]]),
                "medae": np.array([[metrics["medae"]]]),
                "r2": np.array([[metrics["r2"]]]),
                "mape": np.array([[metrics["mape"]]]),
            },
        )
        plt.show()
        plt.close()

        # Prediction-error histogram for external data only.
        if dataset_label.lower().startswith("external"):
            plt.figure(figsize=(8, 5))
            plt.hist(
                prediction_error,
                bins=40,
                edgecolor="black",
                alpha=0.7,
            )
            plt.xlabel(
                f"Prediction Error: Actual - Predicted {target_col}"
            )
            plt.ylabel("Count")
            plt.title(
                f"External Test Error Histogram for {target_col}"
            )
            plt.grid(True, linestyle=":")
            plt.tight_layout()
            plt.show()
            plt.close()

        # Absolute error vs actual
        plt.figure(figsize=(8, 5))
        plt.scatter(
            y_true,
            absolute_error,
            alpha=0.5,
            s=18,
            edgecolors="k",
            linewidths=0.3,
        )
        plt.xlabel(
            f"Actual {display_name}",
            fontsize=18 if large_font else None,
        )
        plt.ylabel(
            "Absolute Error",
            fontsize=18 if large_font else None,
        )
        plt.title(
            f"{dataset_label}: Absolute Error vs Actual "
            f"{display_name}"
        )

        if large_font:
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

        plt.grid(True, linestyle=":")
        plt.tight_layout()

        save_current_figure(
            f"{file_prefix}_abs_error_vs_actual_{safe_target}",
            folders,
            {
                "y_true": y_true.reshape(-1, 1),
                "absolute_error": absolute_error.reshape(-1, 1),
                "prediction_error": prediction_error.reshape(-1, 1),
                "mse": np.array([[metrics["mse"]]]),
                "rmse": np.array([[metrics["rmse"]]]),
                "mae": np.array([[metrics["mae"]]]),
                "medae": np.array([[metrics["medae"]]]),
                "r2": np.array([[metrics["r2"]]]),
                "mape": np.array([[metrics["mape"]]]),
            },
        )
        plt.show()
        plt.close()


# =============================================================================
# External test evaluation
# =============================================================================

def evaluate_external_dataset(
    external_data_path: Path,
    model: nn.Module,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler | MinMaxScaler | None,
    use_asinh: bool,
    asinh_scales: np.ndarray,
    device: torch.device,
    batch_size: int,
    sample_index: int,
    timing_repeats: int,
    folders: dict[str, Path],
) -> tuple[
    bool,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Evaluate the external CSV when available."""
    if not external_data_path.exists():
        print(
            "\nExternal test file not found. "
            "Skipping external evaluation:"
        )
        print(external_data_path)
        return False, None, None, None

    print("\n--- External Test Evaluation ---")

    df_new = load_csv(
        external_data_path,
        "External dataset",
    )

    try:
        df_new_model = select_required_columns(df_new)
    except ValueError as exc:
        print(f"External evaluation skipped: {exc}")
        return False, None, None, None

    print(f"Usable external samples: {len(df_new_model)}")

    x_new_np, y_new_np = dataframe_to_arrays(df_new_model)

    if sample_index < 0 or sample_index >= len(x_new_np):
        print(
            f"Warning: sample_index={sample_index} is outside "
            "the external dataset. Using sample_index=0."
        )
        sample_index = 0

    x_one_np = x_new_np[
        sample_index:sample_index + 1
    ].astype(np.float32)

    (
        y_one_pred_np,
        end_to_end_seconds,
        model_only_seconds,
    ) = measure_single_sample_inference(
        model=model,
        x_one_np=x_one_np,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        use_asinh=use_asinh,
        asinh_scales=asinh_scales,
        device=device,
        repeats=timing_repeats,
    )

    timing_path = (
        folders["timing"] / "inference_time_results.txt"
    )
    save_inference_timing(
        output_path=timing_path,
        device=device,
        sample_index=sample_index,
        prediction=y_one_pred_np,
        end_to_end_seconds=end_to_end_seconds,
        model_only_seconds=model_only_seconds,
    )

    y_pred_new_np = predict_original_scale(
        model=model,
        x_np=x_new_np,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        use_asinh=use_asinh,
        asinh_scales=asinh_scales,
        device=device,
        batch_size=batch_size * 2,
    )

    print_metrics(
        y_new_np,
        y_pred_new_np,
        "External Test",
    )

    df_results_new = df_new_model.copy()

    for k, target_col in enumerate(TARGET_COLS):
        y_true_new = y_new_np[:, k]
        y_pred_new = y_pred_new_np[:, k]

        df_results_new[
            f"predicted_{target_col}"
        ] = y_pred_new

        df_results_new[
            f"absolute_error_{target_col}"
        ] = np.abs(y_true_new - y_pred_new)

        df_results_new[
            f"relative_error_percent_{target_col}"
        ] = (
            np.abs(y_true_new - y_pred_new)
            / np.maximum(np.abs(y_true_new), 1e-9)
        ) * 100

    predictions_path = (
        folders["predictions"]
        / "external_test_predictions_NF_and_gain.csv"
    )
    df_results_new.to_csv(predictions_path, index=False)

    print(f"\nPredictions saved to: {predictions_path}")

    plot_dataset_predictions(
        y_true_np=y_new_np,
        y_pred_np=y_pred_new_np,
        dataset_label="External Test",
        file_prefix="plot_06_external",
        folders=folders,
        large_font=True,
    )

    return True, x_new_np, y_new_np, y_pred_new_np


# =============================================================================
# Main program
# =============================================================================

def main() -> int:
    args = parse_args()

    data_path = args.data_path.expanduser().resolve()
    external_data_path = (
        args.external_data_path.expanduser().resolve()
    )
    output_dir = args.output_dir.expanduser().resolve()

    folders = create_output_directories(output_dir)

    if args.checkpoint_path is None:
        checkpoint_path = (
            folders["checkpoints"]
            / DEFAULT_CHECKPOINT_NAME
        )
    else:
        checkpoint_path = (
            args.checkpoint_path.expanduser().resolve()
        )
        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("--- Configuration ---")
    print(f"Training data: {data_path}")
    print(f"External data: {external_data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input columns: {INPUT_COLS}")
    print(f"Target columns: {TARGET_COLS}")
    print(f"Hidden sizes: {HIDDEN_SIZES}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {args.batch_size}")
    print(f"Maximum epochs: {args.epochs}")
    print(f"Dropout rate: {DROPOUT_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"LR scheduler: CosineAnnealingLR")
    print(f"Minimum learning rate: {MIN_LEARNING_RATE}")
    print(f"Use ASINH targets: {USE_ASINH_TARGET}")
    print(f"ASINH scales: {ASINH_SCALES.flatten().tolist()}")
    print("-" * 60)

    if args.epochs <= 0:
        print("ERROR: --epochs must be positive.")
        return 1

    if args.batch_size <= 0:
        print("ERROR: --batch-size must be positive.")
        return 1

    print("\n--- Device Setup ---")
    device = get_device()
    set_random_seed(RANDOM_SEED)
    print(f"Random seed: {RANDOM_SEED}")

    # -------------------------------------------------------------------------
    # Load and preprocess training data
    # -------------------------------------------------------------------------
    print("\n--- Loading Training Data ---")
    start_load_time = time.time()

    try:
        df = load_csv(data_path, "Training dataset")
        df_model = select_required_columns(df)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}")
        return 1
    except Exception as exc:
        print(
            "\nUnexpected error while loading training data: "
            f"{exc}"
        )
        return 1

    print(
        "Training data loading finished in "
        f"{time.time() - start_load_time:.2f}s."
    )

    print("\n--- Preprocessing Training Data ---")
    start_preprocess_time = time.time()

    x_data_np, y_data_np = dataframe_to_arrays(df_model)

    print(f"Number of usable samples: {len(df_model)}")
    print(f"X shape: {x_data_np.shape}")
    print(f"Y shape: {y_data_np.shape}")

    show_sample_rows(x_data_np, y_data_np)
    show_target_distributions(df_model, y_data_np)

    print("\nSplitting and scaling data...")
    split_data = split_and_scale_data(
        x_data_np,
        y_data_np,
    )

    train_loader, val_loader = create_data_loaders(
        split_data,
        batch_size=args.batch_size,
        device=device,
    )

    print(
        "Preprocessing finished in "
        f"{time.time() - start_preprocess_time:.2f}s."
    )

    # -------------------------------------------------------------------------
    # Create and train model
    # -------------------------------------------------------------------------
    print("\n--- Defining Neural Network ---")

    model = MultiTargetRegressor(
        input_size=INPUT_SIZE,
        hidden_sizes=HIDDEN_SIZES,
        output_size=OUTPUT_SIZE,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    print(model)
    print(
        "Total parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    history, effective_epochs, _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        split_data=split_data,
        device=device,
        checkpoint_path=checkpoint_path,
        epochs=args.epochs,
    )

    # -------------------------------------------------------------------------
    # Load best checkpoint and evaluate validation data
    # -------------------------------------------------------------------------
    print("\n--- Loading Best Model ---")

    (
        eval_model,
        eval_x_scaler,
        eval_y_scaler,
        use_asinh_loaded,
        asinh_scales_loaded,
    ) = load_checkpoint(
        checkpoint_path,
        device,
    )

    print("\n--- Evaluating Validation Set ---")

    preds_val_np = predict_original_scale(
        model=eval_model,
        x_np=split_data["x_val"],
        x_scaler=eval_x_scaler,
        y_scaler=eval_y_scaler,
        use_asinh=use_asinh_loaded,
        asinh_scales=asinh_scales_loaded,
        device=device,
        batch_size=args.batch_size * 2,
    )

    print_metrics(
        split_data["y_val"],
        preds_val_np,
        "Validation",
    )

    # -------------------------------------------------------------------------
    # Save training/validation plots
    # -------------------------------------------------------------------------
    print(
        "\n--- Generating and Saving "
        "Non-Histogram Training/Validation Plots ---"
    )

    history_arrays = plot_training_history(
        history,
        effective_epochs,
        folders,
    )

    plot_dataset_predictions(
        y_true_np=split_data["y_val"],
        y_pred_np=preds_val_np,
        dataset_label="Validation",
        file_prefix="plot_04_validation",
        folders=folders,
        large_font=False,
    )

    # -------------------------------------------------------------------------
    # Optional external evaluation
    # -------------------------------------------------------------------------
    (
        external_test_available,
        x_new_np,
        y_new_np,
        y_pred_new_np,
    ) = evaluate_external_dataset(
        external_data_path=external_data_path,
        model=eval_model,
        x_scaler=eval_x_scaler,
        y_scaler=eval_y_scaler,
        use_asinh=use_asinh_loaded,
        asinh_scales=asinh_scales_loaded,
        device=device,
        batch_size=args.batch_size,
        sample_index=args.sample_index,
        timing_repeats=args.timing_repeats,
        folders=folders,
    )

    # -------------------------------------------------------------------------
    # Save MATLAB summary
    # -------------------------------------------------------------------------
    summary_data = {
        "epochs": history_arrays["epochs"],
        "train_loss": history_arrays["train_loss"],
        "val_loss": history_arrays["val_loss"],
        "val_mae_scaled_NF_dB": history_arrays["val_mae_nf"],
        "val_mae_scaled_max_gain_dB": history_arrays["val_mae_gain"],
        "val_r2_scaled_NF_dB": history_arrays["val_r2_nf"],
        "val_r2_scaled_max_gain_dB": history_arrays["val_r2_gain"],
        "best_epoch": np.array(
            [[history_arrays["best_epoch"]]]
        ),
        "x_val": split_data["x_val"],
        "y_val_true": split_data["y_val"],
        "y_val_pred": preds_val_np,
    }

    if (
        external_test_available
        and x_new_np is not None
        and y_new_np is not None
        and y_pred_new_np is not None
    ):
        summary_data["x_external"] = x_new_np
        summary_data["y_external_true"] = y_new_np
        summary_data["y_external_pred"] = y_pred_new_np

    summary_path = (
        folders["mat"]
        / "summary_regression_non_histogram_plots.mat"
    )
    savemat(summary_path, summary_data)

    print("\n--- Output Summary ---")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"PDF figures: {folders['pdf']}")
    print(f"PNG figures: {folders['png']}")
    print(f"MATLAB data: {folders['mat']}")
    print(f"Predictions: {folders['predictions']}")
    print(f"Timing results: {folders['timing']}")

    print("\n--- Script Finished Successfully ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
