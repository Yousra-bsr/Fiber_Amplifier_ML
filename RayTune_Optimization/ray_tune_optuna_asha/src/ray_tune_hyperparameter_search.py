"""
Ray Tune + Optuna + ASHA hyperparameter search for a four-input,
two-target PyTorch MLP, with GPU-oriented execution options.

This script performs hyperparameter optimization only. It DOES NOT train a separate
final DNN after Ray Tune finishes. The best hyperparameters are saved for use by
 another DNN training script (DNN_model_I and DNN_model_II).

Important: Ray Tune trials still train temporary candidate networks. This is
required so Optuna can compare configurations and ASHA can stop weak trials.

# ============================================================
# B300-optimized Ray Tune + Optuna + ASHA search
# Four-input / two-target MLP
#
# Optimizations for GPU-heavy cloud instances (tested design target: Runpod
# 2x NVIDIA B300, 56 vCPU, hundreds of GB RAM/VRAM):
#   - Ray Tune uses GPUs by default when CUDA is available.
#   - Multiple independent trials can share each physical GPU via Ray
#     fractional GPU resources (TRIALS_PER_GPU, default 4).
#   - Training/validation tensors are copied to the assigned GPU once per
#     trial; no DataLoader CPU->GPU copy occurs inside the epoch loop.
#   - Validation is evaluated in one GPU-resident pass.
#   - Per-batch .item() synchronizations are removed.
#   - BF16 autocast is enabled by default on CUDA.
#   - Fused AdamW is used when supported by the installed PyTorch build.
#   - TF32 / high-precision TensorFloat matrix paths are enabled.
#   - Target transforms/scalers are precomputed once for the discrete search
#     space and shared with Ray workers through the object store.
#   
#
# Main speed knobs (environment variables):
#   TRIALS_PER_GPU=4          # try 2, 4, then 8 on B300
#   BATCH_CHOICES= 64,128,256
#   USE_BF16=1
#   TUNE_COMPILE=0            # compile cost can hurt short ASHA trials
#   CPUS_PER_TRIAL=2
#   SYSTEM_RESERVED_CPUS=4
#   BASE_DIR=/workspace/labs
#   RAY_STORAGE_PATH=/tmp/ray_results_b300
# ============================================================
"""

from __future__ import annotations

import json
import math
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


# ============================================================
# Host-thread limits
# ============================================================

# Keep host-side libraries from creating large thread pools in every Ray worker.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ============================================================
# Small configuration helpers
# ============================================================


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}



def parse_int_list_env(name: str, default: str) -> list[int]:
    raw = os.environ.get(name, default)
    values = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    return values


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASE_DIR_ENV = os.environ.get("BASE_DIR")
if BASE_DIR_ENV:
    BASE_DIR = Path(BASE_DIR_ENV).expanduser().resolve()
    DATA_DIR = BASE_DIR
else:
    BASE_DIR = PROJECT_ROOT
    DATA_DIR = PROJECT_ROOT / "data"

# Dataset used for the hyperparameter search.
# Model II:

RAYTUNE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = RAYTUNE_DIR / "outputs"
DATA_PATH = DATA_DIR / "dataset_nlse_300000.csv"

# Model I :
#RAYTUNE_DIR = Path(__file__).resolve().parents[2]
#OUTPUT_DIR = RAYTUNE_DIR / "outputs"
#DATA_PATH = DATA_DIR / "dataset_200000.csv"

RAY_STORAGE_PATH = Path(
    os.environ.get("RAY_STORAGE_PATH", str(OUTPUT_DIR / "ray_results"))
).expanduser().resolve()

SEARCH_OUTPUT_DIR = OUTPUT_DIR / "hyperparameter_search"
BEST_CONFIG_JSON = SEARCH_OUTPUT_DIR / "best_hyperparameters.json"
BEST_CONFIG_CSV = SEARCH_OUTPUT_DIR / "best_hyperparameters.csv"
TRIAL_RESULTS_CSV = SEARCH_OUTPUT_DIR / "ray_tune_trial_results.csv"

for directory in (OUTPUT_DIR, RAY_STORAGE_PATH, SEARCH_OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# Data/model definition
# ============================================================

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

INPUT_SIZE = len(INPUT_COLS)
OUTPUT_SIZE = len(TARGET_COLS)

RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2
# ------------------------------------------------------------
# CosineAnnealingLR settings
# ------------------------------------------------------------
# eta_min is the minimum learning rate approached at the end of
# the cosine cycle. During tuning, T_max is TUNE_EPOCHS.
MIN_LEARNING_RATE = 1e-6


# ============================================================
# Ray Tune controls
# ============================================================

TUNE_EPOCHS = int(os.environ.get("TUNE_EPOCHS", "500"))
TUNE_EARLY_STOPPING_PATIENCE = int(
    os.environ.get("TUNE_EARLY_STOPPING_PATIENCE", "80")
)
NUM_TRIALS = int(os.environ.get("NUM_TRIALS", "200"))

NF_ASINH_CHOICES = [2e-4, 5e-4, 1e-3, 2e-3]
GAIN_ASINH_CHOICES = [0.05, 0.1, 0.2, 0.5]


# ============================================================
# Server / GPU resource configuration
# ============================================================


def detect_available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


TOTAL_CPUS = detect_available_cpus()
TOTAL_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

SYSTEM_RESERVED_CPUS = max(
    0,
    int(os.environ.get("SYSTEM_RESERVED_CPUS", "4" if TOTAL_GPUS else "0")),
)
USABLE_CPUS = max(1, TOTAL_CPUS - SYSTEM_RESERVED_CPUS)

# GPU tuning is the default on a CUDA machine. For a tiny MLP, a whole B300
# per trial wastes most of the accelerator, so Ray fractional GPU resources
# are used to place several independent trials on each physical GPU.
TUNE_USE_GPU = env_flag("TUNE_USE_GPU", TOTAL_GPUS > 0)
TRIALS_PER_GPU = max(1, int(os.environ.get("TRIALS_PER_GPU", "4")))
GPUS_PER_TRIAL = (1.0 / TRIALS_PER_GPU) if TUNE_USE_GPU else 0.0

# Two CPU slots are usually enough because all minibatch work is GPU-resident.
CPUS_PER_TRIAL = max(1, int(os.environ.get("CPUS_PER_TRIAL", "2")))
CPU_TRIAL_CAPACITY = max(1, USABLE_CPUS // CPUS_PER_TRIAL)

if TUNE_USE_GPU:
    if TOTAL_GPUS < 1:
        raise RuntimeError("TUNE_USE_GPU=True but CUDA is not available.")
    GPU_TRIAL_CAPACITY = max(
        1,
        int(math.floor(TOTAL_GPUS / GPUS_PER_TRIAL + 1e-9)),
    )
else:
    GPU_TRIAL_CAPACITY = NUM_TRIALS

DEFAULT_CONCURRENCY = min(NUM_TRIALS, CPU_TRIAL_CAPACITY, GPU_TRIAL_CAPACITY)
TARGET_CONCURRENT_TRIALS = max(
    1,
    min(
        NUM_TRIALS,
        int(
            os.environ.get(
                "TARGET_CONCURRENT_TRIALS",
                str(DEFAULT_CONCURRENCY),
            )
        ),
    ),
)
MAX_CONCURRENT_TRIALS = min(
    NUM_TRIALS,
    TARGET_CONCURRENT_TRIALS,
    CPU_TRIAL_CAPACITY,
    GPU_TRIAL_CAPACITY,
)

# GPU math modes. BF16 keeps FP32 master weights/optimizer states while using
# Tensor Cores for Linear operations under autocast.
USE_BF16 = env_flag("USE_BF16", True)
if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported"):
    USE_BF16 = USE_BF16 and bool(torch.cuda.is_bf16_supported())

USE_FUSED_ADAMW = env_flag("USE_FUSED_ADAMW", True)
TUNE_COMPILE = env_flag("TUNE_COMPILE", False)

BATCH_CHOICES = parse_int_list_env("BATCH_CHOICES", "64,128,256")
ASHA_GRACE_PERIOD = max(1, int(os.environ.get("ASHA_GRACE_PERIOD", "75")))
ASHA_REDUCTION_FACTOR = max(
    2,
    int(os.environ.get("ASHA_REDUCTION_FACTOR", "3")),
)

if torch.cuda.is_available():
    # High throughput for Linear layers on NVIDIA Tensor Cores.
    torch.set_float32_matmul_precision("high")
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


# ============================================================
# Reproducibility
# ============================================================


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(RANDOM_SEED)


# ============================================================
# Diagnostics
# ============================================================


def print_resource_summary() -> None:
    print("\n============================================")
    print("SERVER / RAY / GPU CONFIGURATION")
    print("============================================")
    print(f"Detected logical CPUs      : {TOTAL_CPUS}")
    print(f"Reserved CPUs              : {SYSTEM_RESERVED_CPUS}")
    print(f"CPUs exposed to Ray        : {USABLE_CPUS}")
    print(f"CPUs per Ray trial         : {CPUS_PER_TRIAL}")
    print(f"Detected CUDA GPUs         : {TOTAL_GPUS}")

    if TOTAL_GPUS:
        for gpu_idx in range(TOTAL_GPUS):
            print(f"GPU {gpu_idx}                    : {torch.cuda.get_device_name(gpu_idx)}")

    print(f"Tune uses GPU              : {TUNE_USE_GPU}")
    print(f"Trials per physical GPU    : {TRIALS_PER_GPU if TUNE_USE_GPU else 0}")
    print(f"GPU resource per trial     : {GPUS_PER_TRIAL:.4f}")
    print(f"Concurrent Ray trials      : {MAX_CONCURRENT_TRIALS}")
    print(f"BF16 autocast              : {USE_BF16 and TUNE_USE_GPU}")
    print(f"Fused AdamW                : {USE_FUSED_ADAMW and TUNE_USE_GPU}")
    print(f"Tune torch.compile         : {TUNE_COMPILE}")
    print(f"Batch choices              : {BATCH_CHOICES}")
    print(f"Ray storage path           : {RAY_STORAGE_PATH}")
    print(f"Search summary directory   : {SEARCH_OUTPUT_DIR}")


# ============================================================
# Target transformations
# ============================================================


def make_asinh_scales(
    nf_asinh_scale: float,
    gain_asinh_scale: float,
) -> np.ndarray:
    return np.array(
        [nf_asinh_scale, gain_asinh_scale],
        dtype=np.float32,
    ).reshape(1, -1)



def transform_targets(y_np: np.ndarray, scales: np.ndarray) -> np.ndarray:
    y_np = y_np.astype(np.float32, copy=True)
    return np.arcsinh(y_np / scales).astype(np.float32)


# ============================================================
# Neural network used INSIDE each tuning trial
# ============================================================


class MultiTargetMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
    ) -> None:
        super().__init__()

        if len(hidden_sizes) not in (4, 5, 6):
            raise ValueError("hidden_sizes must contain 4, 5, or 6 values.")

        layers: list[nn.Module] = []
        previous_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ============================================================
# Load and split tuning data
# ============================================================


def load_and_prepare_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    print("\n--- Loading Tuning Data ---")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required_cols = INPUT_COLS + TARGET_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df_model = df[required_cols].copy().dropna()

    x_data_np = df_model[INPUT_COLS].to_numpy(dtype=np.float32)
    y_data_np = df_model[TARGET_COLS].to_numpy(dtype=np.float32)

    gain_idx = TARGET_COLS.index("max_gain_dB")
    y_data_np[:, gain_idx] = np.maximum(y_data_np[:, gain_idx], 0.0)

    all_indices = np.arange(len(df_model))
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    x_train_np = x_data_np[train_indices]
    x_val_np = x_data_np[val_indices]
    y_train_np = y_data_np[train_indices]
    y_val_np = y_data_np[val_indices]

    print(f"Total samples      : {len(df_model)}")
    print(f"Training samples   : {len(train_indices)}")
    print(f"Validation samples : {len(val_indices)}")

    return x_train_np, x_val_np, y_train_np, y_val_np


# ============================================================
# Precompute trial-invariant quantities once
# ============================================================

# X scaling is independent of all tuned hyperparameters.
# Target transforms/scalers are precomputed for every discrete ASINH-scale
# pair so that each Ray worker can reuse them instead of recomputing them.


def build_tune_data_bundle(
    x_train_np: np.ndarray,
    x_val_np: np.ndarray,
    y_train_np: np.ndarray,
    y_val_np: np.ndarray,
) -> dict:
    x_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train_np).astype(np.float32)
    x_val_scaled = x_scaler.transform(x_val_np).astype(np.float32)

    target_cache: dict[tuple[float, float], dict[str, np.ndarray]] = {}

    for nf_scale in NF_ASINH_CHOICES:
        for gain_scale in GAIN_ASINH_CHOICES:
            asinh_scales = make_asinh_scales(nf_scale, gain_scale)

            y_train_trans = transform_targets(y_train_np, asinh_scales)
            y_val_trans = transform_targets(y_val_np, asinh_scales)

            y_scaler = StandardScaler()
            y_train_scaled = y_scaler.fit_transform(y_train_trans).astype(np.float32)
            y_val_scaled = y_scaler.transform(y_val_trans).astype(np.float32)

            target_cache[(float(nf_scale), float(gain_scale))] = {
                "y_train": y_train_scaled,
                "y_val": y_val_scaled,
            }

    return {
        "x_train": x_train_scaled,
        "x_val": x_val_scaled,
        "target_cache": target_cache,
    }



def get_target_cache_entry(
    data_bundle: dict,
    nf_asinh_scale: float,
    gain_asinh_scale: float,
) -> dict:
    key = (float(nf_asinh_scale), float(gain_asinh_scale))
    try:
        return data_bundle["target_cache"][key]
    except KeyError as exc:
        raise KeyError(
            f"No precomputed target transform for nf={key[0]}, gain={key[1]}."
        ) from exc



def host_array_to_device(
    array: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array))

    if device.type == "cuda":
        return tensor.pin_memory().to(device, non_blocking=True)

    return tensor.to(device)



def prepare_trial_device_data(
    config: dict,
    device: torch.device,
    data_bundle: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    entry = get_target_cache_entry(
        data_bundle,
        float(config["nf_asinh_scale"]),
        float(config["gain_asinh_scale"]),
    )

    x_train_t = host_array_to_device(data_bundle["x_train"], device)
    y_train_t = host_array_to_device(entry["y_train"], device)
    x_val_t = host_array_to_device(data_bundle["x_val"], device)
    y_val_t = host_array_to_device(entry["y_val"], device)

    train_matrix = torch.cat([x_train_t, y_train_t], dim=1).contiguous()

    return train_matrix, x_val_t.contiguous(), y_val_t.contiguous()


# ============================================================
# Trial-training helpers
# ============================================================


def autocast_context(device: torch.device):
    if device.type == "cuda" and USE_BF16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()



def make_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> optim.Optimizer:
    kwargs = {
        "lr": float(lr),
        "weight_decay": float(weight_decay),
    }

    if device.type == "cuda" and USE_FUSED_ADAMW:
        try:
            return optim.AdamW(model.parameters(), fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass

    return optim.AdamW(model.parameters(), **kwargs)



def maybe_compile_model(model: nn.Module) -> nn.Module:
    if (
        TUNE_COMPILE
        and next(model.parameters()).device.type == "cuda"
        and hasattr(torch, "compile")
    ):
        try:
            return torch.compile(model, mode="reduce-overhead", dynamic=True)
        except Exception as exc:
            print(f"torch.compile disabled after setup error: {exc}")

    return model



def train_one_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_matrix: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    model.train()
    n_train = train_matrix.shape[0]

    permutation = torch.randperm(n_train, device=device)
    epoch_matrix = train_matrix.index_select(0, permutation)

    loss_sum = torch.zeros((), device=device, dtype=torch.float32)

    for start in range(0, n_train, batch_size):
        batch = epoch_matrix[start : start + batch_size]
        inputs = batch[:, :INPUT_SIZE]
        targets = batch[:, INPUT_SIZE : INPUT_SIZE + OUTPUT_SIZE]

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device):
            outputs = model(inputs)

        loss = torch.abs(outputs.float() - targets).mean()
        loss.backward()
        optimizer.step()

        loss_sum.add_(loss.detach() * batch.shape[0])

    return float((loss_sum / n_train).item())



def evaluate_validation_loss(
    model: nn.Module,
    x_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()

    with torch.inference_mode():
        with autocast_context(device):
            outputs = model(x_val_t)

        val_loss = torch.abs(outputs.float() - y_val_t).mean()

    return float(val_loss.item())


# ============================================================
# Ray Tune training function
# ============================================================


def train_ray_tune_impl(config: dict, data_bundle: dict) -> None:
    torch.set_num_threads(1 if TUNE_USE_GPU else CPUS_PER_TRIAL)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if TUNE_USE_GPU:
        local_device = torch.device("cuda:0")
        torch.cuda.set_device(local_device)
    else:
        local_device = torch.device("cpu")

    seed_everything(RANDOM_SEED)

    batch_size = int(config["batch_size"])
    hidden_sizes = list(config["hidden_sizes"])

    train_matrix, x_val_t, y_val_t = prepare_trial_device_data(
        config,
        local_device,
        data_bundle,
    )

    model = MultiTargetMLP(
        INPUT_SIZE,
        hidden_sizes,
        OUTPUT_SIZE,
    ).to(local_device)
    model = maybe_compile_model(model)

    optimizer = make_optimizer(
        model,
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        device=local_device,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TUNE_EPOCHS,
        eta_min=MIN_LEARNING_RATE,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(TUNE_EPOCHS):
        train_loss = train_one_epoch(
            model,
            optimizer,
            train_matrix,
            batch_size,
            local_device,
        )

        val_loss = evaluate_validation_loss(
            model,
            x_val_t,
            y_val_t,
            local_device,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        tune.report(
            {
                "val_loss": val_loss,
                "train_loss": train_loss,
                "lr_current": float(current_lr),
                "epoch": int(epoch + 1),
            }
        )

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= TUNE_EARLY_STOPPING_PATIENCE:
            break

    del model, optimizer, train_matrix, x_val_t, y_val_t

    if local_device.type == "cuda":
        torch.cuda.empty_cache()



def train_ray_tune(config: dict, data_bundle: dict) -> None:
    native_threads = 1 if TUNE_USE_GPU else CPUS_PER_TRIAL
    with threadpool_limits(limits=native_threads):
        train_ray_tune_impl(config, data_bundle)


# ============================================================
# Search-space / output helpers
# ============================================================


def short_dirname_creator(trial) -> str:
    return f"trial_{trial.trial_id}"



def build_search_space() -> dict:
    # Search space preserved from the original Ray Tune script.
    # Ray automatically converts these Tune distributions into the
    # representation expected by OptunaSearch.
    return {
        "lr": tune.loguniform(1e-4, 1e-3),
        "batch_size": tune.choice(BATCH_CHOICES),
        "hidden_sizes": tune.choice(
            [
                # Four hidden layers
                [512, 512, 256, 256],
                [512, 256, 256, 256],
                [512, 512, 512, 256],
                # Five hidden layers
                [512, 512, 256, 256, 128],
                [512, 512, 512, 256, 256],
                [512, 512, 256, 256, 256],
                # Six hidden layers
                [512, 512, 256, 256, 128, 64],
                [512, 512, 256, 256, 256, 128],
                [512, 512, 256, 256, 128, 128],
            ]
        ),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "nf_asinh_scale": tune.choice(NF_ASINH_CHOICES),
        "gain_asinh_scale": tune.choice(GAIN_ASINH_CHOICES),
    }



def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return value



def save_search_outputs(results, best_result) -> None:
    best_config = json_safe(best_result.config)

    best_summary = {
        "best_config": best_config,
        "best_val_loss": float(best_result.metrics["val_loss"]),
        "best_epoch_reported": int(best_result.metrics.get("epoch", -1)),
        "metric": "val_loss",
        "mode": "min",
        "num_trials_requested": NUM_TRIALS,
        "tune_epochs": TUNE_EPOCHS,
        "validation_split": VALIDATION_SPLIT,
        "random_seed": RANDOM_SEED,
        "scheduler": "CosineAnnealingLR",
        "scheduler_eta_min": MIN_LEARNING_RATE,
    }

    BEST_CONFIG_JSON.write_text(
        json.dumps(best_summary, indent=2),
        encoding="utf-8",
    )

    flat_best_row = {
        "lr": best_config["lr"],
        "batch_size": best_config["batch_size"],
        "hidden_sizes": json.dumps(best_config["hidden_sizes"]),
        "number_hidden_layers": len(best_config["hidden_sizes"]),
        "weight_decay": best_config["weight_decay"],
        "nf_asinh_scale": best_config["nf_asinh_scale"],
        "gain_asinh_scale": best_config["gain_asinh_scale"],
        "best_val_loss": best_summary["best_val_loss"],
        "best_epoch_reported": best_summary["best_epoch_reported"],
    }
    pd.DataFrame([flat_best_row]).to_csv(BEST_CONFIG_CSV, index=False)

    results_df = results.get_dataframe()
    results_df.to_csv(TRIAL_RESULTS_CSV, index=False)

    print(f"\nSaved best hyperparameters JSON: {BEST_CONFIG_JSON}")
    print(f"Saved best hyperparameters CSV : {BEST_CONFIG_CSV}")
    print(f"Saved all trial results CSV    : {TRIAL_RESULTS_CSV}")


# ============================================================
# Main: hyperparameter search ONLY
# ============================================================


def main() -> None:
    print_resource_summary()

    x_train_np, x_val_np, y_train_np, y_val_np = load_and_prepare_data()
    tune_data_bundle = build_tune_data_bundle(
        x_train_np,
        x_val_np,
        y_train_np,
        y_val_np,
    )

    ray_init_kwargs = {
        "num_cpus": USABLE_CPUS,
        "include_dashboard": False,
        "ignore_reinit_error": True,
    }

    if TUNE_USE_GPU:
        ray_init_kwargs["num_gpus"] = TOTAL_GPUS

    ray.init(**ray_init_kwargs)

    try:
        print("\nRay cluster resources:")
        print(ray.cluster_resources())

        search_space = build_search_space()

        # Optuna chooses promising configurations from previous trial results.
        # ASHA below remains responsible for early stopping.
        # metric/mode are defined in TuneConfig.
        optuna_search = OptunaSearch(seed=RANDOM_SEED)

        # ASHA stops underperforming trials early while allowing promising trials
        # to continue up to TUNE_EPOCHS.
        asha_scheduler = ASHAScheduler(
            max_t=TUNE_EPOCHS,
            grace_period=ASHA_GRACE_PERIOD,
            reduction_factor=ASHA_REDUCTION_FACTOR,
        )

        resources_per_trial = {
            "cpu": CPUS_PER_TRIAL,
            "gpu": GPUS_PER_TRIAL if TUNE_USE_GPU else 0,
        }

        parameterized_trainable = tune.with_parameters(
            train_ray_tune,
            data_bundle=tune_data_bundle,
        )

        trainable = tune.with_resources(
            parameterized_trainable,
            resources=resources_per_trial,
        )

        tuner = tune.Tuner(
            trainable,
            param_space=search_space,
            run_config=RunConfig(
                name="Extended_NN_4_5_6_Layer_Hyperparameter_Search",
                storage_path=str(RAY_STORAGE_PATH),
                verbose=1,
            ),
            tune_config=tune.TuneConfig(
                metric="val_loss",
                mode="min",
                search_alg=optuna_search,
                scheduler=asha_scheduler,
                num_samples=NUM_TRIALS,
                max_concurrent_trials=MAX_CONCURRENT_TRIALS,
                trial_dirname_creator=short_dirname_creator,
            ),
        )

        print("\n--- Starting Ray Tune + Optuna + ASHA Search ---")
        print(f"Resources per trial        : {resources_per_trial}")
        print(f"Max concurrent trials      : {MAX_CONCURRENT_TRIALS}")
        print(f"Number of requested trials : {NUM_TRIALS}")
        print("Search algorithm            : Optuna")
        print("Early-stopping scheduler    : ASHA")
        print(f"Trials per physical GPU     : {TRIALS_PER_GPU if TUNE_USE_GPU else 0}")
        print(f"Batch choices               : {BATCH_CHOICES}")
        print(f"BF16 tuning                 : {USE_BF16 and TUNE_USE_GPU}")

        # Run the hyperparameter search. No separate final-model retraining follows.
        results = tuner.fit()

        best_result = results.get_best_result(
            metric="val_loss",
            mode="min",
            scope="all",
        )

        best_config = best_result.config

        print("\n============================================")
        print("BEST HYPERPARAMETERS")
        print("============================================")
        for key, value in best_config.items():
            print(f"{key}: {value}")

        print(f"\nBest validation loss: {best_result.metrics['val_loss']:.8f}")
        print(
            "Number of hidden layers: "
            f"{len(best_config['hidden_sizes'])}"
        )

        save_search_outputs(results, best_result)

        print("\n============================================")
        print("HYPERPARAMETER SEARCH FINISHED")
        print("============================================")
        print(
            "No final DNN model was retrained or evaluated. "
            "Use the saved best_hyperparameters.json/CSV in your separate "
            "training script."
        )

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
