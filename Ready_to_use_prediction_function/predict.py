"""Fast inference for fiber-amplifier NF and gain : ready-to-use prediction function

This module loads the checkpoint of the DNN model and applies the *saved*
training preprocessing and inverse output transformations.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import codecs
import numpy as np
import torch
from torch import nn

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILES = {
    "I": BASE_DIR / "checkpoints" / "best_model_NF_and_gain_model_I.pth",
   # "I": BASE_DIR / "checkpoints" / "best_model_NF_and_gain_model_I.pth",
}
INPUT_COLS = (
    "pump_signal_sep_nm", "pump_power_dBm", "Dcc_ps",
    "zero_disp_wavelength_nm",
)
TARGET_COLS = ("NF_dB", "max_gain_dB")

# Input domain used to generate the training datasets for BOTH models.
# Bounds are inclusive, and values use the physical units shown below.
INPUT_RANGES = {
    "pump_signal_sep_nm": (2.0, 25.0, "nm"),
    "pump_power_dBm": (19.0, 25.0, "dBm"),
    "Dcc_ps": (0.01, 0.07, "ps/(nm^2 km)"),
    "zero_disp_wavelength_nm": (1500.0, 1600.0, "nm"),
}


def validate_inputs(values: np.ndarray) -> None:
    """Reject nonfinite and out-of-domain inputs for either trained model."""
    if not np.isfinite(values).all():
        raise ValueError("All four input values must be finite numbers")

    for column, value in zip(INPUT_COLS, values[0]):
        lower, upper, unit = INPUT_RANGES[column]
        if not lower <= float(value) <= upper:
            raise ValueError(
                f"{column} = {float(value):g} {unit} is outside the "
                f"training range [{lower:g}, {upper:g}] {unit}. "
                "Choose an input within this range."
            )


class MultiTargetRegressor(nn.Module):
    """The architecture used by both original DNN training scripts."""

    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, dropout_rate: float = 0.0):
        super().__init__()
        layers = []
        previous = input_size
        for width in hidden_sizes:
            layers.extend([nn.Linear(previous, width), nn.LeakyReLU(0.01)])
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            previous = width
        layers.append(nn.Linear(previous, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _load_checkpoint_safely(path: Path) -> dict:
    """Allowlist NumPy array metadata in these legacy checkpoint dictionaries.

    Only use checkpoint files supplied by a trusted source. Do not change
    weights_only=True to False for untrusted checkpoints.
    """
    from numpy.core.multiarray import _reconstruct

    allowed = [
        (_reconstruct, "numpy.core.multiarray._reconstruct"),
        (np.ndarray, "numpy.ndarray"),
        (np.dtype, "numpy.dtype"),
        type(np.dtype("float32")),
        type(np.dtype("float64")),
        (codecs.encode, "_codecs.encode"),
    ]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    path = MODEL_FILES[model_name]
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    ckpt = _load_checkpoint_safely(path)
    if tuple(ckpt["input_cols"]) != INPUT_COLS or tuple(ckpt["target_cols"]) != TARGET_COLS:
        raise ValueError("Checkpoint feature/target ordering does not match the prediction API")
    network = MultiTargetRegressor(
        input_size=ckpt["input_size"],
        hidden_sizes=list(ckpt["hidden_sizes"]),
        output_size=ckpt["output_size"],
        dropout_rate=float(ckpt.get("dropout_rate", 0.0)),
    )
    network.load_state_dict(ckpt["model_state_dict"], strict=True)
    network.eval()
    return network, ckpt


def predict_nf_gain(
    pump_signal_sep_nm: float,
    pump_power_dBm: float,
    Dcc_ps: float,
    zero_disp_wavelength_nm: float,
    model: str = "II",
) -> tuple[float, float]:
    """Return (noise_figure_dB, maximum_gain_dB) for one four-parameter input.

    Args are physical, unscaled inputs in the exact order of training features.
    The checkpoint is cached after the first call. Works on CPU without Ray,
    CUDA, MATLAB, a dataset, or a training pass.

    Input values must be within the training domain (inclusive) of both models.
    Values outside this domain raise ValueError instead of extrapolating.
    """
    model_name = str(model).upper().replace("MODEL", "").replace("_", "").strip()
    if model_name not in MODEL_FILES:
        raise ValueError("model must be 'I' or 'II'")
    values = np.asarray([[pump_signal_sep_nm, pump_power_dBm,
                          Dcc_ps, zero_disp_wavelength_nm]], dtype=np.float32)
    validate_inputs(values)

    network, ckpt = _load_model(model_name)
    mean = np.asarray(ckpt["x_scaler_mean"], dtype=np.float64)
    scale = np.asarray(ckpt["x_scaler_scale"], dtype=np.float64)
    x_scaled = ((values.astype(np.float64) - mean) / scale).astype(np.float32)
    with torch.inference_mode():
        y_scaled = network(torch.from_numpy(x_scaled)).numpy()

    scaler = ckpt.get("y_scaler_state")
    if scaler is None:
        y_trans = y_scaled
    elif scaler["type"] == "standard":
        y_trans = y_scaled * np.asarray(scaler["scale"]) + np.asarray(scaler["mean"])
    elif scaler["type"] == "minmax":
        y_trans = (y_scaled - np.asarray(scaler["min_"])) / np.asarray(scaler["scale_"])
    else:
        raise ValueError(f"Unknown saved target scaler: {scaler['type']}")

    if ckpt.get("use_asinh_target", False):
        y = np.asarray(ckpt["asinh_scales"], dtype=np.float32) * np.sinh(
            np.asarray(y_trans, dtype=np.float32)
        )
    else:
        y = np.asarray(y_trans, dtype=np.float32)
    return float(y[0, 0]), float(y[0, 1])


if __name__ == "__main__":
    for name in MODEL_FILES:
    
        nf, gain = predict_nf_gain(5, 25, 0.02, 1555.0, model=name) 
        #                        ( "pump_signal_sep_nm", "pump_power_dBm",
        #                          "Dcc_ps", "zero_disp_wavelength_nm")
        print(f"Model {name}: NF = {nf:.6f} dB, maximum gain = {gain:.6f} dB")
