# Fiber Amplifier ML — ready-to-use prediction function

The `predict_nf_gain` function runs Model I (Inoue) or Model II (NLSE) on **CPU** without MATLAB, Ray Tune, or retraining DNN models.

## Install

```bash
pip install -r requirements.txt
```

## Valid input ranges (Models I and II)

**Both models were trained with datasets generated using the following input ranges.**
Only provide values inside these limits (including the endpoints):

| Input parameter | Minimum | Maximum |
| --- | ---: | ---: |
| Pump power (`pump_power_dBm`) | 19 dBm | 25 dBm |
| Pump–signal wavelength separation (`pump_signal_sep_nm`) | 4 nm | 16 nm |
| Fiber zero-dispersion wavelength (`zero_disp_wavelength_nm`) | 1500 nm | 1600 nm |
| Second-order dispersion (`beta2_ps2_per_km`) | −10 ps²/km | −0.1 ps²/km |

The function **raises a `ValueError`** if any input lies outside these ranges.
This is intentional: outside the training domain, model extrapolation is not
reliable. Staying within these bounds guarantee predictive accuracy of 99% for model I and 
96% for model II. 


## Predict

From the `Prediction_Function` folder:

```python
from predict import predict_nf_gain

nf, gain = predict_nf_gain(
    pump_signal_sep_nm=5.0,
    pump_power_dBm=22.0,
    beta2_ps2_per_km=-0.1,
    zero_disp_wavelength_nm=1520.0,
    model="II",  # change to "I" for Model I to test both models.
)
print(f"NF = {nf:.6f} dB, max gain = {gain:.6f} dB")
```


## Files

- `predict.py`: network definition, checkpoint loader, prediction function
- `checkpoints/best_model_NF_and_gain_model_I.pth`: 4-hidden-layer Model I
- `checkpoints/best_model_NF_and_gain_model_II.pth`: 6-hidden-layer Model II

The saved checkpoints include input scaling, output scaling, and ASINH transform parameters; the code reuses them rather than refitting any scalers. The model is loaded only on the first call and reused thereafter. Each call returns `(NF_dB, max_gain_dB)`.

**Important:** These are ML predictions, not direct MATLAB simulation outputs.

