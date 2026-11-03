# Fiber Amplifier ML — ready-to-use prediction function

The `predict_nf_gain` function predicts the noise figure (NF) and maximum gain for a given set of input parameters, without requiring numerical simulations or retraining the DNN model.

## Install

```bash
pip install -r requirements.txt
```

## Valid input ranges 

**The model was trained with datasets generated using the following input ranges.**
Only provide values inside these limits (including the endpoints):

| Input parameter | Minimum | Maximum |
| --- | ---: | ---: |
| Pump power (`pump_power_dBm`) | 19 dBm | 25 dBm |
| Pump–signal wavelength separation (`pump_signal_sep_nm`) | 2 nm | 25 nm |
| Fiber zero-dispersion wavelength (`zero_disp_wavelength_nm`) | 1500 nm | 1600 nm |
| Second-order dispersion (`beta2_ps2_per_km`) | 0.01 ps/(nm^2 km) | 0.07 ps/(nm^2 km) |

The function **raises a `ValueError`** if any input lies outside these ranges.
This is intentional: outside the training domain, model extrapolation is not
reliable. 


## Predict

From the `Prediction_Function` folder:

```python
from predict import predict_nf_gain

nf, gain = predict_nf_gain(
    pump_signal_sep_nm=5.0,
    pump_power_dBm=22.0,
    Dcc_ps=0.02,
    zero_disp_wavelength_nm=1520.0,
    model="I",  # Use only model I. 
)
print(f"NF = {nf:.6f} dB, max gain = {gain:.6f} dB")
```


## Files

- `predict.py`: network definition, checkpoint loader, prediction function
- `checkpoints/best_model_NF_and_gain_model_I.pth`: 6-hidden-layer Model

The saved checkpoints include input scaling and output scaling; the code reuses them rather than refitting any scalers. The model is loaded only on the first call and reused thereafter. Each call returns `(NF_dB, max_gain_dB)`.

**Important:** These are ML predictions, not direct Python numerical simulation outputs.

