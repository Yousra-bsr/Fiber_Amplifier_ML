# DNN versus Numerical/classical PSA comparison

This code compares the predictions of the trained deep neural network (DNN) models with the corresponding MATLAB-based numerical models (Model I and Model II). 
Model I uses the Inoue pump ODE and transfer matrix, while model II
uses the NLSE/SSFM branch. 

The objective is to evaluate whether the ML models reproduce the numerical results for **100 randomly generated input configurations**.

For each configuration, the code calculates and compares two outputs:
- Noise figure (`NF_dB`)
- Maximum gain (`max_gain_dB`)

The differences between the ML and numerical predictions are evaluated to assess the prediction accuracy of both DNN models.

## Files

- `compare_psa_models.py`: Runs the comparison for 100 input configurations and saves the results.
- `predict.py`: Loads the trained DNN models and generates ML predictions.
- `psa_nf_gain.py`: Implements the numerical models translated from MATLAB into Python.

## Usage

Install the required dependencies:

```bash
pip install numpy scipy torch matplotlib
```

Run the comparison:

```bash
python compare_psa_models.py --n 100 --seed 42
```

Each CSV row contains the four inputs and eight outputs: NF and maximum gain
from DNN I, classical I, DNN II and classical II, all in dB. Four error columns
are DNN minus classical within each model. Four further columns compare model
II minus model I within DNN and classical calculations separately. The console
displays all eight outputs and the four DNN-versus-classical errors per case.

Sampling follows Table I of the manuscript: independent uniform draws of power per
pump 19-25 dBm, separation 4-16 nm, beta2 -10 to -0.1 ps^2/km and ZDW 1500-1600 nm.
Fiber length is 300 m and gamma is 12 (W km)^-1. Seed 42 makes this run reproducible;
it does not reproduce the MATLAB random-number stream or the training samples.

## Outputs

The results are saved in the `results/` folder:

- `comparison_results.csv`: Numerical and ML predictions for each input configuration, one row per sampled case.
- `fig1_gain_error.png`: Differences in maximum gain predictions, signed gain errors; red I, blue II.
- `fig2_nf_error.png`: Differences in noise figure predictions, signed NF errors; red I, blue II.
- `summary.json`: Statistical error metrics (MAE, RMSE, mean error, and maximum absolute error).







