# Data

This project expects the following project-owned CSV datasets in this directory:

- `inputdata_nlse_300000.csv` — training/validation dataset
- `dataTest_nlse_40000.csv` — external test dataset

Required columns:

## Inputs

- `pump_signal_sep_nm`
- `pump_power_dBm`
- `beta2_ps2_per_km`
- `zero_disp_wavelength_nm`

## Targets

- `NF_dB`
- `max_gain_dB`

These CSV files are intentionally tracked by Git because the data is intended
to be shared together with the source code.
