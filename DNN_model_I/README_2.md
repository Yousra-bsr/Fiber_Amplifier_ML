# Data

Place the training and optional external-test CSV files in this directory.

Default filenames expected by the training script:

- `datatest_pump_beta2_ZDW_200000.csv`
- `Input_Output20000Data_feat.csv`

The required columns are:

## Inputs

- `pump_signal_sep_nm`
- `pump_power_dBm`
- `beta2_ps2_per_km`
- `zero_disp_wavelength_nm`

## Targets

- `NF_dB`
- `max_gain_dB`

CSV files are ignored by Git by default. This is intentional so that large or
restricted research datasets are not published accidentally.

If you are allowed to redistribute the data and want to include it in the
repository, edit the data-related section of `.gitignore`.
