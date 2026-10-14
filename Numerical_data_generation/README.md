# MATLAB Numerical Data Generation for PSA Machine Learning

These MATLAB scripts generate numerical datasets for training machine-learning models to predict the **noise figure (`NF_dB`)** and **maximum PSA gain (`max_gain_dB`)**.

## Which script generates each dataset?

`generatemldataset.m` | Generates **Model I (Inoue)** training data by calling `compute_nf_and_maxgain.m`.

`compute_nf_and_maxgain.m` | Calculates NF and maximum gain for one **Model I** configuration; it is called by `generatemldataset.m`.

`generatemldataset_compare_inoue_nlse.m` | Generates **both Model I (Inoue) and Model II (NLSE)** datasets from the **same randomly sampled input configurations**. Use its NLSE output for **Model II** training data. 

`Inoue_NLSE_Fig1_Fig2v1.m` | Compares pump evolution and NLSE spectra for figures. **Not used to generate NF/gain training datasets.** 

## Generate Model I data

Run in MATLAB:

```matlab
generatemldataset(100000, 1)
```

This uses the Inoue coupled-wave model through `compute_nf_and_maxgain.m` and saves:

```text
data/datatest_pump_beta2_ZDW_100000.csv
```

The default call `generatemldataset()` generates **100 samples** with random seed **1**. The script also saves transfer-coefficient diagnostics under `outputs/inoue_generation/diagnostics/`.

## Generate Model II data (and a paired Model I dataset)

Run in MATLAB:

```matlab
generatemldataset_compare_inoue_nlse(300000, 14)
```

This calculates both models on identical sampled inputs and saves:

```text
data/inoue_dataset_300000.csv       # Model I
data/inputdata_nlse_300000.csv      # Model II
```

**Model I** uses Inoue coupled pump equations, while **Model II** uses the nonlinear Schrödinger equation (NLSE) solved by the split-step Fourier method (SSFM).

The default call `generatemldataset_compare_inoue_nlse()` generates **20,000 paired samples** with random seed **14**. Comparison figures and additional results are saved in `outputs/inoue_nlse_comparison/`.

**Note:** The Model I calculation in `generatemldataset.m` and the paired comparison script do not use identical numerical setups. The former sets the signal wavelength to the zero-dispersion wavelength and uses second-order dispersion; the paired script sets it to 1550 nm and includes third-order dispersion. 

## Dataset parameters

| Parameter | Range or fixed value |
| --- | --- |
| Pump–signal separation | 4–16 nm |
| Pump power (each pump) | 19–25 dBm |
| Group-velocity dispersion (`beta2`) | −10 to −0.1 ps²/km |
| Zero-dispersion wavelength | 1500–1600 nm |
| Fiber length | 300 m |
| Nonlinear coefficient | 12 (W km)⁻¹ |

Both generators write the input parameters and the two regression targets `NF_dB` and `max_gain_dB` to CSV files in `data/`.

## Additional figure script

Run `Inoue_NLSE_Fig1_Fig2v1()` to compare pump and higher-order-pump propagation and NLSE output spectra. It does not produce NF/gain training targets. Its figures and supporting data are saved under `outputs/figure_comparison/`.
