# Dataset Generation for Dual-Pump Fiber-Optic Phase-Sensitive Amplifiers

This script generates numerical data for a dual-pump fiber-optic phase-sensitive amplifier (PSA) using a semi-quantum approach. 

The main CSV contains four input parameters and two targets (`NF_dB` and `max_gain_dB`) for machine-learning training.

## Files

```text
.
├── generate_train_test_datasets.py  # Numerical dataset generator
├── README.md                        # Documentation
├── requirements.txt                 # Required Python packages
└── .gitignore                       # Local Python files and temporary output
```

## Method

1. Generate all four input columns with uniform random sampling before running the simulations.
2. For each input row, calculate the pump frequencies, nonlinear coefficient, and dispersion coefficients. `beta2` and `beta3` are calculated from `Dcc` and the zero-dispersion wavelength.
3. Propagate the two pump fields and two higher-order fields with an explicit fixed-step, fourth-order Runge–Kutta (RK4) method.
4. At each propagation step, build the local 6 × 6 transfer matrix and update the full matrix as `M = Mk @ M`.
5. Use the first row of the final matrix to calculate the noise figure, maximum gain, and the magnitudes of six transfer coefficients.
6. Save the results to CSV.

## Input ranges

All inputs are sampled independently from uniform distributions.

| CSV column | Range | Unit |
| --- | --- | --- |
| `pump_signal_sep_nm` | 2 to 25 | nm |
| `pump_power_dBm` | 19 to 25 | dBm per pump |
| `Dcc_ps` | 0.01 to 0.07 | ps/(nm² km) |
| `zero_disp_wavelength_nm` | 1500 to 1600 | nm |

The signal wavelength is fixed at **1550 nm**.

## Default simulation settings

| Setting | Value |
| --- | --- |
| Number of samples | 300,000 |
| Random seed | 42  |
| Fiber length | 300 m |
| Nonlinear coefficient | 12 W⁻¹ km⁻¹ |
| Maximum spatial step | 0.25 m |
| Progress update | Every 1,000 samples |

The number of propagation steps is `max(1000, ceil(length_m / max_step_m))`. This gives **1,200 steps for the default 300 m fiber**. Each case runs in a Python loop, so a large dataset can take considerable time to generate.

## Installation

Python 3 and NumPy are required. From this folder, install the dependency with:

```bash
python -m pip install -r requirements.txt
```

## Run

Generate a small test dataset:

```bash
python generate_train_test_dataset.py --n 5 --out quick_check.csv
```

Generate the default 300,000 samples:

```bash
python generate_train_test_dataset.py
```

Generate 50,000 samples with a different random seed:

```bash
python ggenerate_train_test_dataset.py --n 50000 --seed 20 --out dataset_50000.csv
```

Save the six transfer-coefficient magnitudes in a second CSV:

```bash
python generate_train_test_dataset.py --n 100 --save-coefficients
```

You can also change the fiber length or spatial step using `--length-m` and `--max-step-m`.

## Output files

By default, `dataset_300000.csv` has six columns:

```text
pump_signal_sep_nm,pump_power_dBm,Dcc_ps,zero_disp_wavelength_nm,NF_dB,max_gain_dB
```

Here, `NF_dB` is the calculated noise figure in dB, and `max_gain_dB` is the maximum gain in dB.

With `--save-coefficients`, the script also creates `dataset_300000_with_coefficients.csv` (or a matching name based on `--out`). It contains the four inputs, the two targets, and `C1` to `C6`. These last six columns store the **magnitudes** of the complex transfer-matrix coefficients, not the complex values themselves.

Files are saved in the path selected by `--out`, and the script creates the output directory if it does not exist. All results must be finite for CSV export to proceed.

