# MATLAB Numerical Data Generation for PSA Machine Learning

This repository contains the MATLAB numerical models used to generate and
compare datasets for the machine-learning regression workflows.

The repository is configured to share **numerical source code + project-owned
CSV datasets**, while excluding regenerated figures, diagnostic MAT files, and
other numerical outputs.

## Files

```text
matlab/
├── compute_nf_and_maxgain.m
├── generatemldataset.m
├── generatemldataset_compare_inoue_nlse.m
└── Inoue_NLSE_Fig1_Fig2v1.m
```

### `compute_nf_and_maxgain.m`

Computes:

- noise figure `NF_dB`;
- maximum PSA gain `Gmax_dB`;
- complex transfer coefficients `C1` ... `C6`.

Thz assumptions:

- the signal wavelength is set equal to the supplied zero-dispersion wavelength;
- a second-order dispersion model is used around that wavelength;
- dual-pump frequencies satisfy `fs = (fp1 + fp2)/2`;
- the pump/FWM subsystem is solved with `ode45`;
- the six-dimensional transfer matrix is accumulated along the fiber.


### `generatemldataset.m`

Generates a random Inoue-model (Model I) dataset using `compute_nf_and_maxgain`.

Default call:

```matlab
generatemldataset()
```

Current source-compatible defaults:

```text
N = 100
RNG seed = 1
```

For a larger run:

```matlab
generatemldataset(100000, 1)
```

The primary output is one training-ready CSV:

```text
data/datatest_pump_beta2_ZDW_<N>.csv
```

It contains the six sampled physical inputs plus:

```text
NF_dB
max_gain_dB
```

The complex `C1...C6` values are kept in a diagnostic MAT file under
`outputs/inoue_generation/diagnostics/` instead of being mixed into the ML CSV.


### `generatemldataset_compare_inoue_nlse.m`

Generates exactly the same sampled input cases for two pump-propagation models:

1. Inoue four-wave coupled equations; (Model I)
2. scalar NLSE propagated with symmetric split-step Fourier method (SSFM).  (Model II)

Default call:

```matlab
generatemldataset_compare_inoue_nlse()
```

Defaults:

```text
N = 20000
RNG seed = 14
```

For a different run:

```matlab
generatemldataset_compare_inoue_nlse(300000, 14)
```

The two primary ML CSV outputs are:

```text
data/inoue_dataset_<N>.csv
data/inputdata_nlse_<N>.csv
```

Each has the same physical input columns and the standard ML targets:

```text
NF_dB
max_gain_dB
```

Full coefficient-magnitude tables, gain-difference diagnostics, and the
comparison scatter plot are saved under:

```text
outputs/inoue_nlse_comparison/
```


`compute_nf_and_maxgain.m` sets:

```text
signal wavelength = zero-dispersion wavelength
```

and uses a second-order dispersion model.

By contrast, `generatemldataset_compare_inoue_nlse.m` uses:

```text
signal wavelength = 1550 nm
```

## `Inoue_NLSE_Fig1_Fig2v1.m`

This file is for numerical model/figure comparison. It compares:

- pump-1 and high-order-pump evolution;
- Inoue and NLSE propagation;
- NLSE output comb-line spectra.

It does **not** calculate `NF_dB`, `max_gain_dB`, or `C1...C6`.

Run:

```matlab
Inoue_NLSE_Fig1_Fig2v1()
```

Generated PNG, FIG, CSV, and MAT figure-support files are written under:

```text
outputs/figure_comparison/
```

and are ignored by Git.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
└── matlab/
    ├── compute_nf_and_maxgain.m
    ├── generatemldataset.m
    ├── generatemldataset_compare_inoue_nlse.m
    └── Inoue_NLSE_Fig1_Fig2v1.m
```

After numerical runs, the local repository may additionally contain:

```text
outputs/
├── inoue_generation/
├── inoue_nlse_comparison/
└── figure_comparison/
```

`outputs/` remains on the local machine but is excluded from Git.

## MATLAB requirements

No Python packages are required.

MATLAB R2020a or newer is recommended because the figure utility uses functions
such as `tiledlayout` and `exportgraphics`.

The supplied source relies on base MATLAB numerical functionality including:

- `ode45`;
- FFT/IFFT;
- tables and CSV I/O;
- MAT-file I/O;
- plotting and figure export.

No additional MATLAB toolbox dependency is apparent from the supplied code.

## Sampling ranges

The data generators preserve the research-source ranges:

| Parameter | Range / value |
| --- | --- |
| Pump-signal separation | 4 to 16 nm |
| Fiber length | 300 m |
| Nonlinear coefficient | 12 (W km)^-1 |
| beta2 | -0.1 to -10 ps^2/km in the source endpoint order |
| Zero-dispersion wavelength | 1500 to 1600 nm |
| Pump power | 19 to 25 dBm per pump |


## Reproducibility

The default random seeds are preserved:

```text
generatemldataset.m                    -> seed 1
generatemldataset_compare_inoue_nlse.m -> seed 14
Inoue_NLSE_Fig1_Fig2v1.m              -> seed 14
```

The generators now accept `N` and the random seed as function arguments rather
than requiring source-code edits.

## Version-control policy

The `.gitignore` excludes:

- `outputs/`;
- MATLAB autosave files;
- MATLAB code-generation/cache folders;
- editor/OS files.

It intentionally does **not** ignore:

```text
data/*.csv
```

because the generated training datasets are project-owned data intended to be
shared with the ML code.


# Citation

Publication citation, DOI, or BibTeX entry here.
