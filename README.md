# Deep-Learning Surrogate Model for Quantum-Noise Computation in Fiber-Optic Phase-Sensitive Amplifiers

This repository contains the numerical simulation codes, datasets, deep-learning models, optimization scripts, and prediction tools developed as part of the research work presented in the paper:

**"Deep-learning surrogate model for rapid quantum-noise computation in dual-pump fiber-optic phase-sensitive amplifiers"**

**Authors:** Yousra Bouasria, Debanuj Chatterjee, Abdessamad Belfakir, and Debabrota Bas

## Project Overview

This project investigates the application of deep learning to the rapid prediction of quantum-noise in dual-pump fiber-optic phase-sensitive amplifiers (PSAs).

The primary objective is to develop deep-learning surrogate models capable of predicting two important amplifier characteristics:

- **Noise Figure (NF)**, expressed in dB.
- **Maximum Gain**, expressed in dB.

The surrogate models are trained using numerical datasets generated from two physical modeling approaches, referred to throughout the repository as **Model I** and **Model II**.

The repository includes the complete computational workflow, from numerical data generation and deep-learning model development to model evaluation, clustering-based analysis, inverse design, and prediction.

## Repository Structure

The repository is organized into the following directories:

### 1. `Numerical_data_generation/`

Contains MATLAB scripts used to perform numerical simulations and generate the datasets required for the machine-learning models.

The numerical simulations are based on the two physical approaches, Model I and Model II.

These scripts are used to compute the noise figure and maximum gain for different amplifier input parameters.

### 2. `Data/`

Contains the numerical datasets used in the machine-learning workflow.

These datasets provide the input parameters and corresponding noise figure and maximum gain values obtained from numerical simulations.

### 3. `RayTune_Optimization/ray_tune_optuna_asha/`

Contains the hyperparameter optimization workflow for the deep-learning models.

The optimization is performed using:

- Ray Tune for experiment management and hyperparameter search.
- Optuna for hyperparameter sampling.
- ASHA (Asynchronous Successive Halving Algorithm) for early stopping of less promising trials.

The objective is to identify suitable neural-network hyperparameters for predicting the noise figure and maximum gain.

The optimization workflow supports GPU acceleration when available and can also run on a CPU.

### 4. `DNN_model_I/`

Contains the deep neural network implementation for **Model I**, including the scripts for training and evaluating the surrogate model.

The neural network learns the relationship between the amplifier input parameters and the corresponding noise figure and maximum gain values generated using Model I.

### 5. `DNN_model_II/`

Contains the deep neural network implementation for **Model II**, including the scripts for training and evaluating the corresponding surrogate model.

This model is trained using the numerical data associated with Model II.

### 6. `Num_Simulation_ML_comparison_check/`

Contains Python scripts used to compare predictions from the trained deep-learning surrogate models with the corresponding numerical simulation results.

The comparison checks the consistency between numerical simulations and machine-learning predictions for a selected set of 100 input instances.


### 7. `Clustering/`

Contains the clustering and inverse-design workflow based on the numerical and machine-learning results.

The analysis uses KMeans clustering to investigate regions of the amplifier parameter space associated with different noise figure and gain characteristics.

It includes:

- Clustering analysis and diagnostic plots.
- Comparison of Model I and Model II.
- Identification of promising operating regions.
- Inverse-design analysis for desired output characteristics.
- Visualization and export of numerical results.

### 8. `Ready_to_use_prediction_function/`

Contains ready-to-use Python prediction functions and trained neural-network checkpoints for Model I and Model II.

These tools allow users to obtain predictions of noise figure and maximum gain from the amplifier input parameters without retraining the neural networks or running the complete numerical simulations.

## Computational Workflow

The overall computational workflow can be summarized as follows:

1. **Numerical simulation:** Generate amplifier datasets using the physical models implemented in MATLAB.
2. **Hyperparameter optimization:** Determine suitable neural-network configurations using Ray Tune, Optuna, and ASHA.
3. **Deep-learning training:** Train surrogate models for Model I and Model II.
4. **Model verification:** Compare the deep-learning predictions with the numerical simulation results.
5. **Clustering and inverse design:** Analyze the amplifier parameter space and identify operating regions corresponding to desired noise figure and gain characteristics.
6. **Prediction:** Use the trained surrogate models to rapidly predict amplifier performance.

## Software and Dependencies

The repository uses MATLAB and Python.

The Python workflows rely on scientific computing and machine-learning libraries, including PyTorch, NumPy, SciPy, scikit-learn, Matplotlib, and, where required, Ray Tune and Optuna.

Dependencies may vary between directories. Please consult the individual README files and available `requirements.txt` files for the relevant installation and execution instructions.

## Usage

Each directory contains a dedicated README describing its purpose, associated scripts, required inputs, and execution procedure.

Users are encouraged to consult the README of the relevant directory before running the corresponding code.

For direct predictions using the pretrained surrogate models, refer to `Ready_to_use_prediction_function/`.

For numerical simulation and dataset generation, refer to `Numerical_data_generation/`.

## Associated Research Paper

The codes and computational workflows in this repository are associated with the research paper:

**Yousra Bouasria, Debanuj Chatterjee, Abdessamad Belfakir, and Debabrota Bas**

*"Deep-learning surrogate model for rapid quantum-noise computation in dual-pump fiber-optic phase-sensitive amplifiers"*

