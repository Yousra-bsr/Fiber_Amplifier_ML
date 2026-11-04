"""Generate a PSA dataset based on a semi-quantum approach."""


import argparse
import math
import time
from pathlib import Path

import numpy as np


C0 = 299792458.0
TWO_PI = 2.0 * math.pi
SIGNAL_NM = 1550.0
OMEGA_SIGNAL = TWO_PI * C0 / (SIGNAL_NM * 1e-9)

INPUT_COLS = (
    "pump_signal_sep_nm",
    "pump_power_dBm",
    "Dcc_ps",
    "zero_disp_wavelength_nm",
)
OUTPUT_COLS = ("NF_dB", "max_gain_dB")
COEFFICIENT_COLS = ("C1", "C2", "C3", "C4", "C5", "C6")


def beta_of_f(f, f_ref, beta2, beta3):
    """Dispersion expansion around the signal frequency."""
    dw = TWO_PI * (f - f_ref)
    return 0.5 * beta2 * dw**2 + (beta3 / 6.0) * dw**3


def inoue_rhs(z, Y, gamma, beta2, beta3, fs, fp1, fp2, fF1, fF2):
    """Derivatives of the two pump fields and two higher-order fields."""
    Ap1, Ap2, AF1, AF2 = Y
    P1 = abs(Ap1)**2
    P2 = abs(Ap2)**2

    dBF10 = (
        2 * beta_of_f(fp1, fs, beta2, beta3)
        - beta_of_f(fp2, fs, beta2, beta3)
        - beta_of_f(fF1, fs, beta2, beta3)
    )
    dBF20 = (
        2 * beta_of_f(fp2, fs, beta2, beta3)
        - beta_of_f(fp1, fs, beta2, beta3)
        - beta_of_f(fF2, fs, beta2, beta3)
    )
    dBF1 = dBF10 - gamma * (2 * P1 - P2)
    dBF2 = dBF20 - gamma * (2 * P2 - P1)

    dAp1 = (
        2j * gamma * AF1 * Ap2 * np.conj(Ap1) * np.exp(-1j * dBF1 * z)
        + 1j * gamma * Ap2**2 * np.conj(AF2) * np.exp(1j * dBF2 * z)
    )
    dAp2 = (
        2j * gamma * AF2 * Ap1 * np.conj(Ap2) * np.exp(-1j * dBF2 * z)
        + 1j * gamma * Ap1**2 * np.conj(AF1) * np.exp(1j * dBF1 * z)
    )
    dAF1 = 1j * gamma * Ap1**2 * np.conj(Ap2) * np.exp(1j * dBF1 * z)
    dAF2 = 1j * gamma * Ap2**2 * np.conj(Ap1) * np.exp(1j * dBF2 * z)

    return np.array([dAp1, dAp2, dAF1, dAF2], dtype=np.complex128)


def propagate_pumps_inoue(zVec, gamma, beta2, beta3,
                          fs, fp1, fp2, fF1, fF2, P0):
    """Propagate the four fields using fixed-step RK4."""
    nSteps = len(zVec) - 1
    dz = zVec[1] - zVec[0]

    Y = np.array([math.sqrt(P0), math.sqrt(P0), 0j, 0j], dtype=np.complex128)
    Ap1 = np.zeros(nSteps + 1, dtype=np.complex128)
    Ap2 = np.zeros(nSteps + 1, dtype=np.complex128)
    Ap1[0], Ap2[0] = Y[0], Y[1]

    for k in range(nSteps):
        z = zVec[k]
        k1 = inoue_rhs(z, Y, gamma, beta2, beta3, fs, fp1, fp2, fF1, fF2)
        k2 = inoue_rhs(z + dz / 2, Y + dz * k1 / 2,
                       gamma, beta2, beta3, fs, fp1, fp2, fF1, fF2)
        k3 = inoue_rhs(z + dz / 2, Y + dz * k2 / 2,
                       gamma, beta2, beta3, fs, fp1, fp2, fF1, fF2)
        k4 = inoue_rhs(z + dz, Y + dz * k3,
                       gamma, beta2, beta3, fs, fp1, fp2, fF1, fF2)

        Y = Y + dz * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        Ap1[k + 1], Ap2[k + 1] = Y[0], Y[1]

    return Ap1, Ap2


def transfer_matrix_nf_gain(Ap1, Ap2, zVec, gamma, beta2, beta3,
                            fs, fp1, fp2, fi1, fi2):
    """Build the 6x6 transfer matrix and calculate NF, gain and |C1| to |C6|."""
    nSteps = len(zVec) - 1
    dz = zVec[1] - zVec[0]
    M = np.eye(6, dtype=np.complex128)

    for k in range(nSteps):
        z = zVec[k]
        a1, a2 = Ap1[k], Ap2[k]
        P1, P2 = abs(a1)**2, abs(a2)**2

        dBs = (
            beta_of_f(fp1, fs, beta2, beta3)
            + beta_of_f(fp2, fs, beta2, beta3)
            - 2 * beta_of_f(fs, fs, beta2, beta3)
            - gamma * (P1 + P2)
        )
        dBi1 = (
            2 * beta_of_f(fp1, fs, beta2, beta3)
            - beta_of_f(fs, fs, beta2, beta3)
            - beta_of_f(fi1, fs, beta2, beta3)
            - 2 * gamma * P1
        )
        dBi2 = (
            2 * beta_of_f(fp2, fs, beta2, beta3)
            - beta_of_f(fs, fs, beta2, beta3)
            - beta_of_f(fi2, fs, beta2, beta3)
            - 2 * gamma * P2
        )

        cs = 2j * gamma * a1 * a2 * np.exp(-1j * dBs * z) * dz
        ci1 = 1j * gamma * a1**2 * np.exp(-1j * dBi1 * z) * dz
        ci2 = 1j * gamma * a2**2 * np.exp(-1j * dBi2 * z) * dz

        Mk = np.array([
            [1, cs, 0, ci1, 0, ci2],
            [np.conj(cs), 1, np.conj(ci1), 0, np.conj(ci2), 0],
            [0, ci1, 1, 0, 0, 0],
            [np.conj(ci1), 0, 0, 1, 0, 0],
            [0, ci2, 0, 0, 1, 0],
            [np.conj(ci2), 0, 0, 0, 0, 1],
        ], dtype=np.complex128)
        M = Mk @ M

    C = M[0, :]
    G = (abs(C[0]) + abs(C[1])) ** 2
    NF = 1.0 + sum(abs(C[k])**2 for k in range(2, 6)) / G

    return np.array([
        10.0 * math.log10(float(NF)),
        10.0 * math.log10(float(G)),
        *[abs(coef) for coef in C],
    ], dtype=np.float64)


def compute_case_inoue(x, length_m=300.0, gamma_km=12.0,
                       max_step_m=0.25):
    """Calculate one case from [separation, power, Dcc, ZDW]."""
    sep_nm, power_dbm, Dcc_ps, lambda0_nm = [float(v) for v in x]

    fs = C0 / (SIGNAL_NM * 1e-9)
    df = C0 * (sep_nm * 1e-9) / (SIGNAL_NM * 1e-9)**2
    fp1, fp2 = fs - df, fs + df
    fF1, fF2 = 2 * fp1 - fp2, 2 * fp2 - fp1
    fi1, fi2 = 2 * fp1 - fs, 2 * fp2 - fs

    gamma = gamma_km / 1000.0
    P0 = 1e-3 * 10**(power_dbm / 10.0)

    # Convert Dcc and the zero-dispersion wavelength into beta3 and beta2.
    lambda0_m = lambda0_nm * 1e-9
    omega0 = TWO_PI * C0 / lambda0_m
    Dcc_SI = Dcc_ps * 1e3
    beta3 = Dcc_SI * lambda0_m**4 / (TWO_PI * C0)**2
    beta2 = beta3 * (OMEGA_SIGNAL - omega0)

    nSteps = max(1000, math.ceil(length_m / max_step_m))
    zVec = np.linspace(0.0, length_m, nSteps + 1)

    Ap1, Ap2 = propagate_pumps_inoue(
        zVec, gamma, beta2, beta3, fs, fp1, fp2, fF1, fF2, P0
    )
    return transfer_matrix_nf_gain(
        Ap1, Ap2, zVec, gamma, beta2, beta3, fs, fp1, fp2, fi1, fi2
    )


def generate_inoue_dataset(N=300000, out=Path("dataset_300000.csv"),
                           seed=42, length_m=300.0, gamma_km=12.0,
                           max_step_m=0.25, save_coefficients=False,
                           progress_every=1000):
    """Draw all input samples first, then simulate each row separately."""
    if N <= 0 or max_step_m <= 0 or length_m <= 0 or gamma_km <= 0:
        raise ValueError("N, length_m, gamma_km, and max_step_m must be positive")

    rng = np.random.default_rng(seed)
    inputData = np.zeros((N, 4), dtype=np.float64)
    inputData[:, 0] = rng.uniform(2.0, 25.0, N)
    inputData[:, 1] = rng.uniform(19.0, 25.0, N)
    inputData[:, 2] = rng.uniform(0.01, 0.07, N)
    inputData[:, 3] = rng.uniform(1500.0, 1600.0, N)

    outputInoue = np.full((N, 8), np.nan, dtype=np.float64)
    start = time.perf_counter()
    steps = max(1000, math.ceil(length_m / max_step_m))
    print(f"Cases: {N:,}; seed={seed}; spatial steps/case={steps:,}; "
          "processing ONE configuration at a time", flush=True)

    for n in range(N):
        outputInoue[n, :] = compute_case_inoue(
            inputData[n, :], length_m, gamma_km, max_step_m
        )
        if (n + 1) % progress_every == 0 or n + 1 == N:
            elapsed = time.perf_counter() - start
            print(f"{n + 1:,} / {N:,}; elapsed {elapsed:.1f} s", flush=True)

    if not np.all(np.isfinite(outputInoue)):
        raise RuntimeError("Nonfinite outputs. CSV export aborted.")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = np.column_stack((inputData, outputInoue[:, :2]))
    np.savetxt(out, data, delimiter=",", fmt="%.12g",
               header=",".join(INPUT_COLS + OUTPUT_COLS), comments="")
    print(f"Saved DNN-ready dataset: {out.resolve()}")

    if save_coefficients:
        full_out = out.with_name(out.stem + "_with_coefficients" + out.suffix)
        full_data = np.column_stack((inputData, outputInoue))
        np.savetxt(full_out, full_data, delimiter=",", fmt="%.12g",
                   header=",".join(INPUT_COLS + OUTPUT_COLS + COEFFICIENT_COLS),
                   comments="")
        print(f"Saved full output (NF/gain/C1-C6): {full_out.resolve()}")

    print(f"Complete: {N:,} rows in {time.perf_counter() - start:.2f} s")
    return inputData, outputInoue


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=300000)
    parser.add_argument("--out", type=Path, default=Path("dataset_300000.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length-m", type=float, default=300.0)
    parser.add_argument("--gamma-km", type=float, default=12.0)
    parser.add_argument("--max-step-m", type=float, default=0.25)
    parser.add_argument("--save-coefficients", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")

    generate_inoue_dataset(
        args.n, args.out, args.seed, args.length_m,
        args.gamma_km, args.max_step_m,
        args.save_coefficients, args.progress_every
    )


if __name__ == "__main__":
    main()
