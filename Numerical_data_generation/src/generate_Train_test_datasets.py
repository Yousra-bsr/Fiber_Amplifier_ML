#
"""Generate training/validation and test datatests using a semi-quantum approach"""


import argparse
import csv
import math
import os
from pathlib import Path
import time

import numpy as np
from numba import njit, prange, set_num_threads

C0 = 299792458.0
TWO_PI = 2.0 * math.pi
SIGNAL_NM = 1550.0
OMEGA_SIGNAL = TWO_PI * C0 / (SIGNAL_NM * 1e-9)

INPUT_COLS = [
    "pump_signal_sep_nm",
    "pump_power_dBm",
    "Dcc_ps",
    "zero_disp_wavelength_nm",
]
TARGET_COLS = ["NF_dB", "max_gain_dB"]
CSV_COLS = INPUT_COLS + TARGET_COLS


@njit(inline="always")
def _beta(offset_omega, beta2, beta3):
    return 0.5 * beta2 * offset_omega ** 2 + (beta3 / 6.0) * offset_omega ** 3


@njit(inline="always")
def _rhs(z, y, out, gamma, bf10, bf20, bs0, bi10, bi20, phase_mode):
    """Original four-wave pump RHS; optional integrated phase variables.

    y[:4] = Ap1, Ap2, AF1, AF2 (complex)
    y[4:9] = theta_F1, theta_F2, theta_s, theta_i1, theta_i2
    phase_mode = 0: literal local mismatch * z
                 1: integrated mismatch phases
    """
    a1, a2, f1, f2 = y[0], y[1], y[2], y[3]
    p1 = a1.real * a1.real + a1.imag * a1.imag
    p2 = a2.real * a2.real + a2.imag * a2.imag

    dbf1 = bf10 - gamma * (2.0 * p1 - p2)
    dbf2 = bf20 - gamma * (2.0 * p2 - p1)
    if phase_mode == 0:
        theta_f1 = dbf1 * z
        theta_f2 = dbf2 * z
    else:
        theta_f1 = y[4].real
        theta_f2 = y[5].real

    e1p = np.exp(1j * theta_f1)
    e2p = np.exp(1j * theta_f2)
    e1m = np.conj(e1p)
    e2m = np.conj(e2p)

    out[0] = (2j * gamma * f1 * a2 * np.conj(a1) * e1m
              + 1j * gamma * a2 * a2 * np.conj(f2) * e2p)
    out[1] = (2j * gamma * f2 * a1 * np.conj(a2) * e2m
              + 1j * gamma * a1 * a1 * np.conj(f1) * e1p)
    out[2] = 1j * gamma * a1 * a1 * np.conj(a2) * e1p
    out[3] = 1j * gamma * a2 * a2 * np.conj(a1) * e2p

    if phase_mode == 1:
        out[4] = complex(dbf1, 0.0)
        out[5] = complex(dbf2, 0.0)
        out[6] = complex(bs0 - gamma * (p1 + p2), 0.0)
        out[7] = complex(bi10 - 2.0 * gamma * p1, 0.0)
        out[8] = complex(bi20 - 2.0 * gamma * p2, 0.0)
    else:
        for k in range(4, 9):
            out[k] = 0.0j


@njit
def _simulate_one(sep_nm, power_dbm, dcc, lambda0_nm,
                  length_m, gamma, n_steps, phase_mode):
    """Return Inoue (NF_dB, max_gain_dB) for one sampled parameter set."""
    lam0 = lambda0_nm * 1e-9
    w0 = TWO_PI * C0 / lam0

    # At zero-dispersion wavelength beta2(w0) = 0.
    # Dcc := dD/dlambda at lambda0 in ps/(nm^2 km).
    # 1 ps / (nm^2 km) = 1e3 s/m^3.
    slope_si = dcc * 1e3
    beta3 = slope_si * lam0**4 / (TWO_PI * C0)**2   # [s^3/m]
    beta2 = beta3 * (OMEGA_SIGNAL - w0)             # [s^2/m]

    # Consistent with MATLAB: wavelength offset converted to symmetric frequency offset.
    delta_f = C0 * (sep_nm * 1e-9) / (SIGNAL_NM * 1e-9)**2
    om = TWO_PI * delta_f

    bp1 = _beta(-om, beta2, beta3)
    bp2 = _beta(+om, beta2, beta3)
    bf1 = _beta(-3.0 * om, beta2, beta3)
    bf2 = _beta(+3.0 * om, beta2, beta3)
    bi1 = _beta(-2.0 * om, beta2, beta3)
    bi2 = _beta(+2.0 * om, beta2, beta3)

    bf10 = 2.0 * bp1 - bp2 - bf1
    bf20 = 2.0 * bp2 - bp1 - bf2
    bs0 = bp1 + bp2  # beta_s = 0 in the signal-centered frame
    bi10 = 2.0 * bp1 - bi1
    bi20 = 2.0 * bp2 - bi2

    dz = length_m / n_steps
    p0 = 1e-3 * (10.0 ** (power_dbm / 10.0))

    y = np.zeros(9, dtype=np.complex128)
    y[0] = math.sqrt(p0)
    y[1] = math.sqrt(p0)
    k1 = np.empty(9, dtype=np.complex128)
    k2 = np.empty(9, dtype=np.complex128)
    k3 = np.empty(9, dtype=np.complex128)
    k4 = np.empty(9, dtype=np.complex128)
    trial = np.empty(9, dtype=np.complex128)

    # Store only the pump values required by each local segment matrix.
    a1_hist = np.empty(n_steps, dtype=np.complex128)
    a2_hist = np.empty(n_steps, dtype=np.complex128)
    ph_s_hist = np.empty(n_steps, dtype=np.float64)
    ph_i1_hist = np.empty(n_steps, dtype=np.float64)
    ph_i2_hist = np.empty(n_steps, dtype=np.float64)

    for step in range(n_steps):
        z = step * dz
        a1_hist[step] = y[0]
        a2_hist[step] = y[1]
        if phase_mode == 1:
            ph_s_hist[step] = y[6].real
            ph_i1_hist[step] = y[7].real
            ph_i2_hist[step] = y[8].real

        _rhs(z, y, k1, gamma, bf10, bf20, bs0, bi10, bi20, phase_mode)
        for j in range(9):
            trial[j] = y[j] + (dz / 2.0) * k1[j]
        _rhs(z + dz / 2.0, trial, k2, gamma, bf10, bf20, bs0, bi10, bi20, phase_mode)
        for j in range(9):
            trial[j] = y[j] + (dz / 2.0) * k2[j]
        _rhs(z + dz / 2.0, trial, k3, gamma, bf10, bf20, bs0, bi10, bi20, phase_mode)
        for j in range(9):
            trial[j] = y[j] + dz * k3[j]
        _rhs(z + dz, trial, k4, gamma, bf10, bf20, bs0, bi10, bi20, phase_mode)
        for j in range(9):
            y[j] += (dz / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j])

    # Uploaded MATLAB uses M = Mk @ M, with M(0)=I.
    # First row of M_N ... M_1 equals e1^T M_N ... M_1, so multiplying
    # a single row backwards (N to 1) is algebraically identical.
    r0 = 1.0 + 0.0j
    r1 = 0.0 + 0.0j
    r2 = 0.0 + 0.0j
    r3 = 0.0 + 0.0j
    r4 = 0.0 + 0.0j
    r5 = 0.0 + 0.0j

    for step in range(n_steps - 1, -1, -1):
        z = step * dz
        a1, a2 = a1_hist[step], a2_hist[step]
        p1 = a1.real*a1.real + a1.imag*a1.imag
        p2 = a2.real*a2.real + a2.imag*a2.imag
        if phase_mode == 0:
            th_s = (bs0 - gamma * (p1 + p2)) * z
            th_i1 = (bi10 - 2.0 * gamma * p1) * z
            th_i2 = (bi20 - 2.0 * gamma * p2) * z
        else:
            th_s = ph_s_hist[step]
            th_i1 = ph_i1_hist[step]
            th_i2 = ph_i2_hist[step]

        # Corrected coefficient: one (not two) mismatch exponent for c_s.
        cs = 2j * gamma * a1 * a2 * np.exp(-1j * th_s) * dz
        ci1 = 1j * gamma * a1 * a1 * np.exp(-1j * th_i1) * dz
        ci2 = 1j * gamma * a2 * a2 * np.exp(-1j * th_i2) * dz

        old0, old1, old2, old3, old4, old5 = r0, r1, r2, r3, r4, r5
        r0 = old0 + old1 * np.conj(cs) + old3 * np.conj(ci1) + old5 * np.conj(ci2)
        r1 = old0 * cs + old1 + old2 * ci1 + old4 * ci2
        r2 = old1 * np.conj(ci1) + old2
        r3 = old0 * ci1 + old3
        r4 = old1 * np.conj(ci2) + old4
        r5 = old0 * ci2 + old5

    gain = (abs(r0) + abs(r1)) ** 2
    noise = 1.0 + (abs(r2)**2 + abs(r3)**2 + abs(r4)**2 + abs(r5)**2) / gain
    return 10.0 * math.log10(noise), 10.0 * math.log10(gain)


@njit(parallel=True)
def _simulate_batch(seps, powers, slopes, lambda0s, length_m, gamma, n_steps, mode):
    n = len(seps)
    nf_db = np.empty(n, dtype=np.float64)
    gain_db = np.empty(n, dtype=np.float64)
    for j in prange(n):
        nf_db[j], gain_db[j] = _simulate_one(
            seps[j], powers[j], slopes[j], lambda0s[j],
            length_m, gamma, n_steps, mode
        )
    return nf_db, gain_db


def self_test():
    # Verify that the sparse row update reproduces the literal 6x6 product.
    rng = np.random.default_rng(3)
    M = np.eye(6, dtype=complex)
    steps = []
    for _ in range(12):
        cs, c1, c2 = (rng.normal(size=3) + 1j*rng.normal(size=3)) * 0.01
        Mk = np.array([
            [1, cs, 0, c1, 0, c2],
            [np.conj(cs), 1, np.conj(c1), 0, np.conj(c2), 0],
            [0, c1, 1, 0, 0, 0],
            [np.conj(c1), 0, 0, 1, 0, 0],
            [0, c2, 0, 0, 1, 0],
            [np.conj(c2), 0, 0, 0, 0, 1],
        ], dtype=complex)
        M = Mk @ M
        steps.append(Mk)
    row = np.zeros(6, dtype=complex)
    row[0] = 1
    for Mk in reversed(steps):
        row = row @ Mk
    assert np.allclose(row, M[0, :], rtol=1e-13, atol=1e-13)

    # Dispersion must change beta2 sign at 1550 nm for Dcc > 0.
    def dispersion(lambda0_nm, slope):
        lambda0 = lambda0_nm * 1e-9
        beta3 = slope*1e3 * lambda0**4 / (TWO_PI*C0)**2
        beta2 = beta3 * (OMEGA_SIGNAL-TWO_PI*C0/lambda0)
        return beta2/1e-27
    assert dispersion(1500, 0.02) < 0
    assert abs(dispersion(1550, 0.02)) < 1e-10
    assert dispersion(1600, 0.02) > 0

    # Verify the JIT produces finite outputs for both mismatch conventions.
    v = np.array([4.0, 10.0], dtype=np.float64)
    powers = np.array([19.0, 23.0])
    slope = np.array([0.02, 0.05])
    zero_nm = np.array([1545.0, 1555.0])
    for mode in (0, 1):
        nf, gain = _simulate_batch(v, powers, slope, zero_nm, 300.0, 0.012, 32, mode)
        assert np.all(np.isfinite(nf)) and np.all(np.isfinite(gain))
    print("Self-test passed: row products, dispersion signs, both phase conventions.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300000, help="Number of samples, default 300000")
    ap.add_argument("--out", type=Path, default=Path("dataset_300000.csv"))
    ap.add_argument("--chunk-size", type=int, default=500, help="Simulation/write batch size")
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--threads", type=int, default=None, help="Number of Numba CPU threads")
    ap.add_argument("--phase-mode", choices=("original", "accumulated"), default="original",
                    help="original exactly follows local DeltaBeta(z)*z in the MATLAB file; accumulated is the B variant")
    ap.add_argument("--pump-min", type=float, default=19.0, help="dBm per pump (uploaded MATLAB default: 12)")
    ap.add_argument("--pump-max", type=float, default=25.0)
    ap.add_argument("--sep-min", type=float, default=2.0, help="nm, approx signal-pump wavelength separation")
    ap.add_argument("--sep-max", type=float, default=25.0)
    ap.add_argument("--lambda0-min", type=float, default=1500.0)
    ap.add_argument("--lambda0-max", type=float, default=1600.0)
    ap.add_argument("--dcc-min", type=float, default=0.01, help="Dispersion slope ps/(nm^2 km)")
    ap.add_argument("--dcc-max", type=float, default=0.07, help="Dispersion slope ps/(nm^2 km)")
    ap.add_argument("--length-m", type=float, default=300.0)
    ap.add_argument("--gamma-km", type=float, default=12.0, help="W^-1 km^-1")
    ap.add_argument("--max-step-m", type=float, default=0.1, help="Max spatial step [m] (MATLAB default 0.25)")
    ap.add_argument("--self-test", action="store_true", help="Check the implementation and quit")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.n <= 0 or args.chunk_size <= 0 or args.max_step_m <= 0:
        ap.error("--n, --chunk-size and --max-step-m must be positive")
    for label, low, high in (
        ("pump", args.pump_min, args.pump_max),
        ("separation", args.sep_min, args.sep_max),
        ("zero-dispersion wavelength", args.lambda0_min, args.lambda0_max),
        ("Dcc", args.dcc_min, args.dcc_max),
    ):
        if not low < high:
            ap.error(f"{label} range requires min < max")
    if args.dcc_min <= 0 or args.lambda0_min <= 0 or args.length_m <= 0 or args.gamma_km <= 0:
        ap.error("Dcc, wavelength, length and gamma must be positive")
    if args.threads is not None:
        set_num_threads(args.threads)

    n_steps = max(1000, math.ceil(args.length_m / args.max_step_m))
    gamma = args.gamma_km / 1_000.0
    mode = 0 if args.phase_mode == "original" else 1
    rng = np.random.default_rng(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".partial")
    t0 = time.perf_counter()

    print(f"Cases: {args.n:,}; phase: {args.phase_mode}; steps/case: {n_steps:,}")
    print(f"Pump range: {args.pump_min:g}–{args.pump_max:g} dBm; Dcc range: {args.dcc_min:g}–{args.dcc_max:g} ps/(nm^2 km)")
    print(f"Output: {args.out.resolve()}")
    print("Compiling the simulation kernel on its first call (Numba JIT)...")

    written = 0
    try:
        with temporary.open("w", newline="", encoding="utf-8") as fh:
            fh.write(",".join(CSV_COLS) + "\n")
            while written < args.n:
                n = min(args.chunk_size, args.n - written)
                sep = rng.uniform(args.sep_min, args.sep_max, n)
                power = rng.uniform(args.pump_min, args.pump_max, n)
                dcc = rng.uniform(args.dcc_min, args.dcc_max, n)
                l0 = rng.uniform(args.lambda0_min, args.lambda0_max, n)
                nf, gain = _simulate_batch(sep, power, dcc, l0,
                                          args.length_m, gamma, n_steps, mode)
                if not np.all(np.isfinite(nf)) or not np.all(np.isfinite(gain)):
                    bad = np.flatnonzero(~(np.isfinite(nf) & np.isfinite(gain)))
                    raise RuntimeError(f"Nonfinite simulation result at dataset row {written + int(bad[0]) + 1}; no completed CSV was written")
                values = np.column_stack((sep, power, dcc, l0, nf, gain))
                np.savetxt(fh, values, delimiter=",", fmt="%.12g")
                written += n
                if written == args.n or written % max(args.chunk_size, 500) == 0:
                    print(f"  Wrote {written:,}/{args.n:,} rows; elapsed {time.perf_counter()-t0:.1f} s", flush=True)
        os.replace(temporary, args.out)
    except Exception:
        print(f"Generation stopped; incomplete data remain at {temporary}")
        raise
    print(f"Complete: {written:,} rows, exact columns {CSV_COLS}")


if __name__ == "__main__":
    main()
