"""Classical PSA NF/gain port of the supplied MATLAB files.

Implements the numerical models translated from MATLAB into Python.
Model I: compute_nf_and_maxgain.m (Inoue ODE + transfer matrix).
Model II: NLSE branch of generatemldataset_compare_inoue_nlse.m.
The original dispersion, phase factors, FFT grid, and segment sampling are
preserved; this module does not load trained neural-network weights.
"""
import numpy as np
from scipy.integrate import solve_ivp


def compute_nf_and_maxgain(pump_power_dBm, pump_signal_sep_nm,
                           beta2_ps2_per_km, lambda_zdw_nm, model,
                           *, fiber_length_m=300.0, gamma_Winv_km=12.0):
    """Return (NF_dB, max_gain_dB), two Python floats, for one input case.

    pump_power_dBm: power of EACH pump, in dBm (not total power or watts).
    pump_signal_sep_nm: positive pump/signal separation in nm. Model I
        uses exact wavelength allocation; II uses the MATLAB frequency
        approximation df=c*sep/lambda_signal**2.
    beta2_ps2_per_km: signed beta2 in ps^2/km.
    lambda_zdw_nm: zero-dispersion wavelength in nm.
    model: 'I' or 'II' (case-insensitive).
    Optional fiber length [m] and gamma [(W km)^-1] default to training values.

    I: signal at lambda_zdw_nm; quadratic dispersion; 1500 z samples;
       RK45 rtol=1e-8, atol=1e-10, max_step=1 m.
    II: signal at 1550 nm; cubic dispersion with beta3 set by lambda_zdw;
        symmetric SSFM, 8192 time samples, max(1000,ceil(L/0.25)) steps.
        Preserves cs=2j*gamma*a1*a2*exp(-2j*dBs*z)*dz from MATLAB.
    No gain threshold/filter from dataset generation is applied.
    """
    model = str(model).strip().upper()
    if model not in ('I', 'II'):
        raise ValueError("model must be 'I' or 'II'")
    vals = np.asarray([pump_power_dBm, pump_signal_sep_nm,
                       beta2_ps2_per_km, lambda_zdw_nm,
                       fiber_length_m, gamma_Winv_km], dtype=float)
    if vals.shape != (6,) or not np.all(np.isfinite(vals)):
        raise ValueError('Inputs must be finite real scalars')
    power, sep, b2_input, lambda0, length, g_input = vals
    if sep <= 0 or lambda0 <= 0 or length < 0 or g_input < 0:
        raise ValueError('Separation and wavelength must be positive; L and gamma nonnegative')
    c = 299792458.0
    gamma = g_input / 1000
    beta2 = b2_input * 1e-27
    with np.errstate(over='ignore', under='ignore'):
        p0 = float(10.0 ** ((power - 30) / 10))
    if not np.isfinite(p0) or p0 <= 0:
        raise ValueError('Pump power is outside the representable range')
    if model == 'I':
        fs = c / (lambda0 * 1e-9)
        if lambda0 <= sep:
            raise ValueError('Model I requires lambda_zdw_nm > separation')
        fp1 = c / ((lambda0-sep)*1e-9)
        fp2 = 2*fs-fp1
        w0 = 2*np.pi*c/(lambda0*1e-9)
        def beta(f):
            return 0.5*beta2*(2*np.pi*f-w0)**2
        z = np.linspace(0, length, 1500)
        phase_factor = 1
    else:
        fs = c / (1550e-9)
        df = c*(sep*1e-9)/(1550e-9)**2
        fp1, fp2 = fs-df, fs+df
        delta_omega = 2*np.pi*c/(lambda0*1e-9)-2*np.pi*fs
        beta3 = 0.0 if abs(delta_omega) < 1e9 else -beta2/delta_omega
        def beta(f):
            dw = 2*np.pi*(f-fs)
            return 0.5*beta2*dw**2 + beta3*dw**3/6
        steps = max(1000, int(np.ceil(length/0.25)))
        z = np.arange(steps+1)*(length/steps)
        phase_factor = 2
    fi1, fi2 = 2*fp1-fs, 2*fp2-fs
    fF1, fF2 = 2*fp1-fp2, 2*fp2-fp1
    if min(fp1, fp2, fi1, fi2, fF1, fF2) <= 0:
        raise ValueError('Invalid pump/idler frequency allocation')
    if length == 0:
        return 0.0, 0.0
    if model == 'I':
        dBF10 = 2*beta(fp1)-beta(fp2)-beta(fF1)
        dBF20 = 2*beta(fp2)-beta(fp1)-beta(fF2)
        def rhs(zz, y):
            a1, a2, aF1, aF2 = y
            d1 = dBF10-gamma*(2*abs(a1)**2-abs(a2)**2)
            d2 = dBF20-gamma*(2*abs(a2)**2-abs(a1)**2)
            e1, e2 = np.exp(1j*d1*zz), np.exp(1j*d2*zz)
            return np.array([
                2j*gamma*aF1*a2*np.conj(a1)*np.conj(e1)+1j*gamma*a2**2*np.conj(aF2)*e2,
                2j*gamma*aF2*a1*np.conj(a2)*np.conj(e2)+1j*gamma*a1**2*np.conj(aF1)*e1,
                1j*gamma*a1**2*np.conj(a2)*e1,
                1j*gamma*a2**2*np.conj(a1)*e2])
        sol = solve_ivp(rhs, (0, length), np.array([np.sqrt(p0), np.sqrt(p0), 0, 0], complex),
                        t_eval=z, method='RK45', rtol=1e-8, atol=1e-10, max_step=1.0)
        if not sol.success:
            raise RuntimeError('Pump ODE failed: '+sol.message)
        a1, a2 = sol.y[:2]
    else:
        a1, a2 = _nlse_pumps(z, gamma, beta2, beta3, fs, fp1, fp2, fF1, fF2, p0)
    return _transfer(a1, a2, z, gamma, beta, fs, fp1, fp2, fi1, fi2, phase_factor)


def _nlse_pumps(z, gamma, beta2, beta3, fs, fp1, fp2, fF1, fF2, p0):
    dz = z[1]-z[0]
    nt = 8192
    fspan = 2.5*max(abs(f-fs) for f in (fF1, fp1, fp2, fF2))
    dt = 1/(2*fspan)
    indices = np.arange(-nt//2, nt//2)
    t = indices*dt
    omega = 2*np.pi*indices/(nt*dt)
    e1 = np.exp(1j*2*np.pi*(fp1-fs)*t)
    e2 = np.exp(1j*2*np.pi*(fp2-fs)*t)
    A = np.sqrt(p0)*(e1+e2)
    half_linear = np.exp(1j*(0.5*beta2*omega**2+beta3*omega**3/6)*dz/2)
    a1, a2 = np.empty(len(z), complex), np.empty(len(z), complex)
    def project(k):
        a1[k] = np.mean(A*np.conj(e1))
        a2[k] = np.mean(A*np.conj(e2))
    def linear(A):
        Aw = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A)))
        return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Aw*half_linear)))
    project(0)
    for k in range(len(z)-1):
        A = linear(A)
        A *= np.exp(1j*gamma*np.abs(A)**2*dz)
        A = linear(A)
        project(k+1)
    return a1, a2


def _transfer(a1, a2, z, gamma, beta, fs, fp1, fp2, fi1, fi2, phase_factor):
    M = np.eye(6, dtype=complex)
    bs = beta(fp1)+beta(fp2)-2*beta(fs)
    bi1 = 2*beta(fp1)-beta(fs)-beta(fi1)
    bi2 = 2*beta(fp2)-beta(fs)-beta(fi2)
    for k, dz in enumerate(np.diff(z)):
        p1, p2 = abs(a1[k])**2, abs(a2[k])**2
        cs = 2j*gamma*a1[k]*a2[k]*np.exp(-phase_factor*1j*(bs-gamma*(p1+p2))*z[k])*dz
        ci1 = 1j*gamma*a1[k]**2*np.exp(-1j*(bi1-2*gamma*p1)*z[k])*dz
        ci2 = 1j*gamma*a2[k]**2*np.exp(-1j*(bi2-2*gamma*p2)*z[k])*dz
        Mk = np.array([
            [1, cs, 0, ci1, 0, ci2],
            [np.conj(cs), 1, np.conj(ci1), 0, np.conj(ci2), 0],
            [0, ci1, 1, 0, 0, 0],
            [np.conj(ci1), 0, 0, 1, 0, 0],
            [0, ci2, 0, 0, 1, 0],
            [np.conj(ci2), 0, 0, 0, 0, 1]], complex)
        M = Mk @ M
    C = M[0]
    gain = (abs(C[0])+abs(C[1]))**2
    if not np.isfinite(gain) or gain <= 0:
        raise RuntimeError('Computed invalid gain')
    nf = 1+np.sum(np.abs(C[2:])**2)/gain
    if not np.isfinite(nf) or nf <= 0:
        raise RuntimeError('Computed invalid NF')
    return float(10*np.log10(nf)), float(10*np.log10(gain))


if __name__ == '__main__':
    for selected_model in ('I', 'II'):
        nf, gain = compute_nf_and_maxgain(22, 10, -1, 1550, selected_model)
        print(f'Model {selected_model}: NF = {nf:.6f} dB, max gain = {gain:.6f} dB')
