function [NF_dB, Gmax_dB, C1, C2, C3, C4, C5, C6] = compute_nf_and_maxgain( ...
    pumpSignalSep_nm, ...
    fiberLength_m, ...
    gamma_Winv_km, ...
    beta2_ps2_per_km, ...
    zeroDispWavelength_nm, ...
    pumpPower_dBm)
%COMPUTE_NF_AND_MAXGAIN Compute noise figure and maximum PSA gain.
%
% Inputs
%   pumpSignalSep_nm       Pump-signal wavelength separation [nm]
%   fiberLength_m          Fiber length [m]
%   gamma_Winv_km          Nonlinear coefficient [(W km)^-1]
%   beta2_ps2_per_km       beta2 [ps^2/km]
%   zeroDispWavelength_nm  Zero-dispersion wavelength [nm]
%   pumpPower_dBm          Input power of each pump [dBm]
%
% Outputs
%   NF_dB                  Noise figure [dB]
%   Gmax_dB                Maximum PSA gain [dB]
%   C1 ... C6              Complex transfer-matrix coefficients
%
% Numerical/model assumptions preserved from the research source:
%   - beta2 is used directly rather than Dcc.
%   - A second-order dispersion model is used around lambda0.
%   - The signal wavelength is set equal to lambda0.
%   - Pump 1 is pumpSignalSep_nm below the signal wavelength.
%   - Pump 2 follows fs = (fp1 + fp2)/2.
%   - Pump evolution uses the Eq. (14)-type subsystem implemented below.

    validateattributes(pumpSignalSep_nm, {'numeric'}, ...
        {'scalar','real','finite','positive'}, mfilename, 'pumpSignalSep_nm');
    validateattributes(fiberLength_m, {'numeric'}, ...
        {'scalar','real','finite','positive'}, mfilename, 'fiberLength_m');
    validateattributes(gamma_Winv_km, {'numeric'}, ...
        {'scalar','real','finite','positive'}, mfilename, 'gamma_Winv_km');
    validateattributes(beta2_ps2_per_km, {'numeric'}, ...
        {'scalar','real','finite'}, mfilename, 'beta2_ps2_per_km');
    validateattributes(zeroDispWavelength_nm, {'numeric'}, ...
        {'scalar','real','finite','positive'}, mfilename, 'zeroDispWavelength_nm');
    validateattributes(pumpPower_dBm, {'numeric'}, ...
        {'scalar','real','finite'}, mfilename, 'pumpPower_dBm');

    % Numerical settings.
    Nsamp_z = 1500;
    maxStep_m = 1.0;

    % Constants and unit conversions.
    c = 299792458;                                      % [m/s]
    lambda0 = zeroDispWavelength_nm * 1e-9;            % [m]
    beta2 = beta2_ps2_per_km * 1e-24 / 1000;           % [s^2/m]
    gamma = gamma_Winv_km / 1000;                      % [W^-1 m^-1]
    Ppump_W = 10.^((pumpPower_dBm - 30) / 10);         % [W]

    % Wavelength/frequency allocation.
    lambda_s = lambda0;
    fs = c / lambda_s;

    lambda_p1 = lambda_s - pumpSignalSep_nm * 1e-9;
    if lambda_p1 <= 0
        error('Invalid wavelength allocation: lambda_p1 <= 0.');
    end

    fp1 = c / lambda_p1;

    % Dual-pump frequency condition: fs = (fp1 + fp2)/2.
    fp2 = 2 * fs - fp1;
    if fp2 <= 0
        error('Invalid frequency allocation: fp2 <= 0.');
    end

    % Idlers.
    fi1 = 2 * fp1 - fs;
    fi2 = 2 * fp2 - fs;
    if fi1 <= 0 || fi2 <= 0
        error('Invalid idler frequencies.');
    end

    % High-order pumps.
    fF1 = 2 * fp1 - fp2;
    fF2 = 2 * fp2 - fp1;
    if fF1 <= 0 || fF2 <= 0
        error('Invalid high-order pump frequencies.');
    end

    % Angular frequencies.
    w0 = 2 * pi * c / lambda0;
    ws = 2 * pi * fs;
    wp1 = 2 * pi * fp1;
    wp2 = 2 * pi * fp2;
    wi1 = 2 * pi * fi1;
    wi2 = 2 * pi * fi2;
    wF1 = 2 * pi * fF1;
    wF2 = 2 * pi * fF2;

    % Linear propagation constant model:
    % beta_lin(w) = 0.5 * beta2 * (w - w0)^2.
    beta_lin = @(w) 0.5 * beta2 .* (w - w0).^2;

    % Linear phase mismatch for the pump evolution subsystem.
    DeltaBetaF10 = 2 * beta_lin(wp1) - beta_lin(wp2) - beta_lin(wF1);
    DeltaBetaF20 = 2 * beta_lin(wp2) - beta_lin(wp1) - beta_lin(wF2);

    % Initial conditions.
    Ap1_0 = sqrt(Ppump_W);
    Ap2_0 = sqrt(Ppump_W);
    y0 = [Ap1_0; Ap2_0; 0; 0];

    % Solve pump/FWM ODEs.
    z_eval = linspace(0, fiberLength_m, Nsamp_z);
    opts = odeset( ...
        'RelTol', 1e-8, ...
        'AbsTol', 1e-10, ...
        'MaxStep', maxStep_m);

    odefun = @(z, y) rhs_pump_subsystem( ...
        z, y, gamma, DeltaBetaF10, DeltaBetaF20);

    [z, y] = ode45(odefun, z_eval, y0, opts);

    Ap1_z = y(:, 1);
    Ap2_z = y(:, 2);

    % Build the total transfer matrix from segment matrices.
    numSegments = length(z) - 1;
    M_total = eye(6);

    for k = 1:numSegments
        zk = z(k);
        dzk = z(k + 1) - z(k);

        Ap1k = Ap1_z(k);
        Ap2k = Ap2_z(k);

        Pp1k = abs(Ap1k)^2;
        Pp2k = abs(Ap2k)^2;

        DeltaBeta_s = ...
            (beta_lin(wp1) + beta_lin(wp2) - 2 * beta_lin(ws)) ...
            - gamma * (Pp1k + Pp2k);

        DeltaBeta_i1 = ...
            (2 * beta_lin(wp1) - beta_lin(ws) - beta_lin(wi1)) ...
            - 2 * gamma * Pp1k;

        DeltaBeta_i2 = ...
            (2 * beta_lin(wp2) - beta_lin(ws) - beta_lin(wi2)) ...
            - 2 * gamma * Pp2k;

        c_s = 2i * gamma * Ap1k * Ap2k ...
            * exp(-1i * DeltaBeta_s * zk) * dzk;

        c_i1 = 1i * gamma * (Ap1k^2) ...
            * exp(-1i * DeltaBeta_i1 * zk) * dzk;

        c_i2 = 1i * gamma * (Ap2k^2) ...
            * exp(-1i * DeltaBeta_i2 * zk) * dzk;

        Mk = [ ...
            1,           c_s,         0,           c_i1,        0,           c_i2; ...
            conj(c_s),   1,           conj(c_i1),  0,           conj(c_i2),  0; ...
            0,           c_i1,        1,           0,           0,           0; ...
            conj(c_i1),  0,           0,           1,           0,           0; ...
            0,           c_i2,        0,           0,           1,           0; ...
            conj(c_i2),  0,           0,           0,           0,           1];

        M_total = Mk * M_total;
    end

    % Extract C1...C6 and compute NF/gain.
    C1 = M_total(1, 1);
    C2 = M_total(1, 2);
    C3 = M_total(1, 3);
    C4 = M_total(1, 4);
    C5 = M_total(1, 5);
    C6 = M_total(1, 6);

    Gmax = (abs(C1) + abs(C2))^2;
    NF = 1 + ...
        (abs(C3)^2 + abs(C4)^2 + abs(C5)^2 + abs(C6)^2) / Gmax;

    if ~isfinite(Gmax) || Gmax <= 0
        error('Computed invalid gain.');
    end

    if ~isfinite(NF) || NF <= 0
        error('Computed invalid NF.');
    end

    Gmax_dB = 10 * log10(real(Gmax));
    NF_dB = 10 * log10(real(NF));
end


function dydz = rhs_pump_subsystem( ...
    z, y, gamma, DeltaBetaF10, DeltaBetaF20)
%RHS_PUMP_SUBSYSTEM Pump/FWM subsystem used by compute_nf_and_maxgain.

    Ap1 = y(1);
    Ap2 = y(2);
    AF1 = y(3);
    AF2 = y(4);

    DeltaBetaF1 = DeltaBetaF10 ...
        - gamma * (2 * abs(Ap1)^2 - abs(Ap2)^2);

    DeltaBetaF2 = DeltaBetaF20 ...
        - gamma * (2 * abs(Ap2)^2 - abs(Ap1)^2);

    dAp1 = ...
        2i * gamma * AF1 * Ap2 * conj(Ap1) ...
        * exp(-1i * DeltaBetaF1 * z) ...
        + 1i * gamma * (Ap2^2) * conj(AF2) ...
        * exp(1i * DeltaBetaF2 * z);

    dAp2 = ...
        2i * gamma * AF2 * Ap1 * conj(Ap2) ...
        * exp(-1i * DeltaBetaF2 * z) ...
        + 1i * gamma * (Ap1^2) * conj(AF1) ...
        * exp(1i * DeltaBetaF1 * z);

    dAF1 = 1i * gamma * (Ap1^2) * conj(Ap2) ...
        * exp(1i * DeltaBetaF1 * z);

    dAF2 = 1i * gamma * (Ap2^2) * conj(Ap1) ...
        * exp(1i * DeltaBetaF2 * z);

    dydz = [dAp1; dAp2; dAF1; dAF2];
end
