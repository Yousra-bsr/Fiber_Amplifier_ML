function [inoueDatasetPath, nlseDatasetPath] = ...
    generatemldataset_compare_inoue_nlse( ...
        N, randomSeed, dataDir, outputDir)
%GENERATEMLDATASET_COMPARE_INOUE_NLSE Generate paired Inoue/NLSE datasets.
%
% The exact same random physical inputs are propagated with:
%   1) the Inoue four-wave pump model (Model I);
%   2) the scalar NLSE solved with symmetric SSFM (Model II).
%
% The shared transfer-matrix calculation then produces NF and maximum gain.
%
% Defaults preserve the active research-source settings:
%   N          = 20000
%   randomSeed = 14
%
% Two training-ready CSV files are written to dataDir. Each contains the
% six physical inputs plus NF_dB and max_gain_dB.
%
% Full coefficient-magnitude diagnostics, gain-comparison tables, and the
% comparison figure are written under outputDir.

    if nargin < 1 || isempty(N)
        N = 20000;
    end

    if nargin < 2 || isempty(randomSeed)
        randomSeed = 14;
    end

    projectRoot = fileparts(fileparts(mfilename('fullpath')));

    if nargin < 3 || isempty(dataDir)
        dataDir = fullfile(projectRoot, 'data');
    end

    if nargin < 4 || isempty(outputDir)
        outputDir = fullfile(projectRoot, 'outputs', 'inoue_nlse_comparison');
    end

    validateattributes(N, {'numeric'}, ...
        {'scalar','integer','positive','finite'}, mfilename, 'N');
    validateattributes(randomSeed, {'numeric'}, ...
        {'scalar','integer','nonnegative','finite'}, mfilename, 'randomSeed');

    if ~isfolder(dataDir)
        mkdir(dataDir);
    end

    if ~isfolder(outputDir)
        mkdir(outputDir);
    end

    rng(randomSeed, 'twister');
    startTime = tic;

    % Parameter endpoints preserved from the research source.
    sep_a_nm = 4;
    sep_b_nm = 16;
    L_a_m = 300;
    L_b_m = 300;
    gamma_a_km = 12;
    gamma_b_km = 12;

    % Keep the legacy endpoint order to preserve the seeded random mapping.
    beta2_a_ps2km = -0.1;
    beta2_b_ps2km = -10;

    lambda0_a_nm = 1500;
    lambda0_b_nm = 1600;
    Ppump_a_dBm = 19;
    Ppump_b_dBm = 25;

    inputData = zeros(N, 6);
    inputData(:, 1) = sample_between(sep_a_nm, sep_b_nm, N);
    inputData(:, 2) = sample_between(L_a_m, L_b_m, N);
    inputData(:, 3) = sample_between(gamma_a_km, gamma_b_km, N);
    inputData(:, 4) = sample_between(beta2_a_ps2km, beta2_b_ps2km, N);
    inputData(:, 5) = sample_between(lambda0_a_nm, lambda0_b_nm, N);
    inputData(:, 6) = sample_between(Ppump_a_dBm, Ppump_b_dBm, N);

    inputTable = array2table(inputData, ...
        'VariableNames', { ...
            'pump_signal_sep_nm', ...
            'fiber_length_m', ...
            'gamma_Winv_km', ...
            'beta2_ps2_per_km', ...
            'zero_disp_wavelength_nm', ...
            'pump_power_dBm'});

    % Columns: NF, Gmax, |C1|...|C6|.
    outputInoue = NaN(N, 8);
    outputNLSE = NaN(N, 8);
    comparison = NaN(N, 4);

    progressEvery = max(1, floor(N / 100));

    fprintf(['Generating %d paired Inoue/NLSE cases ' ...
        'with RNG seed %d.\n'], N, randomSeed);

    for n = 1:N
        if n == 1 || n == N || mod(n, progressEvery) == 0
            fprintf('Case %d / %d\n', n, N);
        end

        sep_nm = inputData(n, 1);
        fiberLen_m = inputData(n, 2);
        gamma_km = inputData(n, 3);
        beta2_ps2km = inputData(n, 4);
        lambda0_nm = inputData(n, 5);
        pumpP_dBm = inputData(n, 6);

        try
            [outInoue, outNLSE] = compute_case_both_methods( ...
                sep_nm, ...
                fiberLen_m, ...
                gamma_km, ...
                beta2_ps2km, ...
                lambda0_nm, ...
                pumpP_dBm);

            outputInoue(n, :) = outInoue;
            outputNLSE(n, :) = outNLSE;
            comparison(n, :) = [ ...
                pumpP_dBm, ...
                outInoue(2), ...
                outNLSE(2), ...
                outNLSE(2) - outInoue(2)];

        catch ME
            warning('Case %d failed: %s', n, ME.message);
        end
    end

    inoueTargets = array2table(outputInoue(:, 1:2), ...
        'VariableNames', {'NF_dB', 'max_gain_dB'});

    nlseTargets = array2table(outputNLSE(:, 1:2), ...
        'VariableNames', {'NF_dB', 'max_gain_dB'});

    inoueTrainingTable = [inputTable, inoueTargets];
    nlseTrainingTable = [inputTable, nlseTargets];

    inoueDatasetPath = fullfile( ...
        dataDir, ...
        sprintf('inoue_dataset_%d.csv', N));

    nlseDatasetPath = fullfile( ...
        dataDir, ...
        sprintf('inputdata_nlse_%d.csv', N));

    writetable(inoueTrainingTable, inoueDatasetPath);
    writetable(nlseTrainingTable, nlseDatasetPath);

    coefficientNames = { ...
        'NF_dB', ...
        'max_gain_dB', ...
        'abs_C1', ...
        'abs_C2', ...
        'abs_C3', ...
        'abs_C4', ...
        'abs_C5', ...
        'abs_C6'};

    writetable( ...
        array2table(outputInoue, 'VariableNames', coefficientNames), ...
        fullfile(outputDir, sprintf('inoue_full_outputs_%d.csv', N)));

    writetable( ...
        array2table(outputNLSE, 'VariableNames', coefficientNames), ...
        fullfile(outputDir, sprintf('nlse_full_outputs_%d.csv', N)));

    comparisonTable = array2table(comparison, ...
        'VariableNames', { ...
            'pump_power_dBm', ...
            'Gmax_Inoue_dB', ...
            'Gmax_NLSE_dB', ...
            'Delta_Gmax_NLSE_minus_Inoue_dB'});

    comparisonPath = fullfile( ...
        outputDir, ...
        sprintf('comparison_gain_difference_%d.csv', N));

    writetable(comparisonTable, comparisonPath);

    valid = all(isfinite(comparison), 2);

    fig = figure( ...
        'Visible', 'off', ...
        'Color', 'w', ...
        'Name', 'NLSE vs Inoue max-gain difference');

    scatter( ...
        comparison(valid, 1), ...
        comparison(valid, 4), ...
        35, ...
        'filled');

    grid on;
    box on;
    xlabel('Input pump power per pump (dBm)');
    ylabel('\Delta G_{max} = G_{max,NLSE} - G_{max,Inoue} (dB)');
    title('Difference in maximum PSA gain: NLSE pump model minus Inoue pump model');
    yline(0, '--');

    figurePath = fullfile( ...
        outputDir, ...
        sprintf('scatter_gain_difference_vs_pump_power_%d.png', N));

    exportgraphics(fig, figurePath, 'Resolution', 300);
    close(fig);

    numInoueFailed = sum(any(~isfinite(outputInoue(:, 1:2)), 2));
    numNLSEFailed = sum(any(~isfinite(outputNLSE(:, 1:2)), 2));

    fprintf('\nDone.\n');
    fprintf('Inoue training dataset: %s\n', inoueDatasetPath);
    fprintf('NLSE training dataset : %s\n', nlseDatasetPath);
    fprintf('Comparison table      : %s\n', comparisonPath);
    fprintf('Comparison figure     : %s\n', figurePath);
    fprintf('Inoue valid cases     : %d / %d\n', N - numInoueFailed, N);
    fprintf('NLSE valid cases      : %d / %d\n', N - numNLSEFailed, N);
    fprintf('Elapsed time          : %.2f s\n', toc(startTime));
end


function values = sample_between(endpointA, endpointB, N)
%SAMPLE_BETWEEN Uniform sampling that preserves endpoint order.
    if endpointA == endpointB
        values = repmat(endpointA, N, 1);
    else
        values = endpointA + (endpointB - endpointA) * rand(N, 1);
    end
end

%% ------------------------------------------------------------------------
function [outInoue, outNLSE] = compute_case_both_methods( ...
    sep_nm, fiberLen_m, gamma_km, beta2_ps2km, lambda0_nm, pumpP_dBm)

    c = 299792458;

    % Degenerate dual-pump PSA: signal is midway between pumps.
    lambda_s_nm = 1550;
    fs = c/(lambda_s_nm*1e-9);

    % Approximate signal-pump wavelength separation converted to frequency.
    df = c*(sep_nm*1e-9)/(lambda_s_nm*1e-9)^2;
    fp1 = fs - df;
    fp2 = fs + df;

    fF1 = 2*fp1 - fp2;
    fF2 = 2*fp2 - fp1;
    fi1 = 2*fp1 - fs;
    fi2 = 2*fp2 - fs;

    gamma = gamma_km/1000;                 % [1/(W m)]
    beta2 = beta2_ps2km*1e-27;             % [s^2/m]
    P0 = 1e-3*10^(pumpP_dBm/10);           % [W] per pump
    L = fiberLen_m;                        % [m]

    omega_s = 2*pi*fs;
    omega0  = 2*pi*c/(lambda0_nm*1e-9);
    if abs(omega0 - omega_s) < 1e9
        beta3 = 0;
    else
        beta3 = -beta2/(omega0 - omega_s); % [s^3/m]
    end

    % Common longitudinal sampling for the transfer matrix.
    nSteps = max(1000, ceil(L/0.25));
    dz = L/nSteps;
    zVec = (0:nSteps).' * dz;

    % ---- Method 1: Inoue coupled pump equations ----
    [Ap1_I, Ap2_I] = propagate_pumps_inoue(zVec, gamma, beta2, beta3, fs, fp1, fp2, fF1, fF2, P0);
    outInoue = transfer_matrix_nf_gain(Ap1_I, Ap2_I, zVec, gamma, beta2, beta3, fs, fp1, fp2, fi1, fi2);

    % ---- Method 2: NLSE / SSFM pump propagation ----
    [Ap1_N, Ap2_N] = propagate_pumps_nlse_ssfm(zVec, gamma, beta2, beta3, fs, fp1, fp2, fF1, fF2, P0);
    outNLSE = transfer_matrix_nf_gain(Ap1_N, Ap2_N, zVec, gamma, beta2, beta3, fs, fp1, fp2, fi1, fi2);
end

%% ------------------------------------------------------------------------
function [Ap1, Ap2] = propagate_pumps_inoue(zVec, gamma, beta2, beta3, fs, fp1, fp2, fF1, fF2, P0)
    % Fixed-step RK4 integration of Inoue Eq. (14)-type pump equations.
    % State Y = [Ap1; Ap2; AF1; AF2].

    nSteps = numel(zVec)-1;
    dz = zVec(2)-zVec(1);

    Y = [sqrt(P0); sqrt(P0); 0; 0];
    Ap1 = zeros(nSteps+1,1);
    Ap2 = zeros(nSteps+1,1);
    Ap1(1) = Y(1);
    Ap2(1) = Y(2);

    for k = 1:nSteps
        z = zVec(k);
        k1 = inoue_rhs(z,          Y,             gamma, beta2, beta3, fs, fp1, fp2, fF1, fF2);
        k2 = inoue_rhs(z + dz/2,   Y + dz*k1/2,   gamma, beta2, beta3, fs, fp1, fp2, fF1, fF2);
        k3 = inoue_rhs(z + dz/2,   Y + dz*k2/2,   gamma, beta2, beta3, fs, fp1, fp2, fF1, fF2);
        k4 = inoue_rhs(z + dz,     Y + dz*k3,     gamma, beta2, beta3, fs, fp1, fp2, fF1, fF2);
        Y = Y + dz*(k1 + 2*k2 + 2*k3 + k4)/6;

        Ap1(k+1) = Y(1);
        Ap2(k+1) = Y(2);
    end
end

%% ------------------------------------------------------------------------
function dY = inoue_rhs(z, Y, gamma, beta2, beta3, fs, fp1, fp2, fF1, fF2)
    Ap1 = Y(1); Ap2 = Y(2); AF1 = Y(3); AF2 = Y(4);
    P1 = abs(Ap1)^2; P2 = abs(Ap2)^2;

    dBF10 = 2*beta_of_f(fp1,fs,beta2,beta3) - beta_of_f(fp2,fs,beta2,beta3) - beta_of_f(fF1,fs,beta2,beta3);
    dBF20 = 2*beta_of_f(fp2,fs,beta2,beta3) - beta_of_f(fp1,fs,beta2,beta3) - beta_of_f(fF2,fs,beta2,beta3);

    % Inoue nonlinear mismatch terms for pump FWM processes (a,b).
    dBF1 = dBF10 - gamma*(2*P1 - P2);
    dBF2 = dBF20 - gamma*(2*P2 - P1);

    dAp1 = 2i*gamma*AF1*Ap2*conj(Ap1)*exp(-1i*dBF1*z) + ...
           1i*gamma*Ap2^2*conj(AF2)*exp( 1i*dBF2*z);
    dAp2 = 2i*gamma*AF2*Ap1*conj(Ap2)*exp(-1i*dBF2*z) + ...
           1i*gamma*Ap1^2*conj(AF1)*exp( 1i*dBF1*z);
    dAF1 = 1i*gamma*Ap1^2*conj(Ap2)*exp( 1i*dBF1*z);
    dAF2 = 1i*gamma*Ap2^2*conj(Ap1)*exp( 1i*dBF2*z);

    dY = [dAp1; dAp2; dAF1; dAF2];
end

%% ------------------------------------------------------------------------
function [Ap1, Ap2] = propagate_pumps_nlse_ssfm(zVec, gamma, beta2, beta3, fs, fp1, fp2, fF1, fF2, P0)
    nSteps = numel(zVec)-1;
    dz = zVec(2)-zVec(1);

    maxOffset = max(abs([fF1, fp1, fp2, fF2] - fs));
    Fspan = 2.5*maxOffset;                 % half-span in Hz
    Nt = 2^13;                             % increase if spectral resolution is insufficient
    dt = 1/(2*Fspan);
    T = Nt*dt;
    t = (-Nt/2:Nt/2-1).' * dt;
    f_rel = (-Nt/2:Nt/2-1).' / T;
    omega_rel = 2*pi*f_rel;

    A = sqrt(P0)*exp(1i*2*pi*(fp1-fs)*t) + sqrt(P0)*exp(1i*2*pi*(fp2-fs)*t);

    beta_w = 0.5*beta2*omega_rel.^2 + (1/6)*beta3*omega_rel.^3;
    halfLinear = exp(1i*beta_w*dz/2);

    Ap1 = zeros(nSteps+1,1);
    Ap2 = zeros(nSteps+1,1);
    Ap1(1) = spectral_amplitude(A, t, fp1-fs);
    Ap2(1) = spectral_amplitude(A, t, fp2-fs);

    for k = 1:nSteps
        Aw = fftshift(fft(ifftshift(A)));
        Aw = Aw .* halfLinear;
        A = fftshift(ifft(ifftshift(Aw)));

        A = A .* exp(1i*gamma*abs(A).^2*dz);

        Aw = fftshift(fft(ifftshift(A)));
        Aw = Aw .* halfLinear;
        A = fftshift(ifft(ifftshift(Aw)));

        Ap1(k+1) = spectral_amplitude(A, t, fp1-fs);
        Ap2(k+1) = spectral_amplitude(A, t, fp2-fs);
    end
end

%% ------------------------------------------------------------------------
function out = transfer_matrix_nf_gain(Ap1, Ap2, zVec, gamma, beta2, beta3, fs, fp1, fp2, fi1, fi2)
    nSteps = numel(zVec)-1;
    dz = zVec(2)-zVec(1);

    M = eye(6);
    for k = 1:nSteps
        z = zVec(k);
        a1 = Ap1(k);
        a2 = Ap2(k);
        P1 = abs(a1)^2;
        P2 = abs(a2)^2;

        dBs  = beta_of_f(fp1,fs,beta2,beta3) + beta_of_f(fp2,fs,beta2,beta3) ...
             - 2*beta_of_f(fs,fs,beta2,beta3) - gamma*(P1+P2);
        dBi1 = 2*beta_of_f(fp1,fs,beta2,beta3) - beta_of_f(fs,fs,beta2,beta3) ...
             - beta_of_f(fi1,fs,beta2,beta3) - 2*gamma*P1;
        dBi2 = 2*beta_of_f(fp2,fs,beta2,beta3) - beta_of_f(fs,fs,beta2,beta3) ...
             - beta_of_f(fi2,fs,beta2,beta3) - 2*gamma*P2;

        cs  = 2i*gamma*a1*a2*exp(-2i*dBs*z)*dz;
        ci1 = 1i*gamma*a1^2*exp(-1i*dBi1*z)*dz;
        ci2 = 1i*gamma*a2^2*exp(-1i*dBi2*z)*dz;

        Mk = [ 1,      cs,     0,      ci1,   0,      ci2; ...
               conj(cs),1,     conj(ci1),0,   conj(ci2),0; ...
               0,      ci1,    1,      0,     0,      0; ...
               conj(ci1),0,    0,      1,     0,      0; ...
               0,      ci2,    0,      0,     1,      0; ...
               conj(ci2),0,    0,      0,     0,      1 ];

        M = Mk*M;
    end

    C = M(1,:);
    C1c = C(1); C2c = C(2); C3c = C(3); C4c = C(4); C5c = C(5); C6c = C(6);

    G = (abs(C1c) + abs(C2c))^2;
    NF = 1 + (abs(C3c)^2 + abs(C4c)^2 + abs(C5c)^2 + abs(C6c)^2)/G;

    NF_dB = 10*log10(real(NF));
    Gmax_dB = 10*log10(real(G));

    out = [NF_dB, Gmax_dB, abs(C1c), abs(C2c), abs(C3c), abs(C4c), abs(C5c), abs(C6c)];
end

%% ------------------------------------------------------------------------
function amp = spectral_amplitude(A, t, f0)
    % Projection of the time-domain envelope onto exp(i*2*pi*f0*t).
    % For a pure tone sqrt(P)*exp(i*2*pi*f0*t), this returns sqrt(P).
    amp = mean(A .* exp(-1i*2*pi*f0*t));
end

%% ------------------------------------------------------------------------
function b = beta_of_f(f, f_ref, beta2, beta3)
    % Linear propagation constant relative to beta(f_ref), ignoring beta0/beta1
    % because these cancel in phase-mismatch combinations.
    dw = 2*pi*(f - f_ref);
    b = 0.5*beta2*dw.^2 + (1/6)*beta3*dw.^3;
end
