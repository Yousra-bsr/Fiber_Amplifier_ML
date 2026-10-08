function datasetPath = generatemldataset( ...
    N, randomSeed, dataDir, outputDir)
%GENERATEMLDATASET Generate a model I ML dataset.
%
% datasetPath = generatemldataset()
% datasetPath = generatemldataset(N, randomSeed)
% datasetPath = generatemldataset(N, randomSeed, dataDir, outputDir)
%
% Defaults reproduce the currently active source settings:
%   N          = 100
%   randomSeed = 1
%
% The primary CSV is training-ready: it contains the six sampled physical
% parameters plus NF_dB and max_gain_dB in one table.
%
% Complex C1...C6 coefficients are preserved separately in a diagnostic MAT
% file under outputDir rather than mixed into the ML CSV.

    if nargin < 1 || isempty(N)
        N = 100;
    end

    if nargin < 2 || isempty(randomSeed)
        randomSeed = 1;
    end

    projectRoot = fileparts(fileparts(mfilename('fullpath')));

    if nargin < 3 || isempty(dataDir)
        dataDir = fullfile(projectRoot, 'data');
    end

    if nargin < 4 || isempty(outputDir)
        outputDir = fullfile(projectRoot, 'outputs', 'inoue_generation');
    end

    validateattributes(N, {'numeric'}, ...
        {'scalar','integer','positive','finite'}, mfilename, 'N');
    validateattributes(randomSeed, {'numeric'}, ...
        {'scalar','integer','nonnegative','finite'}, mfilename, 'randomSeed');

    if ~isfolder(dataDir)
        mkdir(dataDir);
    end

    diagnosticsDir = fullfile(outputDir, 'diagnostics');
    if ~isfolder(diagnosticsDir)
        mkdir(diagnosticsDir);
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

    % The source listed -0.1 first and -10 second. Keeping endpoint order
    % preserves the exact random mapping for a given RNG seed.
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

    targetData = NaN(N, 2);
    coefficients = complex(NaN(N, 6));

    progressEvery = max(1, floor(N / 100));

    fprintf('Generating %d Inoue-model cases with RNG seed %d.\n', ...
        N, randomSeed);

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
            [NF_dB, Gmax_dB, C1, C2, C3, C4, C5, C6] = ...
                compute_nf_and_maxgain( ...
                    sep_nm, ...
                    fiberLen_m, ...
                    gamma_km, ...
                    beta2_ps2km, ...
                    lambda0_nm, ...
                    pumpP_dBm);

            targetData(n, :) = [NF_dB, Gmax_dB];
            coefficients(n, :) = [C1, C2, C3, C4, C5, C6];

        catch ME
            warning('Case %d failed: %s', n, ME.message);
        end
    end

    targetTable = array2table(targetData, ...
        'VariableNames', {'NF_dB', 'max_gain_dB'});

    trainingTable = [inputTable, targetTable];

    datasetName = sprintf('datatest_pump_beta2_ZDW_%d.csv', N);
    datasetPath = fullfile(dataDir, datasetName);
    writetable(trainingTable, datasetPath);

    diagnosticPath = fullfile( ...
        diagnosticsDir, ...
        sprintf('inoue_coefficients_N%d_seed%d.mat', N, randomSeed));

    save(diagnosticPath, ...
        'coefficients', ...
        'inputData', ...
        'targetData', ...
        'randomSeed', ...
        'N');

    numFailed = sum(any(~isfinite(targetData), 2));

    fprintf('\nDone.\n');
    fprintf('Training-ready dataset: %s\n', datasetPath);
    fprintf('Coefficient diagnostics: %s\n', diagnosticPath);
    fprintf('Successful cases: %d / %d\n', N - numFailed, N);
    fprintf('Elapsed time: %.2f s\n', toc(startTime));
end


function values = sample_between(endpointA, endpointB, N)
%SAMPLE_BETWEEN Uniform sampling that preserves endpoint order.
    if endpointA == endpointB
        values = repmat(endpointA, N, 1);
    else
        values = endpointA + (endpointB - endpointA) * rand(N, 1);
    end
end
