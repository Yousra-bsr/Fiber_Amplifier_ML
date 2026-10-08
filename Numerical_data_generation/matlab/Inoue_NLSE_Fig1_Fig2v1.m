function Inoue_NLSE_Fig1_Fig2v1(outputDir)
%INOUE_NLSE_FIG1_FIG2V1 Reproduce pump/HOP and NLSE-spectrum comparisons.
%
% This routine compares the four-wave Inoue pump model with scalar-NLSE
% propagation for the selected settings and writes the generated figures and
% supporting XY data under outputDir.
%
% Important: this file does NOT compute signal gain, noise figure, or C1...C6.
% It is a numerical/figure-comparison utility rather than an ML target generator.

    projectRoot = fileparts(fileparts(mfilename('fullpath')));

    if nargin < 1 || isempty(outputDir)
        outputDir = fullfile(projectRoot, 'outputs', 'figure_comparison');
    end

    if ~isfolder(outputDir)
        mkdir(outputDir);
    end

    rng(14, 'twister');

    % User settings preserved from the supplied source.
    N = 1;
    plotCaseN = 1;
    dz_requested_m = 0.25;
    Nt = 2^13;
    spectralFloor_dBm = -120;
    maxCombOrder = 51;

    sep_min_nm = 4;
    sep_max_nm = 4;
    L_min_m = 300;
    L_max_m = 300;
    gamma_min_km = 12;
    gamma_max_km = 12;
    beta2_min_ps2km = -0.2;
    beta2_max_ps2km = -0.2;
    lambda0_min_nm = 1520;
    lambda0_max_nm = 1520;
    pumpPowers_dBm = [19, 25];

    assert( ...
        plotCaseN >= 1 ...
        && plotCaseN <= N ...
        && plotCaseN == round(plotCaseN), ...
        'plotCaseN must be an integer from 1 to N.');

    inputData = zeros(N, 5);
    inputData(:, 1) = uniform_range(sep_min_nm, sep_max_nm, N);
    inputData(:, 2) = uniform_range(L_min_m, L_max_m, N);
    inputData(:, 3) = uniform_range(gamma_min_km, gamma_max_km, N);
    inputData(:, 4) = uniform_range(beta2_min_ps2km, beta2_max_ps2km, N);
    inputData(:, 5) = uniform_range(lambda0_min_nm, lambda0_max_nm, N);

    inputTable = array2table(inputData, ...
        'VariableNames', { ...
            'pump_signal_sep_nm', ...
            'fiber_length_m', ...
            'gamma_Winv_km', ...
            'beta2_ps2_per_km', ...
            'zero_disp_wavelength_nm'});

    writetable( ...
        inputTable, ...
        fullfile(outputDir, sprintf('simulation_inputs_N%d.csv', N)));

    selected = cell(1, numel(pumpPowers_dBm));

    for ip = 1:numel(pumpPowers_dBm)
        pump_dBm = pumpPowers_dBm(ip);
        fprintf('\nInput power = %.1f dBm per pump\n', pump_dBm);

        for n = 1:N
            fprintf('  simulation %d/%d\n', n, N);

            result = simulate_one_case( ...
                inputData(n, :), ...
                pump_dBm, ...
                dz_requested_m, ...
                Nt, ...
                spectralFloor_dBm, ...
                maxCombOrder);

            if n == plotCaseN
                selected{ip} = result;
            end
        end
    end

    blue = [0 0.4470 0.7410];
    red = [0.8500 0.3250 0.0980];

    % Figure 1: pump 1 and HOP 1 versus propagation distance.
    fig1 = figure( ...
        'Visible', 'off', ...
        'Color', 'w', ...
        'Name', 'Fig. 1: Pump and HOP evolution');

    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    for ip = 1:2
        r = selected{ip};

        nexttile;
        hold on;

        plot(r.z_m, r.Pp1_I_dBm, '-', ...
            'Color', blue, 'LineWidth', 3.8);
        plot(r.z_m, r.PF1_I_dBm, ':', ...
            'Color', blue, 'LineWidth', 3.8);
        plot(r.z_m, r.Pp1_N_dBm, '-', ...
            'Color', red, 'LineWidth', 2.2);
        plot(r.z_m, r.PF1_N_dBm, ':', ...
            'Color', red, 'LineWidth', 2.2);

        grid off;
        box on;
        xlabel('Fiber position, z (m)');
        ylabel('Power (dBm)');
        ylim([-20 30]);

        title(sprintf( ...
            '(%c) P = %g dBm', ...
            char('a' + ip - 1), ...
            pumpPowers_dBm(ip)));

        legend( ...
            'Pump 1: model I', ...
            'HOP 1: model I', ...
            'Pump 1: model II', ...
            'HOP 1: model II', ...
            'Location', 'best');

        T = table( ...
            r.z_m, ...
            r.Pp1_I_dBm, ...
            r.PF1_I_dBm, ...
            r.Pp1_N_dBm, ...
            r.PF1_N_dBm, ...
            'VariableNames', { ...
                'z_m', ...
                'pump1_multiwave_dBm', ...
                'HOP1_multiwave_dBm', ...
                'pump1_NLSE_dBm', ...
                'HOP1_NLSE_dBm'});

        writetable( ...
            T, ...
            fullfile( ...
                outputDir, ...
                sprintf( ...
                    'Fig1%c_%ddBm_xy_n%03d.csv', ...
                    char('a' + ip - 1), ...
                    pumpPowers_dBm(ip), ...
                    plotCaseN)));
    end

    exportgraphics( ...
        fig1, ...
        fullfile( ...
            outputDir, ...
            sprintf('Fig1_pump_HOP_evolution_n%03d.png', plotCaseN)), ...
        'Resolution', 300);

    savefig( ...
        fig1, ...
        fullfile( ...
            outputDir, ...
            sprintf('Fig1_pump_HOP_evolution_n%03d.fig', plotCaseN)));

    close(fig1);

    % Figure 2: NLSE output spectrum.
    fig2 = figure( ...
        'Visible', 'off', ...
        'Color', 'w', ...
        'Name', 'Fig. 2: NLSE output spectra');

    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    for ip = 1:2
        r = selected{ip};

        nexttile;

        stem( ...
            r.lambda_nm, ...
            r.spectrum_dBm, ...
            'Color', red, ...
            'LineWidth', 1.2, ...
            'Marker', 'o', ...
            'MarkerSize', 3);

        grid off;
        box on;
        xlabel('Wavelength (nm)');
        ylabel('Comb-line power (dBm)');

        title(sprintf( ...
            '(%c) P_{in} = %g dBm, z = %g m', ...
            char('a' + ip - 1), ...
            pumpPowers_dBm(ip), ...
            r.z_m(end)));

        ylim([-20 30]);

        sepForPlot_nm = inputData(plotCaseN, 1);
        xlim([1550 - 6 * sepForPlot_nm, 1550 + 6 * sepForPlot_nm]);

        T = table( ...
            r.combOrder, ...
            r.lambda_nm, ...
            r.frequency_THz, ...
            r.frequencyOffset_THz, ...
            r.spectrum_dBm, ...
            'VariableNames', { ...
                'comb_order_m', ...
                'wavelength_nm', ...
                'frequency_THz', ...
                'frequency_offset_from_signal_THz', ...
                'comb_line_power_dBm'});

        writetable( ...
            T, ...
            fullfile( ...
                outputDir, ...
                sprintf( ...
                    'Fig2%c_%ddBm_spectrum_xy_n%03d.csv', ...
                    char('a' + ip - 1), ...
                    pumpPowers_dBm(ip), ...
                    plotCaseN)));
    end

    exportgraphics( ...
        fig2, ...
        fullfile( ...
            outputDir, ...
            sprintf('Fig2_NLSE_output_spectra_n%03d.png', plotCaseN)), ...
        'Resolution', 300);

    savefig( ...
        fig2, ...
        fullfile( ...
            outputDir, ...
            sprintf('Fig2_NLSE_output_spectra_n%03d.fig', plotCaseN)));

    close(fig2);

    save( ...
        fullfile(outputDir, 'selected_run_NLSE_spectra.mat'), ...
        'selected', ...
        'inputData', ...
        'pumpPowers_dBm', ...
        'plotCaseN', ...
        'N', ...
        'dz_requested_m', ...
        'Nt', ...
        'maxCombOrder');

    fprintf('\nFinished. Figure outputs saved in:\n  %s\n', outputDir);
end

%% ========================================================================
function x = uniform_range(xmin,xmax,N)
    if xmin == xmax
        x = repmat(xmin,N,1);
    else
        x = xmin + (xmax-xmin)*rand(N,1);
    end
end

function out = simulate_one_case(par,pumpP_dBm,dzRequested,Nt,floor_dBm,maxOrder)
    c = 299792458;
    sep_nm = par(1); L = par(2); gamma_km = par(3);
    beta2_ps2km = par(4); lambda0_nm = par(5);

    lambda_s_nm = 1550;
    fs = c/(lambda_s_nm*1e-9);
    df = c*(sep_nm*1e-9)/(lambda_s_nm*1e-9)^2;
    fp1 = fs-df; fp2 = fs+df;
    fF1 = 2*fp1-fp2; fF2 = 2*fp2-fp1;

    gamma = gamma_km/1000;             % W^-1 m^-1
    beta2 = beta2_ps2km*1e-27;         % s^2 m^-1
    P0 = 1e-3*10^(pumpP_dBm/10);       % W per pump
    omega0 = 2*pi*c/(lambda0_nm*1e-9);
    omegas = 2*pi*fs;
    if abs(omega0 - omegas) < 1e9
        beta3 = 0;
    else
        beta3 = -beta2/(omega0-omegas);    % s^3 m^-1
    end

    nSteps = ceil(L/dzRequested);
    dz = L/nSteps;                     % exactly reaches L
    z = (0:nSteps).'*dz;

    [Ap1I,~,AF1I,~] = inoue_propagation(z,gamma,beta2,beta3, ...
        fs,fp1,fp2,fF1,fF2,P0);
    [Ap1N,~,AF1N,~,frel,Aend] = nlse_propagation(z,gamma,beta2,beta3, ...
        df,P0,Nt);

    out.z_m = z;
    out.Pp1_I_dBm = watts_to_dBm(abs(Ap1I).^2,floor_dBm);
    out.PF1_I_dBm = watts_to_dBm(abs(AF1I).^2,floor_dBm);
    out.Pp1_N_dBm = watts_to_dBm(abs(Ap1N).^2,floor_dBm);
    out.PF1_N_dBm = watts_to_dBm(abs(AF1N).^2,floor_dBm);

    % With T=1/df, all ideal FWM tones lie exactly on FFT bins. Extract
    % selected integer-order lines rather than plotting empty inter-line bins.
    Aw = fftshift(fft(ifftshift(Aend)))/Nt;
    Pbin = abs(Aw).^2;
    orders = (-maxOrder:maxOrder).';
    targetOffsets = orders*df;
    binIndex = round((targetOffsets-frel(1))/(frel(2)-frel(1)))+1;
    assert(all(binIndex>=1 & binIndex<=Nt),'Increase Nt or reduce maxCombOrder.');
    fvalid = fs+frel(binIndex);
    lambda = c./fvalid*1e9;
    spec = watts_to_dBm(Pbin(binIndex),floor_dBm);
    frelvalid = frel(binIndex);
    [lambda,order] = sort(lambda,'ascend');
    out.lambda_nm = lambda;
    out.frequency_THz = fvalid(order)/1e12;
    out.frequencyOffset_THz = frelvalid(order)/1e12;
    out.spectrum_dBm = spec(order);
    out.combOrder = orders(order);
end

function [Ap1,Ap2,AF1,AF2] = inoue_propagation( ...
    z,gamma,beta2,beta3,fs,fp1,fp2,fF1,fF2,P0)
    nSteps = numel(z)-1; dz = z(2)-z(1);
    Y = complex([sqrt(P0);sqrt(P0);0;0]);
    history = complex(zeros(nSteps+1,4)); history(1,:) = Y.';
    for k = 1:nSteps
        zk = z(k);
        k1 = inoue_rhs(zk,Y,gamma,beta2,beta3,fs,fp1,fp2,fF1,fF2);
        k2 = inoue_rhs(zk+dz/2,Y+dz*k1/2,gamma,beta2,beta3,fs,fp1,fp2,fF1,fF2);
        k3 = inoue_rhs(zk+dz/2,Y+dz*k2/2,gamma,beta2,beta3,fs,fp1,fp2,fF1,fF2);
        k4 = inoue_rhs(zk+dz,Y+dz*k3,gamma,beta2,beta3,fs,fp1,fp2,fF1,fF2);
        Y = Y+dz*(k1+2*k2+2*k3+k4)/6;
        history(k+1,:) = Y.';
    end
    Ap1=history(:,1); Ap2=history(:,2); AF1=history(:,3); AF2=history(:,4);
end

function dY = inoue_rhs(z,Y,gamma,beta2,beta3,fs,fp1,fp2,fF1,fF2)
    Ap1=Y(1); Ap2=Y(2); AF1=Y(3); AF2=Y(4);
    P1=abs(Ap1)^2; P2=abs(Ap2)^2;
    dB10 = 2*beta_of_f(fp1,fs,beta2,beta3)-beta_of_f(fp2,fs,beta2,beta3)-beta_of_f(fF1,fs,beta2,beta3);
    dB20 = 2*beta_of_f(fp2,fs,beta2,beta3)-beta_of_f(fp1,fs,beta2,beta3)-beta_of_f(fF2,fs,beta2,beta3);
    dB1=dB10-gamma*(2*P1-P2); dB2=dB20-gamma*(2*P2-P1);
    dY = [2i*gamma*AF1*Ap2*conj(Ap1)*exp(-1i*dB1*z)+1i*gamma*Ap2^2*conj(AF2)*exp(1i*dB2*z); ...
          2i*gamma*AF2*Ap1*conj(Ap2)*exp(-1i*dB2*z)+1i*gamma*Ap1^2*conj(AF1)*exp(1i*dB1*z); ...
          1i*gamma*Ap1^2*conj(Ap2)*exp(1i*dB1*z); ...
          1i*gamma*Ap2^2*conj(Ap1)*exp(1i*dB2*z)];
end

function [Ap1,Ap2,AF1,AF2,frel,A] = nlse_propagation( ...
    z,gamma,beta2,beta3,df,P0,Nt)
    nSteps=numel(z)-1; dz=z(2)-z(1);
    T=1/df; dt=T/Nt;
    t=(-Nt/2:Nt/2-1).'*dt;
    frel=(-Nt/2:Nt/2-1).'/T;
    omega=2*pi*frel;
    A=sqrt(P0)*exp(-1i*2*pi*df*t)+sqrt(P0)*exp(1i*2*pi*df*t);
    halfLinear=exp(1i*(0.5*beta2*omega.^2+beta3*omega.^3/6)*dz/2);
    Ap1=complex(zeros(nSteps+1,1)); Ap2=Ap1; AF1=Ap1; AF2=Ap1;
    [Ap1(1),Ap2(1),AF1(1),AF2(1)] = four_tones(A,t,df);
    for k=1:nSteps
        Aw=fftshift(fft(ifftshift(A))); Aw=Aw.*halfLinear;
        A=fftshift(ifft(ifftshift(Aw)));
        A=A.*exp(1i*gamma*abs(A).^2*dz);
        Aw=fftshift(fft(ifftshift(A))); Aw=Aw.*halfLinear;
        A=fftshift(ifft(ifftshift(Aw)));
        [Ap1(k+1),Ap2(k+1),AF1(k+1),AF2(k+1)] = four_tones(A,t,df);
    end
end

function [p1,p2,F1,F2] = four_tones(A,t,df)
    p1=mean(A.*exp( 1i*2*pi*df*t));
    p2=mean(A.*exp(-1i*2*pi*df*t));
    F1=mean(A.*exp( 1i*2*pi*3*df*t));
    F2=mean(A.*exp(-1i*2*pi*3*df*t));
end

function b = beta_of_f(f,fref,beta2,beta3)
    dw=2*pi*(f-fref);
    b=0.5*beta2*dw.^2+beta3*dw.^3/6;
end

function y = watts_to_dBm(P,floor_dBm)
    floor_W=1e-3*10^(floor_dBm/10);
    y=10*log10(max(real(P),floor_W)/1e-3);
end
