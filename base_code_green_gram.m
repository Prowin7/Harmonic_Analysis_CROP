clc;
clear;

% Load table from Excel
data = readtable('C:\Users\Praveen\Desktop\7th_Sem\Project\Green gram_2023 (1).xlsx');
vh = data.VH_dB;

% Define group ranges
g = [1 29; 30 48; 49 75; 76 98; 99 105;106 116];

% Compute vh_avg using mean
vh_avg = [
    mean(vh(1:29));
    mean(vh(30:48));
    mean(vh(49:75));
    mean(vh(76:98));
    mean(vh(99:105));
    mean(vh(106:116));
];

% Define x for fitting
x_group = [0 11 23 35 47 59 ]';

% Plot raw group averages
figure;
plot(x_group, vh_avg, '--sr', 'MarkerFaceColor','yellow');
xlabel('Custom Group Index');
ylabel('Average VH\_dB');
title('VH\_dB Averages for Custom Groups');
grid on;

% Fit using smoothing spline (you can also try 'poly3', etc.)
[fitresult, gof, interpolatedX, interpolatedY] = fit2(x_group, vh_avg);


% Evaluate fitted curve
interpolatedX = linspace(0, 59, 60);
interpolatedY = feval(fitresult, interpolatedX);

% Ensure column vectors
x = interpolatedX(:);
y = interpolatedY(:);

% Signal parameters
L = max(x);              % Period = 95
T = L;                   % Period
w0 = 2 * pi / T;         % Fundamental angular frequency

% Initialize Fourier coefficients
a0 = (2/T) * trapz(x, y);  % DC component
an = zeros(1, 6);
bn = zeros(1, 6);

% Compute Fourier coefficients
for n = 1:6
    an(n) = (2/T) * trapz(x, y .* cos(n * w0 * x));
   =
    bn(n) = (2/T) * trapz(x, y .* sin(n * w0 * x));
end

% Reconstruct the signal using Fourier series up to 6 harmonics
y_fs = a0 / 2 * ones(size(x));
for n = 1:6
    y_fs = y_fs + an(n) * cos(n * w0 * x) + bn(n) * sin(n * w0 * x);
end

% Plot original interpolated and Fourier reconstructed signal
figure;
plot(x, y, 'b', 'LineWidth', 1.5); hold on;
plot(x, y_fs, 'r--', 'LineWidth', 1.5);
legend('Interpolated VH\_dB', 'Fourier Series (6 Harmonics)');
xlabel('x');
ylabel('VH\_dB');
title('Fourier Series Approximation (Up to 6 Harmonics)');
grid on;
% Plot each harmonic component (excluding DC)
figure;
for n = 1:6
    harmonic = an(n) * cos(n * w0 * x) + bn(n) * sin(n * w0 * x);
    amplitude = sqrt(an(n)^2 + bn(n)^2);
    phase = atan2(-bn(n), an(n)); % Negative for correct sin convention

    subplot(5, 2, n);
    plot(x, harmonic, 'LineWidth', 1.2);
    grid on;
    title(sprintf('Harmonic %d | Amp = %.2f | Phase = %.2f rad', n, amplitude, phase));
    xlabel('x');
    ylabel(sprintf('H_%d(x)', n));
end

sgtitle('Individual Harmonics of VH\_dB (Excluding DC)');
% --- Plot bar chart of Fourier coefficients' amplitudes with cumulative graph ---

% Compute amplitudes of each harmonic
amplitudes = sqrt(an.^2 + bn.^2);

% Compute cumulative sum of amplitudes
cumulative_amplitudes = cumsum(amplitudes);

% Plot bar chart
figure;
yyaxis left;
bar(1:6, amplitudes, 'FaceColor', [0.2 0.6 0.5]);
ylabel('Amplitude of Harmonics');
xlabel('Harmonic Number');
title('Fourier Coefficient Amplitudes and Cumulative Sum');
grid on;

% Annotate each bar with amplitude value
for k = 1:6
    text(k, amplitudes(k) + 0.05, sprintf('%.2f', amplitudes(k)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontSize', 10, 'Color', 'black');
end

% Plot cumulative sum on top
yyaxis right;
plot(1:6, cumulative_amplitudes, '-o', 'Color', 'r', 'MarkerFaceColor', 'r', 'LineWidth', 1.5);
ylabel('Cumulative Amplitude');

% Annotate cumulative values above points
for k = 1:6
    text(k, cumulative_amplitudes(k) + 0.1, sprintf('%.2f', cumulative_amplitudes(k)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontSize', 9, 'Color', 'r');
end

legend('Amplitude', 'Cumulative Sum', 'Location', 'northwest');

% --- Compute phase of each harmonic (in radians) ---
phases = atan2(-bn, an);  % Negative sign for correct sin convention

% --- Compute circular phase variance ---
% Circular variance formula: 1 - R, where R = resultant vector length
R = sqrt((mean(cos(phases)))^2 + (mean(sin(phases)))^2);
phase_variance = 1 - R;

% Display phase variance in command window
fprintf('Circular Phase Variance: %.4f\n', phase_variance);
% --- Plot Phase Spectrum ---
figure;
stem(1:6, phases, 'filled', 'LineWidth', 1.5);
grid on;
xlabel('Harmonic Number');
ylabel('Phase (radians)');
title('Phase Spectrum of VH\_dB (Harmonics 1-6)');
yticks(-pi:pi/2:pi);
yticklabels({'-\pi','-\pi/2','0','\pi/2','\pi'});
% Prepare data for table
harmonic_num = (1:6)';
amplitude = sqrt(an.^2 + bn.^2)';
phase_rad = atan2(-bn, an)';  % Phase in radians
phase_deg = rad2deg(phase_rad);  % Also in degrees if needed

% Create table
fourier_table = table(harmonic_num, amplitude, phase_rad, phase_deg, ...
    'VariableNames', {'Harmonic', 'Amplitude', 'Phase_Rad', 'Phase_Deg'});

% Display table in command window
disp(fourier_table);

% Export to Excel
writetable(fourier_table, 'Fourier_Harmonics_Info.xlsx');


