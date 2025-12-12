clc;
clear;

% Load table from Excel
data = readtable('C:\Users\Praveen\Desktop\7th_Sem\Project\Green gram_2023 (1).xlsx');
vh = data.VH_dB;

% Compute average per group
vh_avg = [
    mean(vh(1:29));
    mean(vh(30:48));
    mean(vh(49:75));
    mean(vh(76:98));
    mean(vh(99:105));
    mean(vh(106:116));
];

% x for fitting
x_group = [0 11 23 35 47 59]';
[fitresult, ~, ~, ~] = fit2(x_group, vh_avg);

% Evaluate fitted curve
interpolatedX = linspace(0, 59, 60);
interpolatedY = feval(fitresult, interpolatedX);

x = interpolatedX(:);
y = interpolatedY(:);

% Signal parameters
L = max(x);
T = L;
w0 = 2*pi/T;

% Number of harmonics
N_harm = 16;

% Compute Fourier coefficients
a0 = (2/T) * trapz(x, y);
an = zeros(1, N_harm);
bn = zeros(1, N_harm);

for n = 1:N_harm
    an(n) = (2/T) * trapz(x, y .* cos(n*w0*x));
    disp(an(n));
    bn(n) = (2/T) * trapz(x, y .* sin(n*w0*x));
end

% Reconstruct signal
y_recon = a0/2 * ones(size(x));
for n = 1:N_harm
    y_recon = y_recon + an(n)*cos(n*w0*x) + bn(n)*sin(n*w0*x);
end

% Residuals and standard error
residuals = y - y_recon;
N = length(x);
SE = sqrt(2/N) * std(residuals);

% Compute p-values
p_an = 2 * (1 - tcdf(abs(an/SE), N-1));
p_bn = 2 * (1 - tcdf(abs(bn/SE), N-1));

% Significance threshold
alpha = 0.05;

% Display results
disp('Harmonic | Cosine Significant? | Sine Significant? | Cosine p-value | Sine p-value');
for n = 1:N_harm
    cos_sig = p_an(n) < alpha;
    sin_sig = p_bn(n) < alpha;
    fprintf('%3d      | %5s             | %5s           | %.4f        | %.4f\n', ...
        n, string(cos_sig), string(sin_sig), p_an(n), p_bn(n));
end
