clear; close all; clc; rng(123);
%%

%%%%%% EXERCISE ONE %%%%%%

%AR1: y_t = \phi y_{t-1} + \epsilon_t
T = 500;
phi = 0.4;
sigma2 = 0.2;
epsilon = sqrt(sigma2)*randn(T,1); %\sigmaN(0,1) = N(0,\sigma^2)

%for loop
y=zeros(T,1);
y(1)=epsilon(1);
for t = 2:T
    y(t) = phi*y(t-1) + epsilon(t);
end
plot(y)
title('AR(1), 500 obs, for loop')
yline(0, 'r', 'LineWidth', 1.5)
% --- Save each figure automatically as PDF ---
    filename = sprintf('Ex1a.pdf');
    exportgraphics(gcf, filename, 'ContentType', 'vector', 'BackgroundColor', 'white');


%function filter
%AR(1): (1-\phiL)y_t = \epsilon_t
y_filter = filter(1,[1 -phi],epsilon);
plot(y_filter)
title('AR(1), filter function')
yline(0, 'r', 'LineWidth', 1.5)
% --- Save each figure automatically as PDF ---
    filename = sprintf('Ex1b.pdf');
    exportgraphics(gcf, filename, 'ContentType', 'vector', 'BackgroundColor', 'white');

%confront
diff_val = max(abs(y - y_filter));
fprintf('max difference: %.2e\n', diff_val);

plot(y,'b'); hold on;
plot(y_filter, '--r');
legend('for loop', 'filter');
title('for loop vs filter (AR(1))');
hold off;
% --- Save each figure automatically as PDF ---
    filename = sprintf('Ex1c.pdf');
    exportgraphics(gcf, filename, 'ContentType', 'vector', 'BackgroundColor', 'white');

%%

%%%%%% EXERCISE TWO %%%%%%

%AR1: y_t = c+\phi y_{t-1} + \epsilon_t
% E(y_t) = \frac{c}{1-\phi} = 3
c = 1.2;

phi_b = 0.6;
sigma2_b = 0.4;
epsilon_b = sqrt(sigma2_b)*randn(T,1); %\sigmaN(0,1) = N(0,\sigma^2)

%for loop
y_b=zeros(T,1);
y_b(1)=20;
for t = 2:T
    y_b(t) = c+phi_b*y_b(t-1) + epsilon_b(t);
end
plot(y_b)
title('AR(1) with constant, E(y_t) = 3, y(1) = 20, 500 obs, for loop')
yline(3, 'r', 'LineWidth', 1.5)
% --- Save each figure automatically as PDF ---
    filename = sprintf('Ex2a.pdf');
    exportgraphics(gcf, filename, 'ContentType', 'vector', 'BackgroundColor', 'white');

% burn-in
T_burn = 500;            % burn-in

eps_long = sqrt(sigma2_b)*randn(T_burn+T,1);

y_long = zeros(T_burn+T,1);
y_long(1) = 20;
for t = 2:(T_burn+T)
    y_long(t) = c + phi_b*y_long(t-1) + eps_long(t);
end

y_burned = y_long(T_burn+1:end);
plot(y_burned)
yline(3, 'r', 'LineWidth', 1.5)
title('AR(1), E(y_t) = 3, y(1) = 20, burn-in = 500')
% --- Save each figure automatically as PDF ---
    filename = sprintf('Ex2b.pdf');
    exportgraphics(gcf, filename, 'ContentType', 'vector', 'BackgroundColor', 'white');

%%

%%%%%% EXERCISE THREE %%%%%%

% OLS estimator: beta = \frac{Cov(y_t,y_{t-1})}{V(y_{t-1})}
sigma2_c = 0.3;
phi_c = 0.4;
T_c = 250;
T_burn = 200;
B = 5000;

beta = zeros(B,1);
tstats = zeros(B,1);

for b = 1:B
    epsilon_c  = sqrt(sigma2_c)*randn(T_c + T_burn,1);
    y_c = zeros(T_c+T_burn,1);
    y_c(1) = 0;
    for t = 2:(T_c + T_burn)
        y_c(t) = phi_c*y_c(t-1) + epsilon_c(t);
    end
    y_c = y_c(T_burn+1:end);             %discard the burn-in
    Y  = y_c(2:end);
    X  = y_c(1:end-1);

    beta(b) = (X'*Y)/(X'*X);
    u = Y - beta(b)*X;                   % residuals
    RSS = u' * u;
    df  = (T_c - 1) - 1;
    sigma2_hat = RSS / df;
    se_beta = sqrt( sigma2_hat / (X' * X) );
    tstat_b = beta(b) / se_beta;         

    tstats(b) = tstat_b;

end
histogram (beta)
title ('OLS estimator distribution for AR(1), T=250, var=0.3, 5000 repetitions')% --- Save each figure automatically as PDF ---
    filename = sprintf('Ex3a.pdf');
    exportgraphics(gcf, filename, 'ContentType', 'vector', 'BackgroundColor', 'white');

crit = 1.96;           % 95% threshold, two-tail
reject = abs(tstats) > crit;
rejection_rate = mean(reject);       % rejection frequence (in [0,1])
fprintf('Rejection rate @95%% (T=250): %.3f\n', rejection_rate);

% statistics
mean_beta = mean(beta);
std_beta  = std(beta);
bias_beta = mean_beta - phi_c;    % bias

fprintf('Mean(phî) = %.4f\n', mean_beta);
fprintf('Std(phî)  = %.4f\n', std_beta);
fprintf('Bias(phî) = %.4f\n', bias_beta);

%% ===============================================================
%  Exercise 4 - Macroeconometrics Problem Set 1
%  Empirical distribution of the OLS estimator in AR(1)

%% === Problem parameters ===
phi_true = 0.9;                      % true value of phi
T_values = [50, 100, 200, 1000];     % sample sizes
nSim     = 1000;                     % number of Monte Carlo simulations
sigma_2  = 0.3;                      % variance of white noise
burnin   = 200;                      % burn-in period

% Store all estimates
phi_hat_all = cell(length(T_values),1); 
mean_vals = zeros(length(T_values),1);
std_vals  = zeros(length(T_values),1);
delta_vals = zeros(length(T_values),1);

%% === Loop over different sample sizes ===
for k = 1:length(T_values)
    
    T = T_values(k);
    N = T + burnin;                 % total simulated length (with burn-in)
    phi_estimates = zeros(nSim,1);
    
    for s = 1:nSim
        
        % ---- 1. Simulate AR(1) process with burn-in ----
        x = zeros(N,1);                       
        eps = sqrt(sigma_2) * randn(N,1);     % white noise
        
        for t = 2:N
            x(t) = phi_true * x(t-1) + eps(t);
        end
        
        % Discard burn-in observations
        x = x(burnin+1:end);
        
        % ---- 2. OLS estimation of phi ----
        y = x(2:end);
        X = x(1:end-1);
        phi_est = (X' * y) / (X' * X);
        phi_estimates(s) = phi_est;
    end
    
    % Store stats
    phi_hat_all{k} = phi_estimates;               
    mean_vals(k) = mean(phi_estimates);
    std_vals(k)  = std(phi_estimates);
    delta_vals(k) = mean_vals(k) - phi_true;
    
end

%% ---- Plot empirical distributions (density, rescaled Y, saved as PDF) ----
x_limits = [0.7, 1.0];   % same x-axis scale for all plots

% 1. Find global maximum density across all histograms
max_density = 0;
for k = 1:length(T_values)
    [counts, edges] = histcounts(phi_hat_all{k}, 40, 'Normalization', 'pdf');
    max_density = max(max_density, max(counts));
end

% 2. Plot each histogram with same Y scale
for k = 1:length(T_values)
    estimates = phi_hat_all{k};

    figure('Color', 'w');
    histogram(estimates, 40, 'Normalization', 'pdf', ...
        'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.85);
    hold on;

    % Reference lines
    xline(phi_true, 'r-', 'LineWidth', 1.8);
    xline(mean_vals(k), 'b--', 'LineWidth', 1.6);

    % Titles and labels
    title(sprintf('Empirical Distribution of OLS Estimates (T = %d)', T_values(k)), ...
        'FontWeight', 'bold', 'FontSize', 12);
    xlabel('Estimated φ', 'FontSize', 12);
    ylabel('Density', 'FontSize', 11);

    % Keep same X and Y scale across all plots
    xlim(x_limits);
    ylim([0, max_density * 1.1]);  % uniform Y scale
    grid on; box on;


    % --- Save each figure automatically as PDF ---
    filename = sprintf('Ex4_fig%d.pdf', k);
    exportgraphics(gcf, filename, 'ContentType', 'vector', 'BackgroundColor', 'white');

    hold off;
end


%% ===============================================================
%  Exercise 5 - Macroeconometrics Problem Set 1
%  OLS estimator of AR(1) when true DGP is MA(1)
%% ===============================================================

clear; clc; close all;
rng(1);

%% === Parameters ===
theta_true = 0.6;                               % true MA(1) parameter
a_true = theta_true / (1 + theta_true^2);       % implied pseudo-AR(1) value
T_values   = [50, 100, 200, 1000];              % sample sizes
nSim       = 1000;                              % number of simulations
sigma_2    = 0.3;                               % variance of white noise
burnin     = 200;                               % burn-in period

% Store estimates and statistics
a_hat_all   = cell(length(T_values),1);
mean_vals   = zeros(length(T_values),1);
std_vals    = zeros(length(T_values),1);
delta_vals  = zeros(length(T_values),1);

%% === Loop over different sample sizes ===
for k = 1:length(T_values)
    
    T = T_values(k);
    N = T + burnin;
    estimates = zeros(nSim,1);
    
    for s = 1:nSim
        % ---- 1. Generate MA(1) process with burn-in ----
        eps = sqrt(sigma_2) * randn(N+1,1);  % N+1 for lag
        x_full = zeros(N,1);
        for t = 1:N
            x_full(t) = eps(t+1) + theta_true * eps(t);
        end
        % Discard burn-in observations
        x = x_full(burnin+1:end);
        
        % ---- 2. Estimate AR(1) by OLS ----
        y = x(2:end);
        X = x(1:end-1);
        a_est = (X' * y) / (X' * X);
        estimates(s) = a_est;
    end
    
    % ---- 3. Compute summary statistics ----
    a_hat_all{k} = estimates;
    mean_vals(k) = mean(estimates);
    std_vals(k)  = std(estimates);
    delta_vals(k) = mean_vals(k) - a_true;
end

%% ---- 4. Determine global Y-scale for density plots ----
x_limits = [0.2, 0.8];  % reasonable range for a estimates
max_density = 0;
for k = 1:length(T_values)
    [counts, edges] = histcounts(a_hat_all{k}, 40, 'Normalization', 'pdf');
    max_density = max(max_density, max(counts));
end

%% ---- 5. Plot empirical distributions ----
for k = 1:length(T_values)
    estimates = a_hat_all{k};

    figure('Color', 'w');
    histogram(estimates, 40, 'Normalization', 'pdf', ...
        'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.85);
    hold on;

    % Reference lines
    xline(a_true, 'r-', 'LineWidth', 1.8);
    xline(mean_vals(k), 'b--', 'LineWidth', 1.6);

    % Titles and labels
    title(sprintf('OLS Estimates (true DGP: MA(1), T = %d)', T_values(k)), ...
        'FontWeight', 'bold', 'FontSize', 12);
    xlabel('Estimated a', 'FontSize', 12);
    ylabel('Density', 'FontSize', 11);

    % Consistent scale across plots
    xlim(x_limits);
    ylim([0, max_density * 1.1]);
    grid on; box on;



    % --- Save figure automatically as PDF ---
    filename = sprintf('Ex5_fig%d.pdf', k);
    exportgraphics(gcf, filename, 'ContentType', 'vector', 'BackgroundColor', 'white');

    hold off;
end

%% ---- 6. Summary table ----
true_vals = a_true * ones(length(T_values),1);
ResultsTable = table(T_values(:), true_vals, mean_vals, std_vals, delta_vals, ...
    'VariableNames', {'T','True_aStar','Mean_aHat','Std_aHat','Delta_(Mean-True)'});

disp('Summary statistics for OLS estimator when true DGP is MA(1):')
disp(ResultsTable);


%% ===============================================================
%  Exercise 6 - Macroeconometrics Problem Set 1
% ================================================================


rng(123); clearvars -except rng; close all;

% --- Parameters ---
T       = 300;
sigma2  = 1;
phi     = [0.7, -0.3];
theta   = [0.5, 0.4];

% --- Generate ARMA(2,2) sample using local function below ---
[x, eps] = generate_ARMA(T, sigma2, phi, theta);

% --- Plot the simulated series ---
figure('Color','w');
plot(x, 'LineWidth', 1.3);
title('Exercise 6: Simulated ARMA(2,2) Series');
xlabel('Time');
ylabel('x_t');
grid on;

exportgraphics(gcf, 'Ex6_fig1.pdf', 'ContentType', 'vector');
