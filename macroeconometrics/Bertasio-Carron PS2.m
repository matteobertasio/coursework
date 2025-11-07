%% ===============================================================
%  Problem Set 2 - Exercise 1(a)
%  Empirical distribution of the OLS estimator under a pure random walk
%  Model:  y_t = y_{t-1} + eps_t ,   T = 250
%% ===============================================================


clear; clc; close all;

% --- Reproducibility ---
rng(123);

%% === Monte Carlo parameters ===
T       = 250;         % sample size
nSim    = 1000;        % number of Monte Carlo replications
sigma2  = 1;           % variance of the innovation [I set the variance ==1]
sigma   = sqrt(sigma2);

% Store all OLS estimates of rho
rho_hat = zeros(nSim,1);

%% === Monte Carlo loop ===
for s = 1:nSim
    
    % ----- 1. Generate a pure random walk -----
    y   = zeros(T,1);
    eps = sigma * randn(T,1);   % innovations ~ N(0, sigma2)
    
    for t = 2:T
        y(t) = y(t-1) + eps(t);   % random walk without drift
    end
    
    % ----- 2. Estimate OLS: y_t = rho * y_{t-1} + u_t -----
    y_lag = y(1:end-1);
    y_now = y(2:end);
    
    % OLS without intercept
    rho_hat(s) = (y_lag' * y_now) / (y_lag' * y_lag);
    
end

%% === Output summary ===
fprintf('Mean of OLS estimates (rho): %.4f\n', mean(rho_hat));
fprintf('Std. deviation             : %.4f\n', std(rho_hat));
fprintf('Median of OLS estimates    : %.4f\n', median(rho_hat))

% --- Plot empirical distribution ---
figure('Color','w');
histogram(rho_hat, 40, 'Normalization','pdf', ...
    'FaceColor',[0.3 0.6 0.9], 'EdgeColor','none', 'FaceAlpha',0.8);
hold on;
xline(1, 'r-', 'LineWidth',1.5);     % true value under H0: unit root
xlabel('\rhô');
ylabel('Density');
grid on;

% --- export as PDF for LaTeX ---
filename = 'Figure_1.pdf';
exportgraphics(gcf, filename, 'ContentType','vector', 'BackgroundColor','white');

%% ===============================================================
%  Problem Set 2 - Exercise 1(b)
%  Dickey–Fuller type t-test under a unit root
%  H0: rho = 0   vs   H1: rho < 0
%  Test regression:  Δy_t = α + ρ y_{t-1} + ε_t
%% ===============================================================

clear; clc; close all;

%% --- reproducibility ---
rng(123);

%% === simulation settings ===
T      = 250;      % sample size
nSim   = 1000;     % number of Monte Carlo replications
sigma  = 1;        % std of innovations

% preallocate
t_stats         = zeros(nSim,1);   % DF t-statistics
rejected_normal = false(nSim,1);   % indicator: reject using N(0,1)?

% 5% *left-tailed* critical value from standard Normal
crit_normal_5 = -1.645;

%% === Monte Carlo loop ===
for s = 1:nSim
    
    % ----- 1. generate a pure random walk (unit root) -----
    y   = zeros(T,1);
    eps = sigma * randn(T,1);
    for t = 2:T
        y(t) = y(t-1) + eps(t);
    end
    
    % ----- 2. build DF regression variables -----
    dy   = diff(y);        % Δy_t , length T-1
    ylag = y(1:end-1);     % y_{t-1}, length T-1
    
    % ----- 3. OLS: Δy_t = α + ρ y_{t-1} + u_t -----
    X = [ones(T-1,1), ylag];
    b = X \ dy;            % OLS estimates: [alpha_hat; rho_hat]
    
    % residuals
    uhat = dy - X*b;
    
    % estimate error variance: (T-1) obs, 2 parameters
    dof  = (T-1) - size(X,2);     % degrees of freedom = 248
    s2   = (uhat' * uhat) / dof;
    
    % variance-covariance matrix of OLS
    varb = s2 * inv(X' * X);
    
    rho_hat = b(2);
    se_rho  = sqrt(varb(2,2));
    
    % ----- 4. DF t-stat -----
    t_val = rho_hat / se_rho;
    t_stats(s) = t_val;
    
    % ----- 5. rejection using WRONG N(0,1) critical value -----
    rejected_normal(s) = (t_val < crit_normal_5);
end

%% === summary results ===
rej_rate  = mean(rejected_normal);     % how often do we reject?
t_mean    = mean(t_stats);
t_median  = median(t_stats);
t_skew    = skewness(t_stats);
t_kurt    = kurtosis(t_stats);
t_prc     = prctile(t_stats, [1 5 10 50 90 95 99]);

fprintf('\n=== PS2 - Exercise 1(b): DF t-test ===\n');
fprintf('Rejection rate using N(0,1) 5%%%% left-tail : %.3f\n', rej_rate);
fprintf('Mean of DF t-stats                         : %.3f\n', t_mean);
fprintf('Median of DF t-stats                       : %.3f\n', t_median);
fprintf('Skewness of DF t-stats                     : %.3f\n', t_skew);
fprintf('Kurtosis of DF t-stats                     : %.3f\n', t_kurt);
fprintf('Percentiles [1 5 10 50 90 95 99]           :\n');
disp(t_prc);

%% === plot empirical distribution of DF t-stat ===
figure('Color','w');
histogram(t_stats, 40, 'Normalization','pdf', ...
    'FaceColor',[0.3 0.6 0.9], 'EdgeColor','none', 'FaceAlpha',0.85);
hold on;
xline(crit_normal_5, 'r--', 'LineWidth', 1.4);   % Normal 5% critical value

xlabel('t-statistic for \rho');
ylabel('Density');
title('Empirical distribution of DF t-stat under H_0: unit root');
legend({'Empirical t-stats','Normal 5% critical value'}, 'Location','best');
grid on;


% --- export as PDF for LaTeX ---
filename = 'Figure_2.pdf';
exportgraphics(gcf, filename, 'ContentType','vector', 'BackgroundColor','white');



%% ===============================================================
%  Problem Set 2 - Exercise 1(c)
%  Compare empirical percentiles of the DF t-statistic
%  with Dickey–Fuller critical values
%% ===============================================================


%% === 1. Compute empirical percentiles ===
percentiles = [0.01 0.025 0.05 0.10 0.50 0.90 0.95 0.975 0.99];
empirical_q = quantile(t_stats, percentiles);

fprintf('\nEmpirical percentiles of the DF t-statistic (T=%d, nSim=%d)\n', T, nSim);
disp(table(percentiles', empirical_q', 'VariableNames', {'Prob','Empirical_tstat'}));

%% === 2. Compare with standard DF critical values ===
% Typical DF critical values (no trend, intercept included)
% Source: Enders (4th ed.) Table 4.A.2 or Hamilton Appendix
df_crit = [-3.43, -3.12, -2.86];    % 1%, 2.5%, 5%

fprintf('\nTypical Dickey–Fuller critical values (with intercept):\n');
fprintf('  1%%  = %.2f\n', df_crit(1));
fprintf('  2.5%%= %.2f\n', df_crit(2));
fprintf('  5%%  = %.2f\n', df_crit(3));

%% === 3. Quick visual comparison ===
figure('Color','w');
histogram(t_stats, 40, 'Normalization','pdf', ...
    'FaceColor',[0.3 0.6 0.9], 'EdgeColor','none', 'FaceAlpha',0.8);
hold on;
xline(df_crit(3), 'r--', 'LineWidth',1.4, 'DisplayName','DF 5% critical value');
xline(-1.645, 'k--', 'LineWidth',1.2, 'DisplayName','Normal 5% critical value');
title('Empirical DF t-statistics and critical values');
xlabel('t-statistic');
ylabel('Density');
legend('Empirical distribution','DF 5%','Normal 5%','Location','best');
grid on;

saveas(gcf, 'PS2_Ex1c_DF_percentiles.png');



%% ===============================================================
%  Problem Set 2 - Exercise 2
%  Spurious regression (Enders, pp. 195–199)
%
%  Goal:
%  - Show, via Monte Carlo, that regressing nonstationary variables
%    can produce apparently significant t-statistics and high R^2
%    even when the variables are unrelated (spurious regression).
%  - Compare 4 cases as in Enders:
%       (1) both stationary  -> valid regression
%       (2) y nonstationary, x stationary -> meaningless
%       (3) both nonstationary, independent -> spurious regression
%       (4) both nonstationary, cointegrated -> valid again
%
%  Output:
%  - rejection rates (|t| > 1.96)
%  - average R^2
%  - plots of example time series (one replication per case)
%  - plots of t-stat and R^2 distributions (case 1 vs case 3)
%  - all figures saved as PDF for LaTeX
%% ===============================================================

clear; clc; close all;
rng(123);                      % reproducibility

%% === simulation settings ===
T    = 250;                    % sample size
nSim = 1000;                   % number of Monte Carlo replications
crit = 1.96;                   % ~5% two-sided critical value

% container for results
results = struct();

% helper OLS function
ols_stats = @(y,x) deal_stats(y,x);

%% ===============================================================
%  CASE 1: both STATIONARY (benchmark, "good regression")
%  y_t = 0.5 y_{t-1} + e_t
%  x_t = 0.4 x_{t-1} + v_t
%  -> both I(0): OLS should be fine, small R^2, rejection ~near nominal
%% ===============================================================
phi_y = 0.5; 
phi_x = 0.4;

beta1 = zeros(nSim,1);
t1    = zeros(nSim,1);
R21   = zeros(nSim,1);

for s = 1:nSim
    e1 = randn(T,1);
    e2 = randn(T,1);
    y  = filter(1,[1 -phi_y], e1);     % stationary AR(1)
    x  = filter(1,[1 -phi_x], e2);     % stationary AR(1)
    [b, tval, R2] = ols_stats(y,x);
    beta1(s) = b;
    t1(s)    = tval;
    R21(s)   = R2;
    % store first replication to plot later
    if s == 1
        y_case1 = y;
        x_case1 = x;
    end
end
results.case1 = struct('t',t1,'R2',R21);

%% ===============================================================
%  CASE 2: y NONSTATIONARY (random walk), x STATIONARY
%  -> different orders of integration: regression has no meaning
%% ===============================================================
phi_x = 0.5;

t2  = zeros(nSim,1);
R22 = zeros(nSim,1);

for s = 1:nSim
    y = cumsum(randn(T,1));           % random walk -> nonstationary
    x = filter(1,[1 -phi_x], randn(T,1));  % stationary AR(1)
    [~, tval, R2] = ols_stats(y,x);
    t2(s)  = tval;
    R22(s) = R2;
    if s == 1
        y_case2 = y;
        x_case2 = x;
    end
end
results.case2 = struct('t',t2,'R2',R22);

%% ===============================================================
%  CASE 3: both NONSTATIONARY, independent random walks
%  -> classic spurious regression (Enders, pp. 195–199)
%  -> high R^2 and many large t-stats even with no true relation
%% ===============================================================
t3  = zeros(nSim,1);
R23 = zeros(nSim,1);

for s = 1:nSim
    y = cumsum(randn(T,1));           % random walk
    x = cumsum(randn(T,1));           % independent random walk
    [~, tval, R2] = ols_stats(y,x);
    t3(s)  = tval;
    R23(s) = R2;
    if s == 1
        y_case3 = y;
        x_case3 = x;
    end
end
results.case3 = struct('t',t3,'R2',R23);

%% ===============================================================
%  CASE 4: both NONSTATIONARY, but COINTEGRATED
%  y_t = x_t + noise, with x_t random walk
%  -> they share the same stochastic trend, residuals are I(0)
%  -> regression becomes meaningful again, high R^2 is genuine
%% ===============================================================
t4  = zeros(nSim,1);
R24 = zeros(nSim,1);

for s = 1:nSim
    u = randn(T,1); 
    v = randn(T,1);
    x = cumsum(u);               % common stochastic trend
    y = x + 0.5*v;               % y shares the trend -> cointegrated
    [~, tval, R2] = ols_stats(y,x);
    t4(s)  = tval;
    R24(s) = R2;
    if s == 1
        y_case4 = y;
        x_case4 = x;
    end
end
results.case4 = struct('t',t4,'R2',R24);

%% ===============================================================
%  NUMERIC SUMMARY (what the PS wants)
%  Rejection rates and mean R^2 per case
%% ===============================================================
rej1 = mean(abs(results.case1.t) > crit);
rej2 = mean(abs(results.case2.t) > crit);
rej3 = mean(abs(results.case3.t) > crit);
rej4 = mean(abs(results.case4.t) > crit);

mR21 = mean(results.case1.R2);
mR22 = mean(results.case2.R2);
mR23 = mean(results.case3.R2);
mR24 = mean(results.case4.R2);

fprintf('\n=== Rejection rates (|t| > 1.96) ===\n');
fprintf('Case 1 (stationary, stationary)        : %.3f\n', rej1);
fprintf('Case 2 (nonstationary y, stationary x) : %.3f\n', rej2);
fprintf('Case 3 (both nonstationary, indep.)    : %.3f\n', rej3);
fprintf('Case 4 (both nonstationary, coint.)    : %.3f\n', rej4);

fprintf('\n=== Average R^2 by case ===\n');
fprintf('Case 1: %.3f\n', mR21);
fprintf('Case 2: %.3f\n', mR22);
fprintf('Case 3: %.3f\n', mR23);
fprintf('Case 4: %.3f\n', mR24);

%% ===============================================================
%  PLOT 1: sample time series for the 4 cases
%  (this is useful to show how the data look before regression)
%% ===============================================================
figure('Color','w');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
plot(y_case1,'b','LineWidth',1); hold on;
plot(x_case1,'r','LineWidth',1);
title('Case 1: both stationary'); legend('y','x'); grid on;

nexttile;
plot(y_case2,'b','LineWidth',1); hold on;
plot(x_case2,'r','LineWidth',1);
title('Case 2: y nonstationary, x stationary'); legend('y','x'); grid on;

nexttile;
plot(y_case3,'b','LineWidth',1); hold on;
plot(x_case3,'r','LineWidth',1);
title('Case 3: both nonstationary, independent'); legend('y','x'); grid on;

nexttile;
plot(y_case4,'b','LineWidth',1); hold on;
plot(x_case4,'r','LineWidth',1);
title('Case 4: both nonstationary, cointegrated'); legend('y','x'); grid on;

% save figure as PDF for LaTeX
print(gcf, 'PS2_Ex2_sample_series', '-dpdf', '-r300');

%% ===============================================================
%  PLOT 2: distributions of t-stat and R^2 for Case 1 vs Case 3
%  -> shows clearly the spurious effect: case 3 has many large t-stats
%     and much higher R^2, even though variables are unrelated
%% ===============================================================
figure('Color','w');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

% t-stat, case 1
nexttile;
histogram(results.case1.t, 40, 'Normalization','pdf', 'FaceAlpha',0.8);
hold on; xline(-crit,'r--'); xline(crit,'r--');
title('t-stat: Case 1 (stationary)'); xlabel('t'); ylabel('Density'); grid on;

% t-stat, case 3
nexttile;
histogram(results.case3.t, 40, 'Normalization','pdf', 'FaceAlpha',0.8);
hold on; xline(-crit,'r--'); xline(crit,'r--');
title('t-stat: Case 3 (nonstationary, indep.)'); xlabel('t'); ylabel('Density'); grid on;

% R^2, case 1
nexttile;
histogram(results.case1.R2, 40, 'Normalization','pdf', 'FaceAlpha',0.8);
title('R^2: Case 1'); xlabel('R^2'); ylabel('Density'); grid on;

% R^2, case 3
nexttile;
histogram(results.case3.R2, 40, 'Normalization','pdf', 'FaceAlpha',0.8);
title('R^2: Case 3'); xlabel('R^2'); ylabel('Density'); grid on;

sgtitle('Exercise 2: Spurious regression (Case 1 vs Case 3)');

% save as PDF
print(gcf, 'PS2_Ex2_spurious_comparison', '-dpdf', '-r300');


%% ===============================================================
%  PLOT A: sample time series, ONE FIGURE PER CASE
%  (così in LaTeX puoi fare \begin{subfigure} ... \end{subfigure})
%% ===============================================================

% --- Case 1 ---
f1 = figure('Color','w');
plot(y_case1,'b','LineWidth',1); hold on;
plot(x_case1,'r','LineWidth',1);
title('Case 1: both stationary'); legend('y','x','Location','best'); grid on;
xlabel('t'); ylabel('level');


% --- Case 2 ---
f2 = figure('Color','w');
plot(y_case2,'b','LineWidth',1); hold on;
plot(x_case2,'r','LineWidth',1);
title('Case 2: y nonstationary, x stationary'); legend('y','x','Location','best'); grid on;
xlabel('t'); ylabel('level');
print(f2, 'PS2_Ex2_TS_case2', '-dpdf', '-r300');


filename = sprintf('PS2_Ex2_TS_case2.pdf');
exportgraphics(f2, filename, 'ContentType','vector', 'BackgroundColor','white');

% --- Case 3 ---
f3 = figure('Color','w');
plot(y_case3,'b','LineWidth',1); hold on;
plot(x_case3,'r','LineWidth',1);
title('Case 3: both nonstationary, independent'); legend('y','x','Location','best'); grid on;
xlabel('t'); ylabel('level');
print(f3, 'PS2_Ex2_TS_case3', '-dpdf', '-r300');

filename = sprintf('PS2_Ex2_TS_case3.pdf');
exportgraphics(f3, filename, 'ContentType','vector', 'BackgroundColor','white');

% --- Case 4 ---
f4 = figure('Color','w');
plot(y_case4,'b','LineWidth',1); hold on;
plot(x_case4,'r','LineWidth',1);
title('Case 4: both nonstationary, cointegrated'); legend('y','x','Location','best'); grid on;
xlabel('t'); ylabel('level');

filename = sprintf('PS2_Ex2_TS_case4.pdf');
exportgraphics(f4, filename, 'ContentType','vector', 'BackgroundColor','white');


%% ===============================================================
%  PLOT B: distributions (t-stat, R2) per case – salviamo separati
%% ===============================================================

% --- t-stat Case 1 ---
ft1 = figure('Color','w');
histogram(results.case1.t, 40, 'Normalization','pdf', 'FaceAlpha',0.8);
hold on; 
xline(-crit,'r--'); 
xline(crit,'r--');
title('t-stat: Case 1 (stationary)'); 
xlabel('t'); ylabel('Density'); grid on;
filename = 'PS2_Ex2_tstat_case1.pdf';
exportgraphics(ft1, filename, 'ContentType','vector', 'BackgroundColor','white');
hold off;

% --- t-stat Case 3 ---
ft3 = figure('Color','w');
histogram(results.case3.t, 40, 'Normalization','pdf', 'FaceAlpha',0.8);
hold on; 
xline(-crit,'r--'); 
xline(crit,'r--');
title('t-stat: Case 3 (nonstationary, indep.)'); 
xlabel('t'); ylabel('Density'); grid on;
filename = 'PS2_Ex2_tstat_case3.pdf';
exportgraphics(ft3, filename, 'ContentType','vector', 'BackgroundColor','white');
hold off;

% --- R^2 Case 1 ---
fr1 = figure('Color','w');
histogram(results.case1.R2, 40, 'Normalization','pdf', 'FaceAlpha',0.8);
title('R^2: Case 1'); 
xlabel('R^2'); ylabel('Density'); grid on;
filename = 'PS2_Ex2_R2_case1.pdf';
exportgraphics(fr1, filename, 'ContentType','vector', 'BackgroundColor','white');
hold off;

% --- R^2 Case 3 ---
fr3 = figure('Color','w');
histogram(results.case3.R2, 40, 'Normalization','pdf', 'FaceAlpha',0.8);
title('R^2: Case 3'); 
xlabel('R^2'); ylabel('Density'); grid on;
filename = 'PS2_Ex2_R2_case3.pdf';
exportgraphics(fr3, filename, 'ContentType','vector', 'BackgroundColor','white');
hold off;



clear; close all; clc; rng(123);
%%
%%%%%% EXERCISE THREE %%%%%%

%% parameters
T_keep = 500;
B = 200;
T = B + T_keep;
beta = 0.6;
sigma_2_eta = 1;
sigma_2_eps = 0.8;

eta = sqrt(sigma_2_eta)*randn(T,1);
eps_today = sqrt(sigma_2_eps)*randn(T,1);
% eps_yesterday
eps_yesterday=zeros(T,1);
eps_yesterday(1)=0;
for t = 2:T
    eps_yesterday(t) = eps_today(t-1);
end   
% eps_twodays
eps_twodays=zeros(T,1);
eps_twodays(1)=0;
for t = 2:T
    eps_twodays(t) = eps_yesterday(t-1);
end 

% x and y
x = zeros(T, 1);
y = zeros(T, 1);
for t = 1:T
    x(t) = eta(t) + eps_twodays(t);
    y(t) = (beta/(1-beta)) * eta(t) + ((beta^2)/(1-beta)) * eps_today(t) + beta*eps_yesterday(t);
end

% burn-in
s = B+1;
x_keep = x(s:end);
y_keep = y(s:end);
X = [x_keep , y_keep];

% graphs of y_t and x_t
t_keep = (1:numel(x_keep))';
tiledlayout(2,1);

nexttile; grid on;
plot(t_keep, x_keep, 'LineWidth', 1.2);
ylabel('x\_keep'); title('x_t and y_t after burn-in');

nexttile; grid on;
plot(t_keep, y_keep, 'LineWidth', 1.2);
xlabel('t'); ylabel('y\_keep');

outdir = 'Ex3';
if ~exist(outdir,'dir'), mkdir(outdir); end
exportgraphics(gcf, fullfile(outdir, 'xy_keep_overlay.pdf'), ...
               'ContentType','vector','BackgroundColor','white');

%% VAR(4) and structural IRFs via C(L)A (using Econometrics Toolbox)
p = 4; H = 20; N = 500;
EstMdl = estimate(varm(2,p), X, 'Display','off');

% VMA coefficients C_h recursively
C = zeros(2,2,H); 
C(:,:,1) = eye(2);
for h = 2:H
    tmp = zeros(2);
    for j = 1:min(p,h-1)
        tmp = tmp + EstMdl.AR{j} * C(:,:,h-j);
    end
    C(:,:,h) = tmp;
end

% matrix A from the handout
A = [1, 0; beta/(1-beta), (beta^2)/(1-beta)];

IRF_hat = zeros(H,2,2);
for h = 1:H
    IRF_hat(h,:,:) = C(:,:,h) * A;   % unit structural shocks
end

% "real" IRF of the DGP
IRF_true = zeros(H,2,2);
% shock 1 = eta_t
IRF_true(1,1,1) = 1;                     
IRF_true(1,2,1) = (beta/(1-beta));       
% shock 2 = eps_t
IRF_true(1,2,2) = (beta^2/(1-beta));     
IRF_true(2,2,2) = beta;                  
IRF_true(3,1,2) = 1;                     

% Monte Carlo
IRF_stack = zeros(H,2,2,N);

for n = 1:N
    % DGP + burn-in
    eta_n       = sqrt(sigma_2_eta) * randn(T,1);
    eps_today_n = sqrt(sigma_2_eps) * randn(T,1);
    eps_yest_n  = [0; eps_today_n(1:end-1)];
    eps_2day_n  = [0; eps_yest_n(1:end-1)];

    x_n = eta_n + eps_2day_n;
    y_n = (beta/(1-beta))*eta_n + ((beta^2)/(1-beta))*eps_today_n + beta*eps_yest_n;

    Xn = [x_n(B+1:end), y_n(B+1:end)];

    % VAR(4) and IRF
    EstMdl_n = estimate(varm(2,p), Xn, 'Display','off');
    % VMA of the estimated VAR (reduced form)
    C_n = zeros(2,2,H); 
    C_n(:,:,1) = eye(2);
        for h = 2:H
            tmp = zeros(2);
            for j = 1:min(p,h-1)
                tmp = tmp + EstMdl_n.AR{j} * C_n(:,:,h-j);
            end
            C_n(:,:,h) = tmp;
        end
        for h = 1:H
            IRF_stack(h,:,:,n) = C_n(:,:,h) * A;
        end
end
% percentiles
IRF_mean = mean(IRF_stack, 4);
IRF_lo   = prctile(IRF_stack, 2.5, 4);
IRF_hi   = prctile(IRF_stack, 97.5, 4);

% graphs
t = (0:H-1)';
var_names   = {'x','y'};
shock_names = {'\eta-shock','\epsilon-shock'};

outdir = 'Ex3';
if ~exist(outdir,'dir'), mkdir(outdir); end
letters = 'abcd';                  % 1..4 -> a,b,c,d

for j = 1:2           % shock
  for i = 1:2
    f = figure; hold on; grid on;

    % MC band behind the lines
    hBand = fill([t; flipud(t)], ...
                 [squeeze(IRF_lo(:,i,j)); flipud(squeeze(IRF_hi(:,i,j)))], ...
                 [0.9 0.9 0.9], 'EdgeColor','none', ...
                 'DisplayName','95% MC band');

    hMean = plot(t, squeeze(IRF_mean(:,i,j)), 'k',  'LineWidth', 1.6, ...
                 'DisplayName','MC mean');
    hTrue = plot(t, squeeze(IRF_true(:,i,j)), '--', 'LineWidth', 1.6, ...
                 'DisplayName','True DGP');
    hHat  = plot(t, squeeze(IRF_hat(:,i,j)), '-.', 'LineWidth', 1.2, ...
                 'DisplayName','Sample C(L)A');

    legend([hBand hMean hTrue hHat], 'Location','Best');
    xlabel('h');
    ylabel(sprintf('IRF %s \\leftarrow %s', var_names{i}, shock_names{j}));
    title(sprintf('IRF %s \\leftarrow %s (VAR(4) C(L)A, [x,y])', var_names{i}, shock_names{j}));

    idx = (j-1)*2 + i;
    filename = fullfile(outdir, sprintf('Ex3%c.pdf', letters(idx)));
    exportgraphics(f, filename, 'ContentType','vector', 'BackgroundColor','white');
  end
end
%%
%%%%%% EXERCISE FOUR %%%%%%

%% Import Romer and Romer Excel Shocks
fname = 'Romer_Romer.xlsx';
T = readtable(fname, 'Sheet','Foglio1', 'VariableNamingRule','preserve');
T.Properties.VariableNames = {'Date','inflation','unemployment','ffr','rr_shock'};

% Y matrix
Y = [T.inflation, T.unemployment, T.ffr, T.rr_shock];

%% VAR(4) and conditional Granger causality
p = 4;
Mdl   = varm(4,p);
EstMdl = estimate(Mdl, Y, 'Display','off');
names = T.Properties.VariableNames(2:4);   % {'inflation','unemployment','ffr'}

% RR → others
for k = 1:3
    Z = Y(:, setdiff(1:4, [4 k]));
    [~,p] = gctest(Y(:,4), Y(:,k), Z, 'NumLags',4,'Constant',true);
    fprintf('RR shocks -> %-13s : p=%.4g\n', names{k}, p);
end

% others → RR
for k = 1:3
    Z = Y(:, setdiff(1:4, [k 4]));
    [~,p] = gctest(Y(:,k), Y(:,4), Z, 'NumLags',4,'Constant',true);
    fprintf('%-13s -> RR shocks : p=%.4g\n', names{k}, p);
end
