clear; clc; close all;

%% ============================================================
%  LOAD DATA
%  Variables in sheet 1:
%    - LOG_GDP_  : log real GDP
%    - LOG_P_    : log price deflator
%    - FFR       : federal funds rate
%% ============================================================

TBL = readtable('data_ps3.xlsx','Sheet',1);

logY = TBL.LOG_GDP_;
logP = TBL.LOG_P_;
FFR  = TBL.FFR;

% Stack series: y_t = [logY_t, logP_t, FFR_t]
y = [logY, logP, FFR];          % T x N matrix
[T, N] = size(y);               % N = 3

p = 4;                          % VAR(4), quarterly data

% Build dependent and regressor matrices
Y = y(p+1:end, :);              % dependent variables (Teff x N)
T_eff = size(Y,1);

X = [];
for i = 1:p
    X = [X, y(p+1-i:end-i,:)];  % lags y_{t-1},...,y_{t-p}
end
X = [ones(T_eff,1), X];         % add constant
k = size(X,2);                  % number of regressors (1 + N*p)

%% ============================================================
%  POINT 1 — Estimate VAR, store coefficients + residuals
%% ============================================================

% OLS estimation: Y = X * B + U
B = X \ Y;                      % coefficient matrix (k x N)
U = Y - X*B;                    % residuals (Teff x N)

% Extract constant and lag coefficient matrices
c_hat = B(1,:)';                % N x 1 constant vector
A_hat = zeros(N, N*p);          % [A1 A2 A3 A4]

for i = 1:p
    rows = 1 + (i-1)*N + (1:N);                 % rows in B for lag i
    A_hat(:, (1+(i-1)*N):(i*N)) = B(rows,:)';   % each block is N x N
end

% Residual covariance matrix with DoF correction
SigmaU = (U' * U)/(T_eff - k);  % N x N

disp('===============================================');
disp('POINT 1 DONE: VAR(4) estimated on original data.');
disp('Coefficient matrix B (constant + 4 lags):');
disp(B);
disp('Residual covariance matrix SigmaU:');
disp(SigmaU);
disp('===============================================');

%% ============================================================
%  BASELINE STRUCTURAL IRFs FROM ORIGINAL VAR (PHASE A)
%% ============================================================

% Cholesky factor (triangular identification): SigmaU = P0 * P0'
P0 = chol(SigmaU, 'lower');     % N x N

% Companion form of VAR(4)
F0 = zeros(N*p);
F0(1:N, :) = A_hat;                             % first block row
F0(N+1:end, 1:N*(p-1)) = eye(N*(p-1));         % subdiagonal identity blocks

% Selector matrix J: picks y_t from the companion state vector z_t
J = [eye(N), zeros(N, N*(p-1))];               % N x (N*p)

% IRF horizon
H = 20;                                        % 20 quarters
IRF0 = zeros(H+1, N, N);                       % IRF0(h+1, i, j)

Fpow = eye(N*p);                               % F^0

for h = 0:H
    Theta_h = J * Fpow * J' * P0;              % N x N
    IRF0(h+1,:,:) = Theta_h;
    Fpow = Fpow * F0;                          % update F^{h+1}
end

% Monetary policy shock = 3rd structural shock (FFR)
mp_shock = 3;
IRF_baseline_mp = squeeze(IRF0(:,:,mp_shock)); % (H+1) x N

%% Optional: display basic info about baseline IRFs
disp('Baseline IRFs to a monetary policy shock (first few horizons):');
for i = 1:N
    fprintf('Variable %d, h=0: %.4f\n', i, IRF_baseline_mp(1,i));
end

%% ============================================================
%  BASELINE FEVD (FORECAST ERROR VARIANCE DECOMPOSITION)
%% ============================================================

FEVD0 = zeros(H+1, N, N);   % FEVD0(h+1, i, j)

for h = 0:H
    irfs_0toh = IRF0(1:h+1,:,:);   % stack IRFs from 0 to h
    for i = 1:N       % variable index
        num = zeros(1,N);
        for j = 1:N   % shock index
            num(j) = sum( squeeze(irfs_0toh(:,i,j)).^2 );
        end
        FEVD0(h+1,i,:) = num / sum(num);
    end
end

% FEVD due to the monetary policy shock (3rd shock)
fevd_baseline_mp = squeeze(FEVD0(:,:,mp_shock));  % (H+1) x N

% Display FEVD at selected horizons for Overleaf table
horizons_to_show = [1 4 8 16];  % horizons in quarters
disp('-----------------------------------------------');
disp('Baseline FEVD: share explained by monetary policy shock');
for hq = horizons_to_show
    idx = hq + 1;  % because index h=0 corresponds to row 1
    fprintf('Horizon %2d: logY = %.3f, logP = %.3f, FFR = %.3f\n', ...
        hq, fevd_baseline_mp(idx,1), fevd_baseline_mp(idx,2), fevd_baseline_mp(idx,3));
end
disp('-----------------------------------------------');

%% Optional: plot baseline IRFs (without bands)
figure;
vars = {'logY','logP','FFR'};
for i = 1:N
    subplot(3,1,i);
    plot(0:H, IRF_baseline_mp(:,i), 'k', 'LineWidth', 2); hold on;
    yline(0, ':');
    title(['Baseline IRF of ', vars{i}, ' to monetary policy shock']);
    xlabel('Horizon (quarters)');
    grid on;
end

% --- export as PDF for LaTeX ---
filename = 'Figure_1.pdf';
exportgraphics(gcf, filename, 'ContentType','vector', 'BackgroundColor','white');

%% ============================================================
%  POINT 2 — Sample WITH replacement from residuals (FIRST DRAW)
%% ============================================================

PER = randi(T_eff, T_eff, 1);       % T_eff integers from 1..T_eff
U_tilde = U(PER, :);                % bootstrap residuals

disp('POINT 2 DONE: bootstrap residuals created (first draw).');

%% ============================================================
%  POINT 3 — Simulate new series y_tilde (FIRST DRAW)
%% ============================================================

% Use original first p observations as starting values
y_tilde = zeros(T, N);
y_tilde(1:p, :) = y(1:p, :);        % initial conditions

for t = p+1:T
    % build [1, y_{t-1},...,y_{t-p}]
    x_t = 1;
    for j = 1:p
        x_t = [x_t, y_tilde(t-j, :)];
    end
    y_tilde(t, :) = x_t * B + U_tilde(t-p,:);   % simulated data
end

disp('POINT 3 DONE: simulated y_tilde for first bootstrap draw.');

%% ============================================================
%  POINT 4 — Estimate VAR on simulated series, compute IRFs (FIRST DRAW)
%% ============================================================

% Build Yb, Xb from y_tilde
Yb = y_tilde(p+1:end, :);
Xb = [];
for i = 1:p
    Xb = [Xb, y_tilde(p+1-i:end-i,:)];
end
Xb = [ones(T_eff,1), Xb];

% Re-estimate VAR on bootstrap sample
Bb = Xb \ Yb;
Ub = Yb - Xb * Bb;
SigmaUb = (Ub' * Ub)/(T_eff - k);

% Extract A_hat_b from Bb
A_hat_b = zeros(N, N*p);
for i = 1:p
    rows = 1 + (i-1)*N + (1:N);
    A_hat_b(:, (1+(i-1)*N):(i*N)) = Bb(rows,:)';
end

% Cholesky identification on bootstrap sample
P_b = chol(SigmaUb, 'lower');

% Companion form for bootstrap VAR
F_b = zeros(N*p);
F_b(1:N,:) = A_hat_b;
F_b(N+1:end, 1:N*(p-1)) = eye(N*(p-1));

IRF_b = zeros(H+1, N, N);
Fpow = eye(N*p);
for h = 0:H
    Theta_h = J * Fpow * J' * P_b;
    IRF_b(h+1,:,:) = Theta_h;
    Fpow = Fpow * F_b;
end

disp('POINT 4 DONE: IRFs from first bootstrap sample computed.');

%% ============================================================
%  POINT 5 — Repeat steps 2-4 K times, build bootstrap distribution
%% ============================================================

K = 1000;                         % number of bootstrap replications
IRF_boot = zeros(K, H+1, N, N);   % store full IRFs

disp('Running bootstrap replications...');

for k_boot = 1:K

    % --- Step 2: sample residuals with replacement ---
    PER = randi(T_eff, T_eff, 1);
    U_tilde = U(PER, :);

    % --- Step 3: simulate new data under estimated VAR ---
    y_tilde = zeros(T, N);
    y_tilde(1:p, :) = y(1:p, :);  % initial conditions

    for t = p+1:T
        x_t = 1;
        for j = 1:p
            x_t = [x_t, y_tilde(t-j,:)];
        end
        y_tilde(t,:) = x_t * B + U_tilde(t-p,:);
    end

    % --- Step 4: re-estimate VAR and compute structural IRFs ---
    Yb = y_tilde(p+1:end,:);
    Xb = [];
    for i = 1:p
        Xb = [Xb, y_tilde(p+1-i:end-i,:)];
    end
    Xb = [ones(T_eff,1), Xb];

    Bb = Xb \ Yb;
    Ub = Yb - Xb*Bb;
    SigmaUb = (Ub' * Ub)/(T_eff - k);

    % extract A_hat_b
    A_hat_b = zeros(N, N*p);
    for i = 1:p
        rows = 1 + (i-1)*N + (1:N);
        A_hat_b(:, (1+(i-1)*N):(i*N)) = Bb(rows,:)';
    end

    % companion and Cholesky for bootstrap VAR
    F_b = zeros(N*p);
    F_b(1:N,:) = A_hat_b;
    F_b(N+1:end,1:N*(p-1)) = eye(N*(p-1));

    P_b = chol(SigmaUb, 'lower');

    Fpow = eye(N*p);
    for h = 0:H
        IRF_boot(k_boot,h+1,:,:) = J * Fpow * J' * P_b;
        Fpow = Fpow * F_b;
    end
end

disp('POINT 5 DONE: bootstrap completed.');

%% ============================================================
%  COMPUTE 95% CONFIDENCE BANDS FOR MONETARY POLICY IRFs
%% ============================================================

% Baseline IRFs (original VAR) already in IRF_baseline_mp
IRF_original = IRF_baseline_mp;   % (H+1) x N

% Extract bootstrap IRFs for monetary policy shock (3rd shock)
IRF_boot_mp = squeeze(IRF_boot(:,:,:,mp_shock));  % K x (H+1) x N

ci_low  = zeros(H+1,N);
ci_high = zeros(H+1,N);

for i = 1:N
    tmp = squeeze(IRF_boot_mp(:,:,i)); % K x (H+1)
    ci_low(:,i)  = prctile(tmp,  2.5, 1)';
    ci_high(:,i) = prctile(tmp, 97.5, 1)';
end

%% Plot IRFs with 95% confidence intervals (for Overleaf figures)

figure;
for i = 1:N
    subplot(3,1,i);
    plot(0:H, IRF_original(:,i), 'k','LineWidth',2); hold on;
    plot(0:H, ci_low(:,i),'r--');
    plot(0:H, ci_high(:,i),'r--');
    yline(0, ':');
    title(['IRF of ', vars{i}, ' to monetary policy shock (with 95% CI)']);
    xlabel('Horizon (quarters)');
    grid on;
end

% --- export as PDF for LaTeX ---
filename = 'Figure_2.pdf';
exportgraphics(gcf, filename, 'ContentType','vector', 'BackgroundColor','white');

disp('ALL DONE: baseline VAR, structural IRFs, FEVD, and bootstrap CIs computed.');
%%
%%%%%% EXERCISE TWO %%%%%%
%% Import data and define VAR variables
fname = 'data_ps3.xlsx';
Tbl = readtable(fname,'Sheet','technology_shock','VariableNamingRule','preserve');
Tbl.Properties.VariableNames = {'date','outperh','hours'};

time = Tbl.date;
x    = Tbl.outperh;
h    = Tbl.hours;

% differencing
dx = diff(x);
dn = diff(h);
Y  = [dx, dn];
dtime = time(2:end);

%% VAR(p) on [dx dn] with constant
[Tobs, nvar] = size(Y);
p = 4;                     % quarterly data

%stacked regression form
T_eff = Tobs - p;
Yt = Y(p+1:end, :);
X  = ones(T_eff, 1);
for L = 1:p
    X = [X, Y(p+1-L:end-L, :)];
end

% OLS
B = X \ Yt;
U = Yt - X*B;
SigmaU = (U' * U) / (T_eff - size(X,2));

% parameters for later
c_hat   = B(1, :)';
A_hat   = B(2:end, :)';
A_cells = mat2cell(A_hat, nvar, nvar*ones(1,p));

resid_hat = U;
Y_init    = Y(1:p, :);

%% Long-run identification (Galí)
nvar = size(Y,2);
I    = eye(nvar);

% long-run multipliers on \Delta y
A_sum = zeros(nvar);
for j = 1:p
    A_sum = A_sum + A_cells{j};
end
C1 = inv(I - A_sum);                  % C(1) = (I - A1 - ... - Ap)^{-1}

% identification: Galì restriction "only tech has permanent effect on productivity"
% the long-run impact on levels must be lower-triangular:
L = chol(C1 * SigmaU * C1', 'lower');
B = C1 \ L; 

% reduced-form MA(\infty) for \Delta y
H = 12;                               % horizon quarterly
Phi = zeros(nvar,nvar,H+1);
Phi(:,:,1) = eye(nvar);
for h = 1:H
    S = zeros(nvar);
    for j = 1:min(h,p)
        S = S + A_cells{j} * Phi(:,:,h-j+1);
    end
    Phi(:,:,h+1) = S;
end

% structural IRFs for \Delta y
Theta = zeros(nvar,nvar,H+1);
for h = 0:H
    Theta(:,:,h+1) = Phi(:,:,h+1) * B;   % \Delta y responses
end
LevelTheta = cumsum(Theta, 3);

% for each shock k, GDP IRF = x_IRF + n_IRF
gdp_irf = squeeze(LevelTheta(1,:,:)+LevelTheta(2,:,:));

% extract series (columns = shocks ; col 1 = tech, col 2 = nontech):
tech = 1; ntech = 2;
prod_tech   = squeeze(LevelTheta(1,tech,:));
hours_tech  = squeeze(LevelTheta(2,tech,:));
prod_ntech  = squeeze(LevelTheta(1,ntech,:));
hours_ntech = squeeze(LevelTheta(2,ntech,:));
gdp_tech    = prod_tech  + hours_tech;
gdp_ntech   = prod_ntech + hours_ntech;

%% graphs
h = 0:H;
t_years = h/4;

lims = @(v) [min(v(:)), max(v(:))];

yl_prod_tech   = lims(prod_tech);
yl_prod_ntech  = lims(prod_ntech);
yl_gdp_tech    = lims(gdp_tech);
yl_gdp_ntech   = lims(gdp_ntech);
yl_hours_tech  = lims(hours_tech);
yl_hours_ntech = lims(hours_ntech);

figure('Name','IRFs (levels) - Baseline','Color','w');
tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

nexttile; plot(t_years, prod_tech,  'LineWidth',1.6); hold on; yline(0,'k:'); hold off;
title('Technology shock'); ylabel('Productivity (x)'); ylim(yl_prod_tech);

nexttile; plot(t_years, prod_ntech, 'LineWidth',1.6); hold on; yline(0,'k:'); hold off;
title('Non-technology shock'); ylabel('Productivity (x)'); ylim(yl_prod_ntech);

nexttile; plot(t_years, gdp_tech,   'LineWidth',1.6); hold on; yline(0,'k:'); hold off;
ylabel('GDP (x+n)'); ylim(yl_gdp_tech);

nexttile; plot(t_years, gdp_ntech,  'LineWidth',1.6); hold on; yline(0,'k:'); hold off;
ylabel('GDP (x+n)'); ylim(yl_gdp_ntech);

nexttile; plot(t_years, hours_tech, 'LineWidth',1.6); hold on; yline(0,'k:'); hold off;
ylabel('Hours (n)'); xlabel('Years'); ylim(yl_hours_tech);

nexttile; plot(t_years, hours_ntech,'LineWidth',1.6); hold on; yline(0,'k:'); hold off;
ylabel('Hours (n)'); xlabel('Years'); ylim(yl_hours_ntech);

sgtitle('IRFs (levels), 1 s.d. structural shocks responses');

outdir = 'Ex2';
if ~exist(outdir,'dir'), mkdir(outdir); end
exportgraphics(gcf, fullfile(outdir, 'Galì_IRFs_noCI.pdf'), ...
               'ContentType','vector','BackgroundColor','white');

%% bootstrap CI
K = 1000;
[Tobs, nvar] = size(Y);
T_eff = Tobs - p;
I = eye(nvar);

boot_prod_tech   = zeros(K, H+1);
boot_gdp_tech    = zeros(K, H+1);
boot_hours_tech  = zeros(K, H+1);
boot_prod_ntech  = zeros(K, H+1);
boot_gdp_ntech   = zeros(K, H+1);
boot_hours_ntech = zeros(K, H+1);

for k = 1:K
    % resample residual rows with replacement
    idx = randi(T_eff, T_eff, 1);           
    eps_tilde = resid_hat(idx, :);        

    % simulate new series using \hat c, \hat A and resampled eps
    Yb = zeros(Tobs, nvar);
    Yb(1:p,:) = Y_init;                       
    for t = p+1:Tobs
        yhat = c_hat';                       
        for j = 1:p
            yhat = yhat + Yb(t-j,:) * A_cells{j}';
        end
        Yb(t,:) = yhat + eps_tilde(t-p,:);    % bootstrapped innovation
    end

    % re-estimate RF VAR(p) with constant on Yb
    [Tloc, ~] = size(Yb);
    T_eff_loc = Tloc - p;
    Yt_loc = Yb(p+1:end,:);
    X_loc = ones(T_eff_loc,1);
    for L = 1:p
        X_loc = [X_loc, Yb(p+1-L:end-L,:)];
    end
    B_loc = X_loc \ Yt_loc;
    U_loc = Yt_loc - X_loc*B_loc;
    SigmaU_b = (U_loc' * U_loc) / (T_eff_loc - size(X_loc,2));
    c_hat_b  = B_loc(1, :)';
    A_hat_loc  = B_loc(2:end, :)';
    A_cells_b  = mat2cell(A_hat_loc, nvar, nvar*ones(1,p));

    % long-run: C(1) = (I - A1 - ... - Ap)^{-1}
    A_sum_b = zeros(nvar);
    for j = 1:p, A_sum_b = A_sum_b + A_cells_b{j}; end
    C1_b = (I - A_sum_b) \ I;

    % identification: lower-triangular long-run (Galí)
    L_b = chol(C1_b * SigmaU_b * C1_b','lower');
    B_b = C1_b \ L_b;

    % RF MA multipliers
    Phi_b = zeros(nvar,nvar,H+1); Phi_b(:,:,1) = I;
    for h = 1:H
        S = zeros(nvar);
        for j = 1:min(h,p), S = S + A_cells_b{j} * Phi_b(:,:,h-j+1); end
        Phi_b(:,:,h+1) = S;
    end
    % structural IRFs for differences
    Theta_b = zeros(nvar,nvar,H+1);
    for h = 0:H, Theta_b(:,:,h+1) = Phi_b(:,:,h+1) * B_b; end
    LevelTheta_b = cumsum(Theta_b, 3);

    % store series needed for bands (columns = shocks: 1 tech, 2 non-tech)
    prod_b_tech   = squeeze(LevelTheta_b(1,1,:))';
    hours_b_tech  = squeeze(LevelTheta_b(2,1,:))';
    prod_b_ntech  = squeeze(LevelTheta_b(1,2,:))';
    hours_b_ntech = squeeze(LevelTheta_b(2,2,:))';

    boot_prod_tech(k,:)   = prod_b_tech;
    boot_hours_tech(k,:)  = hours_b_tech;
    boot_gdp_tech(k,:)    = prod_b_tech + hours_b_tech;
    boot_prod_ntech(k,:)  = prod_b_ntech;
    boot_hours_ntech(k,:) = hours_b_ntech;
    boot_gdp_ntech(k,:)   = prod_b_ntech + hours_b_ntech;
end

% percentile bands (95%)
pct = [2.5 97.5];
PT = prctile(boot_prod_tech,   pct);  prod_tech_lo   = PT(1,:)'; prod_tech_hi   = PT(2,:)';
GT = prctile(boot_gdp_tech,    pct);  gdp_tech_lo    = GT(1,:)'; gdp_tech_hi    = GT(2,:)';
HT = prctile(boot_hours_tech,  pct);  hours_tech_lo  = HT(1,:)'; hours_tech_hi  = HT(2,:)';

PN = prctile(boot_prod_ntech,  pct);  prod_ntech_lo  = PN(1,:)'; prod_ntech_hi  = PN(2,:)';
GN = prctile(boot_gdp_ntech,   pct);  gdp_ntech_lo   = GN(1,:)'; gdp_ntech_hi   = GN(2,:)';
HN = prctile(boot_hours_ntech, pct);  hours_ntech_lo = HN(1,:)'; hours_ntech_hi = HN(2,:)';

%% graphs with bootstrap CI
h = 0:H;  t_years = h/4;
lims = @(v) [min(v(:)), max(v(:))];

yl_prod_tech   = lims([prod_tech,  prod_tech_lo,  prod_tech_hi]);
yl_prod_ntech  = lims([prod_ntech, prod_ntech_lo, prod_ntech_hi]);
yl_gdp_tech    = lims([gdp_tech,   gdp_tech_lo,   gdp_tech_hi]);
yl_gdp_ntech   = lims([gdp_ntech,  gdp_ntech_lo,  gdp_ntech_hi]);
yl_hours_tech  = lims([hours_tech, hours_tech_lo, hours_tech_hi]);
yl_hours_ntech = lims([hours_ntech,hours_ntech_lo,hours_ntech_hi]);

figure('Name','IRFs (levels) with 95% bootstrap bands','Color','w');
tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

% tech: prod
nexttile; 
plot(t_years, prod_tech, 'LineWidth',1.6); hold on;
plot(t_years, prod_tech_lo, 'k--'); plot(t_years, prod_tech_hi, 'k--');
yline(0,'k:'); hold off; ylim(yl_prod_tech);
title('Technology shock'); ylabel('Productivity (x)');

% non tech: prod
nexttile; 
plot(t_years, prod_ntech, 'LineWidth',1.6); hold on;
plot(t_years, prod_ntech_lo, 'k--'); plot(t_years, prod_ntech_hi, 'k--');
yline(0,'k:'); hold off; ylim(yl_prod_ntech);
title('Non-technology shock'); ylabel('Productivity (x)');

% tech: GDP
nexttile; 
plot(t_years, gdp_tech, 'LineWidth',1.6); hold on;
plot(t_years, gdp_tech_lo, 'k--'); plot(t_years, gdp_tech_hi, 'k--');
yline(0,'k:'); hold off; ylim(yl_gdp_tech);
ylabel('GDP (x+n)');

% non-tech: GDP
nexttile; 
plot(t_years, gdp_ntech, 'LineWidth',1.6); hold on;
plot(t_years, gdp_ntech_lo, 'k--'); plot(t_years, gdp_ntech_hi, 'k--');
yline(0,'k:'); hold off; ylim(yl_gdp_ntech);
ylabel('GDP (x+n)');

% tech: hours
nexttile; 
plot(t_years, hours_tech, 'LineWidth',1.6); hold on;
plot(t_years, hours_tech_lo, 'k--'); plot(t_years, hours_tech_hi, 'k--');
yline(0,'k:'); hold off; ylim(yl_hours_tech);
ylabel('Hours (n)'); xlabel('Years');

% non-tech: hours
nexttile; 
plot(t_years, hours_ntech, 'LineWidth',1.6); hold on;
plot(t_years, hours_ntech_lo, 'k--'); plot(t_years, hours_ntech_hi, 'k--');
yline(0,'k:'); hold off; ylim(yl_hours_ntech);
ylabel('Hours (n)'); xlabel('Years');

sgtitle('IRFs (levels): 1 s.d. shock responses with bootstrapped 95% CI');

outdir = 'Ex2';
if ~exist(outdir,'dir'), mkdir(outdir); end
exportgraphics(gcf, fullfile(outdir, 'Galì_IRFs_bootCI.pdf'), ...
               'ContentType','vector','BackgroundColor','white');