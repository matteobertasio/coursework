%% Cake Eating with Taste Shocks
clearvars; clear global; close all;

%% set parameterss

% model parameters
param.betta = 0.96;         % discount factor
param.gamma   = 1.0;         
param.rho   = 0.8;          % persistence of the shock            
param.a0   = 1.0;

% solution method parameters
param.nS    = 3;          % # of nodes for shock
param.nA    = 1000;        % # of nodes for assets
param.amin = 0.0;
param.amax = 1.0;

maxit  = 20000;           % maximum number of iterations
tol = 1e-8;               % convergence criterion

%% build the grid 
% for households assets
param.grid.a = linspace(param.amin,param.amax,param.nA);
param.grid.z = [1 - param.gamma/2, 1, 1 + param.gamma/2]';

% for the shocks
param.Pi  = [param.rho 1-param.rho 0;
             (1-param.rho)/2 param.rho (1-param.rho)/2;
             0 1-param.rho param.rho];


%% Solve the model with ITERATION
%param.Howard=100; % number of iterations using Howard improvement (1 means no improvement)
V_guess = zeros(param.nS,param.nA);
hh_func = @HouseholdProblem_Cake;

for iter=1:maxit
    hh_pol = hh_func(V_guess, param);
    
    % check convergence every 10 iterations
    if mod(iter,10)==0
        crit = max(abs(hh_pol.V(:) - V_guess(:)));
        fprintf('HH problem: iter %d, resid %3.4g \n',iter, crit);
        if max(crit) < tol
            fprintf('HH problem solved in %d iter.\n',iter);
            break
        end
    end
    
    %update guess
    V_guess = hh_pol.V;
end
assert(iter<maxit, 'SS of household problem not found: maxiter exceeded') 

%% plot policy functions

figure(1);
subplot(1,2,1)
hold on
plot(param.grid.a,hh_pol.a');
plot(param.grid.a,param.grid.a,'--k');
hold off
grid on
xlabel('Initial Asset (a)');
ylabel('End-of-Period Assets (a^{\prime})');
xlim([param.grid.a(1), param.grid.a(end)]);
ylim([param.grid.a(1), param.grid.a(end)]);

subplot(1,2,2)
plot(param.grid.a,hh_pol.c');
grid on
xlabel('Initial Asset (a)');
ylabel('Consumption (c)');
axis tight
xlim([param.grid.a(1), param.grid.a(end)]);

%% plot value function
figure(2);
plot(param.grid.a, hh_pol.V(1,:), '-', ...
     param.grid.a, hh_pol.V(2,:), '--', ...
     param.grid.a, hh_pol.V(3,:), ':', 'LineWidth', 1.3);
grid on;
xlabel('Initial Asset (a)');
ylabel('Value Function V(z,a)');
legend(sprintf('z = %.2f',param.grid.z(1)), ...
       sprintf('z = %.2f',param.grid.z(2)), ...
       sprintf('z = %.2f',param.grid.z(3)), ...
       'Location','best');
title('Value Function');

%% AUXILIARY FUNCTION

function hh_pol = HouseholdProblem_Cake(V_next, param)
% function calculating one backward iteration in the household problem
% using the basic discretization and Howard improvement
%
%   Output:  
%   ---------
%   pol: structure with fields
%       V  : Value Function today (array nS*nA)
%       a  : optimal choice of assets (array nS*nA)
%       c  : optimal choice of consumption (array nS*nA)
%
%   Inputs:
%   ---------
%   V_next : value function tomorrow (array nS*nA)
%   param   : strucuture with fields
%       Pi      : Markov transition matrix for idios. shock (matrix nS*nS)
%       grid.a  : nodes for assets  (array nA)
%       grid.z  : nodes for shock (array nS)
%       betta   : discount factor (scalar)
%       CRR     : coefficient of risk aversion (scalar)

V = zeros(param.nS, param.nA);
c = zeros(param.nS, param.nA);
a = zeros(param.nS, param.nA);
optutil = zeros(param.nS, param.nA);
indx = zeros(param.nS, param.nA);

% EV today with P(z,z')
EV_next = param.betta * (param.Pi * V_next);      % (nS x nA)

%maximization (discrete choice)
for iasset = 1:param.nA
    choicesAprime = param.grid.a;
    cChoices = param.grid.a(iasset) - choicesAprime; %possible consumption choices
    
    Ulog = -inf(1,param.nA);
    pos  = (cChoices > 0);
    Ulog(pos) = log(cChoices(pos));
    Util = param.grid.z * Ulog;   

    [V(:,iasset),indx(:,iasset)] = max(Util + EV_next, [], 2);
    a(:,iasset) = param.grid.a(indx(:,iasset));
    c(:,iasset) = param.grid.a(iasset) - a(:,iasset);
    optutil(:,iasset) = param.grid.z .* utilfunc(c(:,iasset),param);
end

posInf = (V==-Inf); V(posInf) = -1e3; % replace -Inf with very small (finite number). 

% Howard improvement (optional)
for h=1:99
    Vnew = zeros(param.nS, param.nA);   
    for iasset=1:param.nA
        Vnew(:,iasset) = optutil(:,iasset) + param.betta*diag(param.Pi*V(:,indx(:,iasset)));
    end
    posInf = (Vnew==-Inf); Vnew(posInf) = -1e3; % replace -Inf with very small (finite number). 
                                      % Needed here becuase 0 income and 0 assets implies 0 consumption 
    V = Vnew;
end

% pack outputs
hh_pol = struct('c', c, 'a', a);
hh_pol.V = V;
end

function util = utilfunc(c, param)
    util = log(max(c,0));   % log(c) se c>0, altrimenti log(0)= -Inf
end



