function  [Fmin, Xmin, h] = alg_purecmaes(ObjFun, opts, Bounds)
% CMA-ES: Evolution Strategy with Covariance Matrix Adaptation for 
% nonlinear function minimization.
%
% This code is "an excerpt" from cmaes.m and implements the key parts of 
% the algorithm. It is intendend to be used for READING and UNDERSTANDING 
% the basic flow and all details of the CMA-ES *algorithm*. To run 
% "serious" simulations better use the cmaes.m code: it is longer, but 
% offers restarts, far better termination options, and, in particular, 
% supposedly quite useful output.
%
% Author: Nikolaus Hansen, 2003-09.
% e-mail: hansen[at]lri.fr
%
% License: This code is released into the public domain (that is, you may 
% use and modify it however you like).
%
% URL: http://www.lri.fr/~hansen/purecmaes.m
% References: See end of file. Last change: September, 05, 2023
%
% --------------------  Initialization --------------------------------
% User defined input parameters (need to be edited)
tic                                                                        % For timeing
strfitnessfct = ObjFun.f;                                                  % name of objective/fitness function
N = opts.dimension;                                                        % number of objective variables/problem dimension
xmean = rand(N, 1);                                                        % objective variables initial point
sigma = 0.5;                                                               % coordinate wise standard deviation (step size)
stopfitness = opts.tol;                                                    % stop if fitness < stopfitness (minimization)
stopeval = opts.maxevals;                                                  % stop after stopeval number of function evaluations

% Strategy parameter setting: Selection
lambda = 4 + floor(3*log(N));                                              % population size, offspring number
lambda = 18*N;
mu = lambda/2;                                                             % number of parents/points for recombination
weights = log(mu + 1/2) - log(1:mu)';                                      % muXone array for weighted recombination
mu = floor(mu);
weights = weights/sum(weights);                                            % normalize recombination weights array
mueff = sum(weights)^2/sum(weights.^2);                                    % variance-effectiveness of sum w_i x_i

% Strategy parameter setting: Adaptation
cc = (4 + mueff/N)/(N + 4 + 2*mueff/N);                                    % time constant for cumulation for C
cs = (mueff + 2)/(N + mueff + 5);                                          % t-const for cumulation for sigma control
c1 = 2/((N + 1.3)^2 + mueff);                                              % learning rate for rank-one update of C
cmu = min(1 - c1, 2*(mueff - 2 + 1/mueff)/((N + 2)^2 + mueff));            % and for rank-mu update
damps = 1 + 2*max(0, sqrt((mueff - 1)/(N + 1)) - 1) + cs;                  % damping for sigma

% Initialize dynamic (internal) strategy parameters and constants
pc = zeros(N, 1); ps = zeros(N, 1);                                        % evolution paths for C and sigma
B = eye(N, N);                                                             % B defines the coordinate system
D = ones(N, 1);                                                            % diagonal D defines the scaling
C = B*diag(D.^2)*B';                                                       % covariance matrix C
invsqrtC = B*diag(D.^-1)*B';                                               % C^-1/2
eigeneval = 0;                                                             % track update of B and D
chiN = N^0.5*(1 - 1/(4*N) + 1/(21*N^2));                                   % expectation of
it = 0; counteval = 0; h = [0, 0, inf, 0]; Fmin = inf; Xmin = [];

% -------------------- Generation Loop --------------------------------
while counteval < stopeval
    % Generate and evaluate lambda offspring
    for k=1:lambda
        arx(:,k) = xmean + sigma*B*(D .* randn(N,1));  %#ok<*AGROW>        % m + sig*Normal(0,C)
        arx(:,k) = zrcad(arx(:,k)', Bounds(:, 1)', Bounds(:, 2)')';
        arfitness(k) = feval(strfitnessfct, arx(:,k));                     % objective function call
        counteval = counteval + 1;
    end

    % Sort by fitness and compute weighted mean into xmean
    [arfitness, arindex] = sort(arfitness);                                % minimization
    
    if any(~isnan(arx(:, arindex(1)))) && Fmin > arfitness(1)
        Fmin = arfitness(1);
        Xmin = arx(:, arindex(1));
    end
    xold = xmean;
    xmean = arx(:,arindex(1:mu))*weights;                                  % recombination, new mean value

    % Cumulation: Update evolution paths
    ps = (1 - cs)*ps + sqrt(cs*(2 - cs)*mueff)*invsqrtC*(xmean - xold)/sigma;
    hsig = sum(ps.^2)/(1 - (1 - cs)^(2*counteval/lambda))/N < 2 + 4/(N + 1);
    pc = (1 - cc)*pc + hsig*sqrt(cc*(2 - cc)*mueff)*(xmean - xold)/sigma;

    % Adapt covariance matrix C
    artmp = (1/sigma)*(arx(:, arindex(1:mu)) - repmat(xold, 1, mu));       % mu difference vectors
    C = (1 - c1 - cmu)*C ...                                               % regard old matrix
        + c1*(pc*pc' ...                                                   % plus rank one update
        + (1 - hsig)*cc*(2 - cc)*C) ...                                    % minor correction if hsig==0
        + cmu*artmp*diag(weights)*artmp';                                  % plus rank mu update

    % Adapt step size sigma
    sigma = sigma*exp((cs/damps)*(norm(ps)/chiN - 1));

    % Update B and D from C
    if counteval - eigeneval > lambda/(c1+cmu)/N/10  % to achieve O(N^2)
        eigeneval = counteval;
        C = triu(C) + triu(C, 1)';                                         % enforce symmetry
        [B, D] = eig(C);                                                   % eigen decomposition, B==normalized eigenvectors
        D = sqrt(diag(D));                                                 % D contains standard deviations now
        invsqrtC = B*diag(D.^-1)*B';
    end

    it = it + 1;
    fprintf('Iter: %4i   f_min: %15.10f    time(s): %10.05f    fn evals: %8i\n', it, Fmin, toc, counteval);
    if arfitness(1) < h(end, 3)
        h(end + 1, :) = [it, counteval, Fmin, toc]; 
    end
    if abs(opts.globalmin - Fmin) <= stopfitness || counteval >= opts.maxevals || toc > 600 
        h(end + 1, :) = [it, counteval, Fmin, toc]; 
        break
    end
end                                                                        % while, end generation loop
end
                                                            
% ------------- Mirroring, Perturbation y into <a,b> ----------------------
function result = zrcad(y, a, b)
y = real(y); zrc = find(y < a | y > b); y_in = y; y_orig = y; %#ok<*NASGU>
b_mod = b; b(b == 0) = 1; a_mod = a; a(a == 0) = -1;
y_in(y_in > b_mod) = mod(y_in(y_in > b_mod), 1.5*b_mod(y_in > b_mod)) + 1.5*b_mod(y_in > b_mod);
y_in(y_in < a_mod) = mod(y_in(y_in < a_mod), 1.5*a_mod(y_in < a_mod)) + 1.5*a_mod(y_in < a_mod);
y = y_in;
for i = zrc
    while (y(i) < a(i) || y(i) > b(i))
        if y(i) > b(i)
            y(i) = 2*b(i) - y(i);
        elseif y(i) < a(i)
            y(i) = 2*a(i) - y(i);
        end
    end
end
result = y;
end

%% REFERENCES
%
% Hansen, N. and S. Kern (2004). Evaluating the CMA Evolution Strategy on 
% Multimodal Test Functions. Eighth International Conference on Parallel 
% Problem Solving from Nature PPSN VIII, Proceedings, pp. 282-291, Berlin: Springer.
% (http://www.bionik.tu-berlin.de/user/niko/ppsn2004hansenkern.pdf)
%
% Further references:
% Hansen, N. and A. Ostermeier (2001). Completely Derandomized Self-Adaptation 
% in Evolution Strategies. Evolutionary Computation, 9(2), pp. 159-195.
% (http://www.bionik.tu-berlin.de/user/niko/cmaartic.pdf).
%
% Hansen, N., S.D. Mueller and P. Koumoutsakos (2003). Reducing the Time 
% Complexity of the Derandomized Evolution Strategy with Covariance Matrix
% Adaptation (CMA-ES). Evolutionary Computation, 11(1).  
% (http://mitpress.mit.edu/journals/pdf/evco_11_1_1_0.pdf).