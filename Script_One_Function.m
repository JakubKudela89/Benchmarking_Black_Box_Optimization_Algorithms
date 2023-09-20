%% a simple script for calling the methods on an example problem
% for the cobyla and bobyqa algorithms, the "PDFO: Powell's Derivative-Free Optimization solvers" toolbox needs to be installed:
% https://www.mathworks.com/matlabcentral/fileexchange/75195-pdfo-powell-s-derivative-free-optimization-solvers
% beware that these two algorithms also require a MATLAB-supported FORTRAN compiler
% https://www.mathworks.com/support/requirements/supported-compilers.html
% the SID_PMS, LGO, OQNLP, glcCluster, and HALO require special installations and/or licenses

clear;clc;close all;
addpath('Algorithms');
val = dir("Algorithms");

% get names of all methods in 'Algorithms' folder
alg_names = {};
for i=1:length(val)
    if ~val(i).isdir
        alg_names{end+1} = val(i).name(1:end-2);
    end
end

% set problem dimension
dimension = 10;

% get info about the problem
getInfo = objective_function();
xL = getInfo.xl(dimension);
xU = getInfo.xu(dimension);
Fmin = getInfo.fmin(dimension);
Xmin = getInfo.xmin(dimension);

% set parameters for the algorihms
opts = struct(); Problem = struct();

vec = rand(dimension,1); %random shift

Problem.f = @(x) objective_function(x+vec);
opts.dimension  = dimension;
opts.globalmin  = Fmin;
opts.globalxmin = Xmin;
opts.maxevals   = 5000;    % max number of function calls
opts.maxits     = 10000;    % max number of iterations
opts.tolabs     = 1e-8;     % absolute tolerance threshold
opts.tol        = 1e-8;
opts.showits    = 1;        % show iterations
opts.testflag   = 1;
opts.population = 100;      % set population size (some methods have inner mechanisms for this)
Bounds = [xL, xU];

% call single method
[fbest, xatmin, history] = alg_DE(Problem, opts, Bounds);

%call all methods in 'Algorithms' folder
% for i=1:length(alg_names)
%     temp_str_res = alg_names{i}; temp_str_res = temp_str_res(5:end); temp_str_res = strcat("Results_", temp_str_res);
%     eval(strcat(temp_str_res,' = [];'));
%     temp_str = strcat('[fbest, xatmin, history] = ', alg_names{i},'(Problem, opts, Bounds);');
%     eval(temp_str);
%     temp_str = strcat(temp_str_res,'.fbest = fbest'); eval(temp_str);
%     temp_str = strcat(temp_str_res,'.xatmin = xatmin;'); eval(temp_str);
%     temp_str = strcat(temp_str_res,'.history = history;'); eval(temp_str);
% end