function [fbest, xbest, history] = alg_MCS(Problem, opts, bounds)
 tic

%clear; clear mex; 
% clear global; 
format compact; format long

% if not(isfolder('mcs'))
%     unzip('mcs.zip'); 
% end

parts = strsplit(pwd, filesep); parts{end + 1} = 'Algorithms'; parts{end + 1} = 'mcs'; 
parent_path = strjoin(parts(1:end), filesep); addpath(parent_path); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% problem definition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define objective function
%
% Each function value f=f(x) is obtained by MCS using a call 
% f=feval(fcn,data,x)
% where data is an arbitrary data object passed to mcs. If you have 
% several data items, put them into a cell array or structure array
% called data.
% If there are no data, use fcn='feval' and the function name as data.
%
fcn = 'feval';
data = Problem.f;
% data = fun;
% data = 'gklsMatlab';	
% select test function from Jones et al. test set
		% bra  n=2   Branin
		% cam  n=2   six-hump camel
		% gpr  n=2   Goldstein-Price   
		% shu  n=2   Shubert
		% hm3  n=3   Hartman 3
		% s10  n=4   Shekel 10
		% sh5  n=4   Shekel 5
		% sh7  n=4   Shekel 7
		% hm6  n=6   Hartman 6
% fcn = data;

% define bounds on variables (+-inf allowed)
%
% u: column vector of lower bounds
% v: column vector of upper bounds
% u(k)<v(k) is required
%
% [u,v,fglob] = defaults(data); 	% returns bounds used by Jones et al.
% 				% and known global optimum
% dimension=length(u)	;	% show dimension
% known_global_opt_value=fglob;	% show known global minimum value


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% change MCS settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% flexible use version - all parameters may be modified

% Return the problem information.

nvars = opts.dimension;


% bounds
u = bounds(:, 1); % left bound
v = bounds(:, 2); % right bound

fglob  = opts.globalmin;

perror = fglob + 1e-8;
% if fglob ~= 0
%     perror = fglob + abs(fglob)/100;
% else
%     perror = 0.01;
% end

% 
% define amount of output printed
prt = 1;	% print level 
		% prt = 0: no output
		% prt = 1: # sweep, minimal nonempty level, # f-calls,
		%          best point and function value
		% prt > 1: in addition levels and function values of
		%          boxes containing the known global minimizers
		%          of a test function

% define global strategy 
%
% smax governs the relative amount of global versus local search. By
% increasing smax, more weight is given to global search.
% 
% Increasing or decreasing stop(1) increases or decreases the amount 
% of work allowed before concluding that nothing more is gained; 
% the default choice is quite conservative, and may try to locate
% a better point long after the global minimum has been found.
% stop(1)=5 works for many easier problems, too, with much fewer
% wasted function values.
% 
% Increasing nf allows to spend more time on boxes that have a chance 
% on their level to contain better points. This may be important for
% hard problems, or for problems with very wide bounds.
% 
% But in each case, it is unclear in advance what change would be most 
% beneficial to a particular problem. 
% We had very mixed experience; if you have many similar problems to 
% solve, the best thing to do is to experiment with a few problems to 
% find the best values, and to use these on the others. 
%
n = length(u);		% problem dimension
smax = 10*n+10;		% number of levels used
nf = opts.maxevals; 		% limit on number of f-calls
stop(1) = 3*n;		% = m, integer defining stopping test
stop(2) = -inf;		% = freach, function value to reach
			% if m>0, run until m sweeps without progress
			% if m=0, run until fbest<=freach
			% (or about nf function calls were used)

% if 0, 	% known global optimum, for tests only
	% then the entries of stop have a different meaning
  stop(1) = 0;	% run until this relative error is achieved
  stop(2) = fglob + opts.tol;	% known global optimum value
  stop(3) = 0;	% stopping tolerance for tiny fglob
  
% end;

% define initialization strategy
%
% for wide boxes, it is advisable (and for unbounded search regions
% strongly advisable) to define a customized initialization list
% that contains for each coordinate at least three reasonable values.
% Without such an initialization list, mcs makes default assumptions
% that roughly amount to estimating reasonable magnitudes from the 
% bounds and in case iinit=1 from assuming order unity if this is 
% within the bounds. 
%
% for a self-defined initialization list, the user should
% write an m-script file init0.m containing a matrix x0 with n
% rows and at least 3 columns and two n-vectors l and L 
% the ith column of x0 contains the initialization list
% values for the ith coordinate, their number is L(i), and
% x0(i,l(i)) is the ith coordinate of the initial point

iinit = 0;	% 0: simple initialization list
		%    (default for finite bounds;
		%     do not use this for very wide bounds)
		% 1: safeguarded initialization list
		%    (default for unbounded search regions)
		% 2: (5*u+v)/6, (u+v)/2, (u+5*v)/6
		% 3: initialization list with line searches
		% else: self-defined initialization list 
		%       stored in file init0.m

% parameters for local search
%
% A tiny gamma (default) gives a quite accurate but in higher 
% dimensions slow local search. Increasing gamma leads to less work 
% per local search but a less accurately localized minimizer
% 
% If it is known that the Hessian is sparse, providing the sparsity 
% pattern saves many function evaluations since the corresponding
% entries in the Hessian need not be estimated. The default pattern
% is a full matrix.
% 
local = 100;		% local = 0: no local search
			% else: maximal number of steps in local search
gamma = eps;		% acceptable relative accuracy for local search
hess = ones(n,n);	% sparsity pattern of Hessian



% defaults are not being used, use the full calling sequence
% (including at least the modified arguments)
%%%%%%%%%%%%%%%%%%%%%%% full MCS call %%%%%%%%%%%%%%%%%%%%%%
[xbest,fbest,xmin,fmi,ncall,ncloc,~,history] =...
  mcs(fcn,data,u,v,prt,smax,nf,stop,iinit,local,gamma,hess);

% xbest	  		% best point found
% fbest     	% best function value
% fglob			% global minimum (known for test functions)
% ncall	  		% number of function values used
% ncloc	  		% number of function values in local searches
% fmi
% xmin	  		% columns are points in 'shopping basket'
			% may be good alternative local minima
% fmi	  		% function values in 'shopping basket'
% nbasket = length(fmi) 	% number of points in 'shopping basket'

s = size(history, 1) + 1;
history(s, 1) = s;
history(s, 2) = ncall;
history(s, 3) = fbest;
history(s, 4) = toc;
return