function [minima, xatmin, history] = alg_DIRMIN(Problem, opts, bounds)
%--------------------------------------------------------------------------
% Function   : DIRMIN
% Author 1   : Linas Stripinis          (linas.stripinis@mif.vu.lt)
% Author 2   : Remigijus Paualvicius    (remigijus.paulavicius@mif.vu.lt)
% Created on : 09/29/2019
% Purpose    : DIRECT optimization algorithm for box constraints.
%--------------------------------------------------------------------------
% [minima, xatmin, history] = dDirmin(Problem, opts, bounds)
%       d         - dynamic memory management in data structure
%       DIR       - DIRECT(DIvide a hyper-RECTangle) algorithm
%       MIN       - hybridizing DIRECT with the local search
%
% Input parameters:
%       Problem - Structure containing problem
%                 Problem.f       = Objective function handle
%
%       opts    - MATLAB structure which contains options.
%                 opts.maxevals  = max. number of function evals
%                 opts.maxits    = max. number of iterations
%                 opts.maxdeep   = max. number of rect. divisions
%                 opts.testflag  = 1 if globalmin known, 0 otherwise
%                 opts.globalmin = globalmin (if known)
%                 opts.globalxmin = globalxmin (if known)
%                 opts.dimension = problem dimension
%                 opts.showits   = 1 print iteration status
%                 opts.ep        = global/local weight parameter
%                 opts.tol       = tolerance for termination if
%                                  testflag = 1
%
%       bounds  - (n x 2) matrix of bound constraints LB <= x <= UB
%                 The first column is the LB bounds and the second
%                 column contains the UB bounds
%
% Output parameters:
%       minima  -  best minimum value which was founded
%
%       xatmin  - coordinate of minimal value
%
%       history - (iterations x 4) matrix of iteration history
%                 First column coresponds iterations
%                 Second column coresponds number of objective function
%                 evaluations
%                 Third column coresponds minima value of objecgtive
%                 function which was founded at iterations
%                 Third column coresponds time cost of the algorithm
%
% Original DIRECT implementation taken from:
%--------------------------------------------------------------------------
% D.R. Jones, C.D. Perttunen, and B.E. Stuckman. "Lipschitzian
% Optimization Without the Lipschitz Constant". Journal of Optimization
% Theory and Application, 79(1):157-181, (1993). DOI 10.1007/BF00941892
%
% Liuzzi, G., Lucidi, S., Piccialli, V.: A DIRECT-based approach exploiting
% local minimizations for the solution of large-scale global optimization 
% problems. Comput. Optim. Appl. 45(2), 353–375 (2010).
% https://doi.org/10.1007/s10589-008-9217-2
%--------------------------------------------------------------------------
if nargin == 2, bounds = []; end
if nargin == 1, bounds = []; opts = []; end

% Get options
[SS, VAL] = Options(opts, nargout, Problem, bounds);

% Alocate sets and create initial variables
[MSS, CE, third, VAL] = Alocate(SS, VAL);

% Initialization step
[MV, MSS, CE, VAL, minval, xatmin] = Initialization(Problem,...
    MSS, CE, VAL, SS);

while VAL.perror > SS.TOL                 % Main loop
    % Selection of potential optimal hyper-rectangles step
    POH = Find_po(MSS, CE, SS, minval, MV(1, :));

    % Run local searches
    VAL = minlocal(MSS, VAL, Problem, POH, SS);
    
    % Subdivide potential optimalhyper-rectangles
    [VAL, MSS, CE, MV, minval, xatmin] = Calulcs(VAL, Problem,...
        MSS, CE, third, POH);
    
    % Update minima and check stopping conditions
    [VAL, SS, minval, xatmin] = Arewedone(MSS, minval, VAL, SS, xatmin);
end                                       % End of while

% Return value
minima      = minval;
if SS.G_nargout == 3
    history   = VAL.history(1:VAL.itctr, 1:4);
end
%--------------------------------------------------------------------------
return

%--------------------------------------------------------------------------
% AUXILIARY FUNCTION BLOCK
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Function  : SS
% Purpose   : Get options from inputs
%--------------------------------------------------------------------------
function [OPTI, VAL] = Options(opts, narg, Problem, bounds)
%--------------------------------------------------------------------------
% Determine option values
if nargin < 3 && isempty(opts)
    opts = [];
end

getopts(opts, 'maxits', 1000,'maxevals', 100000, 'maxdeep', 1000,...
    'testflag', 0, 'tol', 0.01, 'showits', 1, 'dimension', 1, 'ep',...
    1e-4, 'globalmin', 0, 'globalxmin', 0);

if isempty(bounds)
% Return the problem information.
    getInfo = feval(Problem.f);
    
% dimension
    if getInfo.nx == 0
        VAL.n = dimension;
    else
        VAL.n = getInfo.nx;
    end
    
    VAL.a = getInfo.xl(VAL.n); % left bound
    VAL.b = getInfo.xu(VAL.n); % right bound
    
    if testflag == 1
% minimum value of function
        OPTI.globalMIN  = getInfo.fmin(VAL.n);
% minimum point of function
    end
else
    VAL.a = bounds(:, 1);               % left bound
    VAL.b = bounds(:, 2);               % right bound
    VAL.n = size(bounds, 1);            % dimension
    if testflag == 1
        OPTI.globalMIN = globalmin;     % minimum value of function
    end
end
VAL.MAXevals = maxevals;
OPTI.G_nargout = narg;     % output arguments
OPTI.MAXits    = maxits;   % Fmax of iterations
OPTI.MAXevals  = maxevals; % Fmax # of function evaluations
OPTI.MAXdeep   = maxdeep;  % Fmax number of side divisions
OPTI.TESTflag  = testflag; % terminate if global minima is known
OPTI.showITS   = showits;  % print iteration stat
OPTI.TOL       = tol;      % allowable relative error if f_reach is set
OPTI.ep        = ep;       % global/local weight parameter
%--------------------------------------------------------------------------
return

%--------------------------------------------------------------------------
% Function  : Alocate
% Purpose   : Create necessary data accessible to all workers
%--------------------------------------------------------------------------
function [MSS, CE, third, VAL] = Alocate(SS, VAL)
%--------------------------------------------------------------------------
% Create Initial values
tic                                         % Mesure time
[VAL.time, VAL.Local, VAL.nLs] = deal(0);   % initial time
VAL.fcount     = 1;                         % first fcnc counter
VAL.itctr      = 1;                         % initial iteration
VAL.perror     = 2*SS.TOL;                  % initial perror
CE             = zeros(1, VAL.n*SS.MAXdeep);% collum counter

% alociate MAIN sets
MSS = struct('F', zeros(1), 'E', zeros(1), 'C', zeros(VAL.n, 1),...
    'L', zeros(VAL.n, 1));

third           = zeros(1, SS.MAXdeep);     % delta values
third(1)        = 1/3;                      % first delta
for i = 2:SS.MAXdeep                        % all delta
    third(i)    = (1/3)*third(i - 1);
end
if SS.G_nargout == 3
    VAL.history = zeros(SS.MAXits, 4);       % allocating history
end
%--------------------------------------------------------------------------
return

%--------------------------------------------------------------------------
% Function  : Initialization
% Purpose   : Initialization of the DIRECT
%--------------------------------------------------------------------------
function [MV, MSS, CE, VAL, Fmin, Xmin] = Initialization(Problem,...
    MSS, CE, VAL, SS)
%--------------------------------------------------------------------------
% Create Initial values
MV(1:2, 1) = 1;                                     % first fake value
MSS(1).L(:, 1) = zeros(VAL.n, 1);                 	% Lengths
MSS(1).Diam = 1/2*norm((1/3*(ones(VAL.n, 1))).^MSS(1).L(:, 1));
MSS(1).C(:, 1) = ones(VAL.n, 1)/2;               	% Center point
MSS(1).E(1) = 1;                                    % Index
CE(1) = 1;
[VAL.Xmin, Xmin] = deal(abs(VAL.b - VAL.a).*(MSS(1).C(:, 1)) + VAL.a);
[Fmin, MSS(1).F(1), VAL.Flocal, MV(1)] = deal(feval(Problem.f, Xmin));     

% Check stop condition if global minima is known
if SS.TESTflag  == 1
    VAL.perror = abs(SS.globalMIN - Fmin);
    % if SS.globalMIN ~= 0
    %     VAL.perror = 100*(Fmin - SS.globalMIN)/abs(SS.globalMIN);
    % else
    %     VAL.perror = 100*Fmin;
    % end
else
    VAL.perror   = 2;
end

% Store History
if SS.G_nargout == 3
    VAL.history(VAL.itctr, 1) = 0;
    VAL.history(VAL.itctr, 2) = VAL.fcount;
    VAL.history(VAL.itctr, 3) = Fmin;
    VAL.history(VAL.itctr, 4) = VAL.time;
end
%--------------------------------------------------------------------------
return

%--------------------------------------------------------------------------
% Function  : Calulcs
% Purpose   : Calculate; new points, function values, lengths and indexes
%--------------------------------------------------------------------------
function [VAL, MSS, CE, MV, minval, xatmin] = Calulcs(VAL,...
    Problem, MSS, CE, third, POH)
%--------------------------------------------------------------------------
[MSS, CE, VAL] = STORAGE(Problem, MSS, CE, third, VAL, POH);

for i = size(POH, 2):-1:1                
    if ~isempty(POH{i})         
        if (CE(i) - size(POH{i}, 2)) == 0
            if find(CE ~= 0, 1, 'first') == i
                [MSS(i).E, MSS(i).L, MSS(i).F, MSS(i).C] = deal([]);
            end
        else
            C = 1:CE(i); C(POH{i}) = [];
            pp = min(POH{i}):length(C);
            MSS(i).E(pp) = MSS(i).E(C(pp));
            MSS(i).L(:, pp) = MSS(i).L(:, C(pp));
            MSS(i).F(pp) = MSS(i).F(C(pp));
            MSS(i).C(:, pp) = MSS(i).C(:, C(pp));
        end
        CE(i) = CE(i) - size(POH{i}, 2);
    end
end

% Find minima values
[MV, minval, xatmin] = Find_min(CE, MSS);
%--------------------------------------------------------------------------
return

%--------------------------------------------------------------------------
% Function  : STORAGE
% Purpose   : Store information wrom workers
%--------------------------------------------------------------------------
function [MSS, CE, VAL] = STORAGE(Problem, MSS, CE, third, VAL, PH)
%--------------------------------------------------------------------------
for i = 1:size(MSS, 2)
    if ~isempty(PH{i})
        for j = 1:size(PH{i}, 2)
            if VAL.fcount + VAL.Local > VAL.MAXevals
                PH{i}(j) = [];
                break;
            end
            [DD, TT, VAL] = CALCULS_WORKER(Problem, i, MSS, third,...
                VAL, PH{i}(j));
            for h = 1:TT
                II = i + h;      
                if II > size(CE, 2)
                  CE(II) = 0;    
                end
                if CE(II) == 0
                    MSS(II).Diam = 1/2*...
                        norm((1/3*(ones(VAL.n, 1))).^DD.lL(:, h));
                    [MSS(II).L, MSS(II).C] = deal(zeros(VAL.n, 100));
                    [MSS(II).F, MSS(II).E] = deal(zeros(1, 100));
                end
                IL = CE(II) + 1;             
                IR = CE(II) + 2;             
                CE(II) = IR;                    
                if CE(II) > size(MSS(II).F, 2)
                    MSS(II).L = [MSS(II).L, zeros(VAL.n, 100)];
                    MSS(II).F = [MSS(II).F, zeros(1, 100)];
                    MSS(II).E = [MSS(II).E, zeros(1, 100)];
                    MSS(II).C = [MSS(II).C, zeros(VAL.n, 100)];
                end
                MSS(II).F(IL) = DD.L(DD.lsx(h, 1));   
                MSS(II).F(IR) = DD.R(DD.lsx(h, 1));  
                MSS(II).E(IL) = DD.eL(h);            
                MSS(II).E(IR) = DD.eR(h);            
                MSS(II).L(:, IL) = DD.lL(:, h);        
                MSS(II).L(:, IR) = DD.lL(:, h);           
                MSS(II).C(:, IL) = DD.cL(:, DD.lsx(h, 1));  
                MSS(II).C(:, IR) = DD.cR(:, DD.lsx(h, 1));  
            end
            II = i + TT;            
            CE(II) = CE(II) + 1;        
            MSS(II).F(CE(II)) = DD.O;               
            MSS(II).E(CE(II)) = DD.eO;             
            MSS(II).C(:, CE(II)) = DD.cO;           
            MSS(II).L(:, CE(II)) = DD.LO; 
        end
    end
end
%--------------------------------------------------------------------------
return

%--------------------------------------------------------------------------
% Function  : Calulcs
% Purpose   : Calculate; new points, function values, lengths and indexes
%--------------------------------------------------------------------------
function [A, S, VAL] = CALCULS_WORKER(Const, i, MSS, third, VAL, PH)
%--------------------------------------------------------------------------
% Create and allocating empty sets
A.cO = MSS(i).C(:, PH);
A.LO = MSS(i).L(:, PH);
A.eO = MSS(i).E(PH);
A.O = MSS(i).F(PH);
DIVIS = find(A.LO == min(A.LO));
DELTA = third(min(A.LO) + 1);
S = length(DIVIS);
[A.L, A.R, A.eL, A.eR] = deal(zeros(1, S));
A.lL = A.LO*ones(1, S);
A.cL = A.cO*ones(1, S); A.cR = A.cL;
for g = 1:S
    A.cL(DIVIS(g), g) = A.cL(DIVIS(g), g) - DELTA;
    om_point = abs(VAL.b - VAL.a).*A.cL(:,g) + VAL.a;
    A.L(g) = feval(Const.f, om_point);
    A.cR(DIVIS(g), g) = A.cR(DIVIS(g), g) + DELTA;
    om_point = abs(VAL.b - VAL.a).*A.cR(:,g) + VAL.a;
    A.R(g) = feval(Const.f, om_point);
end
[~, A.lsx] = sort([min(A.L, A.R)' DIVIS], 1);
for g = 1:S
    A.lL(DIVIS(A.lsx(1:g, 1)), g) = A.lL(DIVIS(A.lsx(1:g, 1)), g) + 1;
    A.eL(g) = VAL.fcount + 1;
    A.eR(g) = VAL.fcount + 2;
    VAL.fcount = A.eR(g);
end
A.LO = A.lL(:, S);
%--------------------------------------------------------------------------
return

%--------------------------------------------------------------------------
% Function  : Arewedone
% Purpose   : Update minima value and check stopoing conditions
%--------------------------------------------------------------------------
function [VAL, SS, Fm, xatmin] = Arewedone(MSS, Fm, VAL, SS, xatmin)
%--------------------------------------------------------------------------
if Fm < VAL.Flocal
    xatmin = abs(VAL.b - VAL.a).*xatmin + VAL.a;
else
    xatmin = VAL.Xmin;
    Fm = VAL.Flocal;
end
    
VAL.time = toc;

% Check for stop condition
if SS.showITS == 1                % Show iteration stats
    fprintf(...
    'Iter: %4i  f_min: %15.10f    time(s): %10.05f  local searches: %4i    fn evals: %8i\n',...
        VAL.itctr, Fm, VAL.time, VAL.nLs, VAL.fcount + VAL.Local);
end

if SS.TESTflag == 1
    % Calculate error if globalmin known
    VAL.perror = abs(SS.globalMIN - Fm);
    % if SS.globalMIN ~= 0
    %     VAL.perror = 100*(Fm - SS.globalMIN)/abs(SS.globalMIN);
    % else
    %     VAL.perror = 100*Fm;
    % end
    if VAL.perror < SS.TOL
        fprintf('Minima was found with Tolerance: %4i', SS.TOL);
        VAL.perror = -1;
    end
else
    VAL.perror = 10;
end

% Have we exceeded the maxits?
if VAL.itctr >= SS.MAXits
    disp('Exceeded max iterations. Increase maxits'); VAL.perror = -1;
end

% Have we exceeded the maxevals?
if VAL.fcount + VAL.Local > SS.MAXevals
    disp('Exceeded max fcn evals. Increase maxevals'); VAL.perror = -1;
end

% Have we exceeded max deep?
if SS.MAXdeep <= max(MSS(end).L(:, 1)) + 1
    disp('Exceeded Max depth. Increse maxdeep'); VAL.perror = -1;
end

if VAL.time >= 600
    disp('Exceeded Max depth. Increse maxdeep'); VAL.perror = -1;
end

% Store History
if SS.G_nargout == 3
    VAL.history(VAL.itctr, 1) = VAL.itctr;
    VAL.history(VAL.itctr, 2) = VAL.fcount + VAL.Local;
    VAL.history(VAL.itctr, 3) = Fm;
    VAL.history(VAL.itctr, 4) = VAL.time;
end

% Update iteration number
if VAL.perror > SS.TOL
    VAL.itctr = VAL.itctr + 1;
end       
%--------------------------------------------------------------------------
return

%--------------------------------------------------------------------------
% Function  : minlocal
% Purpose   : perform local searches
%--------------------------------------------------------------------------
function VAL = minlocal(MSS, VAL, PP, PH, SS)
%--------------------------------------------------------------------------
if SS.TESTflag == 1
options = optimoptions('fmincon', 'Display', 'none','Algorithm','sqp',...
    'MaxFunctionEvaluations', VAL.n*400, 'StepTolerance', 1e-15,...
    'FiniteDifferenceType', 'central', 'ObjectiveLimit', SS.globalMIN + SS.TOL);
else
    options = optimoptions('fmincon', 'Display', 'none');
end

for i = 1:size(MSS, 2)
    if VAL.fcount + VAL.Local > VAL.MAXevals
        break;
    end
    if abs(SS.globalMIN - VAL.Flocal) <= SS.TOL
        break;
    end
    if ~isempty(PH{i})
        for j = 1:size(PH{i}, 2)
            pp = abs(VAL.b - VAL.a).*MSS(i).C(:, PH{i}(j)) + VAL.a;
            [Xmin_local, Fmin_local, ~, output] = fmincon(PP.f, pp,...
                [], [], [], [], VAL.a, VAL.b, [], options);
            VAL.nLs = VAL.nLs + 1;
            VAL.Local = VAL.Local + output.funcCount;
            if Fmin_local < VAL.Flocal
                VAL.Flocal = Fmin_local;
                VAL.Xmin = Xmin_local;
            end
        end
    end
end
%--------------------------------------------------------------------------
return

%--------------------------------------------------------------------------
% Function  : Find_poh
% Purpose   : Return list of PO hyperrectangles
%--------------------------------------------------------------------------
function final_pos = Find_po(MSS, CE, SS, minval, MV)
% Find all rects on hub
index    = size(MSS, 2);
nonsize  = [MSS.Diam];
hulls    = cell(3, index);

for i = 1:index
    if CE(i) ~= 0
        ties     = find(abs(MSS(i).F(1:CE(i)) - MV(i)) <= 1e-13);
        if ~isempty(ties)
            hulls{1, i} = ties;
            hulls{2, i} = nonsize(i)*ones(1, size(ties, 2));
            hulls{3, i} = MSS(i).F(ties);
        end
    end
end

fc   = cell2mat(hulls(3, :));
szes = cell2mat(hulls(2, :));

% Compute lb and ub for rects on hub
ubound = calc_ubound(fc, hulls, szes, index);
lbound = calc_lbound(fc, hulls, szes, index);

% Find indeces of hull who satisfy
[maybe_po, final_pos] = deal(cell(1, index));
for i = 1:index
    if ~isempty(hulls{1, i})
        maybe_po{i} = find(lbound{i} - ubound{i} <= 0);
    end
end

if minval ~= 0
    for i = 1:index
        if ~isempty(maybe_po{i})
            po = (minval - hulls{3, i}(maybe_po{i}))./abs(minval) +...
                hulls{2, i}(maybe_po{i}).*ubound{i}...
                (maybe_po{i})./abs(minval) >= SS.ep;
            final_pos{i} = hulls{1, i}(po);
        end
    end
else
    for i = 1:index
        if ~isempty(maybe_po{i})
            po = (hulls{3, i}(maybe_po{i})) -...
                (hulls{2, i}(maybe_po{i})).*ubound{i}(maybe_po{i}) <= 0;
            final_pos{i} = hulls{1, i}(po);
        end
    end
end
if isempty(cell2mat(final_pos))
    ii = (find(CE ~= 0, 1, 'first'));
    final_pos{ii} = hulls{1, ii};
end
return

%--------------------------------------------------------------------------
% Function   :  calc_ubound
% Purpose    :  calculate the ubound used in determing potentially
%               optimal hrectangles
%--------------------------------------------------------------------------
function ub = calc_ubound(fc, hull, szes, index)
hull_lengths = szes;
ub           = cell(1, index);

for i = 1:index
    if ~isempty(hull{1, i})
        for j = 1:size(hull{1, i}, 2)
            tmp_rects = find(hull_lengths > hull{2, i}(j));
            if ~isempty(tmp_rects)
                tmp_f    = fc(tmp_rects);
                tmp_szes = szes(tmp_rects);
                tmp_ubs  = (tmp_f - hull{3, i}(j))./...
                    (tmp_szes - hull{2, i}(j));
                ub{i}(j) = min(tmp_ubs);
            else
                ub{i}(j) = 1.976e1004;
            end
        end
    end
end
return

%--------------------------------------------------------------------------
% Function   :  calc_lbound
% Purpose    :  calculate the lbound used in determing potentially
%               optimal hrectangles
%--------------------------------------------------------------------------
function lb = calc_lbound(fc, hull, szes, index)
hull_lengths = szes;
lb           = cell(1, index);

for i = 1:index
    if ~isempty(hull{1, i})
        for j = 1:size(hull{1, i}, 2)
            tmp_rects = find(hull_lengths < hull{2, i}(j));
            if ~isempty(tmp_rects)
                tmp_f    = fc(tmp_rects);
                tmp_szes = szes(tmp_rects);
                tmp_ubs  = (hull{3, i}(j) - tmp_f)./...
                    (hull{2, i}(j) - tmp_szes);
                lb{i}(j) = max(tmp_ubs);
            else
                lb{i}(j) = -1.976e1004;
            end
        end
    end
end
return

%--------------------------------------------------------------------------
% Function   :  Find_min
% Purpose    :  Find min value
%--------------------------------------------------------------------------
function [MV, minval, xatmin] = Find_min(CE, MSS)
%--------------------------------------------------------------------------
MV = nan(2, size(MSS, 2));

for i = 1:size(MSS, 2)
    if CE(i) ~= 0
        [MV(1, i), MV(2, i)] = min(MSS(i).F(1:CE(i)));
    end
end

[minval, Least] = min(MV(1, :));
xatmin = MSS(Least).C(:, MV(2, Least));
%--------------------------------------------------------------------------
return

%--------------------------------------------------------------------------
% GETOPTS Returns options values in an options structure
%--------------------------------------------------------------------------
function varargout = getopts(options, varargin)
K = fix(nargin/2);
if nargin/2 == K
    error('fields and default values must come in pairs')
end
if isa(options,'struct')
    optstruct = 1;
else
    optstruct = 0;
end
varargout = cell(K, 1);
k = 0; ii = 1;
for i = 1:K
    if optstruct && isfield(options, varargin{ii})
        assignin('caller', varargin{ii}, options.(varargin{ii}));
        k = k + 1;
    else
        assignin('caller', varargin{ii}, varargin{ii + 1});
    end
    ii = ii + 2;
end
return
%--------------------------------------------------------------------------
% END of BLOCK
%--------------------------------------------------------------------------