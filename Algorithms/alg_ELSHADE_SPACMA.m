function [minima, xatmin, history] = alg_ELSHADE_SPACMA(Problem, opts, bounds)
% original citation: 
% A. A. Hadi, A. W. Mohamed, and K. M. Jambi, “Single-objective real-
% parameter optimization: Enhanced lshade-spacma algorithm," Heuristics
% for optimization and learning, pp. 103–121, 2021.
% modified by Jakub Kudela
% =========================================================================
tic;
minima = [];
pop_size=opts.population;
D=opts.dimension;
max_nfes = opts.maxevals;
fhd = Problem.f;

lu = [bounds(:,1)'; bounds(:,2)'];

NP=18*D;
max_NP = NP;
min_NP = 4.0;
bsf_fit_var = Inf; xatmin = zeros(D,1); history = []; nfes = 0;

while bsf_fit_var == Inf && nfes < max_nfes
    Pop = repmat(lu(1, :), NP, 1) + rand(NP, D) .* (repmat(lu(2, :) - lu(1, :), NP, 1));
    Fit = zeros(NP,1);
    for it1=1:NP
        Fit(it1) = fhd(Pop(it1,:)');
    end
    %Fit= cec17_func(Pop',func_num);
    %Fit=Fit';
    [bsf_fit_var,I_best]=min(Fit);
    X_best=Pop(I_best,:);
    nfes = nfes + NP;
end
iter = 1;
history(iter,1) = 1;
history(iter,2) = nfes;
history(iter,3) = bsf_fit_var;
history(iter,4) = toc;
if opts.showits
    fprintf("Iter:%5i\tf_min:%15.10f\ttime(s):%10.5f\tfn evals:%9i\n",[history(iter,1),history(iter,3),history(iter,4),history(iter,2)]);
    %Iter:   43   f_min:  -78.9844713606    time(s):    0.34048    fn evals:     1723
end    

Par=[];
Par.p_best_rate = 0.3;
Par.p_best_rate_max = Par.p_best_rate;
Par.p_best_rate_min = 0.15;

Par.All_fit=[];
Par.Record_FEs=max_nfes.*[0.01, 0.02, 0.03, 0.05, 0.1:0.1:1];
Par.max_nfes=max_nfes;
Par.First_calss_percentage=0.5;
Par.Hybridization_flag=1;

Par.memory_size=5;
Par.memory_sf = 0.5 .* ones(Par.memory_size, 1);
Par.memory_cr = 0.5 .* ones(Par.memory_size, 1);
Par.memory_pos = 1;

arc_rate = 1.4;
archive=[];
archive.NP = arc_rate * NP; % the maximum size of the archive
archive.Pop = []; % the solutions stored in te archive;
archive.funvalues = []; % the function value of the archived solutions
Par.memory_1st_class_percentage = Par.First_calss_percentage.* ones(Par.memory_size, 1); % Class#1 probability for Hybridization

[CMAES_Par]= CMAES_Init(NP,D);
Par.CMAES_Par=CMAES_Par;

while nfes<max_nfes
    
    Par.GenRatio=nfes/max_nfes;
    
    [Pop,Fit,archive,Par]= LSHADE_SPACMA(Pop,Fit,lu,fhd,archive,Par,nfes);
    nfes = nfes+NP;
    if(~isempty(Par.Record_FEs))
        if (nfes>=Par.Record_FEs(1))
            Par.All_fit=[Par.All_fit;min(Fit)];
            Par.Record_FEs(1)=[];
        end
    end

    if nfes >= max_nfes;
        [Best_fit,I_best]= min(Fit);
        X_best=Pop(I_best,:);
        All_fit=Par.All_fit;
        break;
    end
    
    
    [Pop,Fit,archive,Par]= EADE_SPA(Pop,Fit,lu,fhd,archive,Par,nfes);
    nfes = nfes+NP;
    if(~isempty(Par.Record_FEs))
        if (nfes>=Par.Record_FEs(1))
            Par.All_fit=[Par.All_fit;min(Fit)];
            Par.Record_FEs(1)=[];
        end
    end

    if nfes >= max_nfes;
        [Best_fit,I_best]= min(Fit);
        X_best=Pop(I_best,:);
        All_fit=Par.All_fit;
        break;
    end
   
    %% Population size & p reduction 
    plan_NP = round((((min_NP - max_NP) / (max_nfes)) * nfes) + max_NP);
    Par.p_best_rate = (((Par.p_best_rate_min - Par.p_best_rate_max) / (max_nfes)) * nfes) + Par.p_best_rate_max;

    if NP > plan_NP
        reduction_ind_num = NP - plan_NP;
        if NP - reduction_ind_num <  min_NP; reduction_ind_num = NP - min_NP;end
        NP = NP - reduction_ind_num;
        for r = 1 : reduction_ind_num
            [valBest, indBest] = sort(Fit, 'ascend');
            worst_ind = indBest(end);
            Pop(worst_ind,:) = [];
            Fit(worst_ind,:) = [];
        end
        archive.NP = NP;
        if size(archive.Pop, 1) > archive.NP
            rndpos = randperm(size(archive.Pop, 1));
            rndpos = rndpos(1 : archive.NP);
            archive.Pop = archive.Pop(rndpos, :);
        end

    end
    iter = iter + 1;
    if min(Fit)< bsf_fit_var
        bsf_fit_var = min(Fit);
    end
    history(iter,1) = iter;
    history(iter,2) = nfes;
    history(iter,3) = bsf_fit_var;
    history(iter,4) = toc;        
    if opts.showits && ~mod(iter,50) 
        fprintf("Iter:%5i\tf_min:%15.10f\ttime(s):%10.5f\tfn evals:%9i\n",[history(iter,1),history(iter,3),history(iter,4),history(iter,2)]);
        %Iter:   43   f_min:  -78.9844713606    time(s):    0.34048    fn evals:     1723
    end
    if bsf_fit_var-opts.globalmin < opts.tolabs || history(iter,4) > 600
        break
    end
end

[Best_fit,I_best]= min(Fit);
X_best=Pop(I_best,:);
All_fit=Par.All_fit;

xatmin = X_best';
minima = Best_fit;

end



function [CMAES_Par]= CMAES_Init(NP,D)
%% Initialize CMAES parameters

CMAES_Par.sigma = 0.5;          % coordinate wise standard deviation (step size)
CMAES_Par.xmean = rand(D,1);    % objective variables initial point
mu = NP/2;               % number of parents/points for recombination
weights = log(mu+1/2)-log(1:mu)'; % muXone array for weighted recombination
weights = weights/sum(weights);     % normalize recombination weights array
mueff=sum(weights)^2/sum(weights.^2); % variance-effectiveness of sum w_i x_i

% Strategy parameter setting: Adaptation
CMAES_Par.cc = (4 + mueff/D) / (D+4 + 2*mueff/D); % time constant for cumulation for C
CMAES_Par.cs = (mueff+2) / (D+mueff+5);  % t-const for cumulation for sigma control
CMAES_Par.c1 = 2 / ((D+1.3)^2+mueff);    % learning rate for rank-one update of C
CMAES_Par.cmu = min(1-CMAES_Par.c1, 2 * (mueff-2+1/mueff) / ((D+2)^2+mueff));  % and for rank-mu update
CMAES_Par.damps = 1 + 2*max(0, sqrt((mueff-1)/(D+1))-1) + CMAES_Par.cs; % damping for sigma usually close to 1

% Initialize dynamic (internal) strategy parameters and constants
CMAES_Par.pc = zeros(D,1);
CMAES_Par.ps = zeros(D,1);   % evolution paths for C and sigma
CMAES_Par.B = eye(D,D);                       % B defines the coordinate system
CMAES_Par.diagD = ones(D,1);                      % diagonal D defines the scaling
CMAES_Par.C = CMAES_Par.B * diag(CMAES_Par.diagD.^2) * CMAES_Par.B';            % covariance matrix C
CMAES_Par.invsqrtC = CMAES_Par.B * diag(CMAES_Par.diagD.^-1) * CMAES_Par.B';    % C^-1/2
CMAES_Par.eigeneval = 0;                      % track update of B and D
CMAES_Par.chiN=D^0.5*(1-1/(4*D)+1/(21*D^2));  % expectation of
end


function [Pop,Fit,archive,Par]= LSHADE_SPACMA(Pop,Fit,lu,fhd,archive,Par,C_nfes)

format long;

L_Rate= 0.80;
[NP,D]=size(Pop);
X = zeros(NP,D); % trial vector

p_best_rate=Par.p_best_rate;
memory_size=Par.memory_size;
memory_sf=Par.memory_sf;
memory_cr=Par.memory_cr;
memory_pos=Par.memory_pos;

CMAES_Par= Par.CMAES_Par;
sigma=CMAES_Par.sigma;
xmean=CMAES_Par.xmean;
B=CMAES_Par.B;
diagD=CMAES_Par.diagD;


mem_rand_index = ceil(memory_size * rand(NP, 1));
mu_sf = memory_sf(mem_rand_index);
mu_cr = memory_cr(mem_rand_index);
mem_rand_ratio = rand(NP, 1);

[~, sorted_index] = sort(Fit, 'ascend');

%% for generating crossover rate
cr = normrnd(mu_cr, 0.1);
term_pos = mu_cr == -1;
cr(term_pos) = 0;
cr = min(cr, 1);
cr = max(cr, 0);

%% for generating scaling factor
if(C_nfes <= Par.max_nfes/2)
    sf=0.45+.1*rand(NP, 1);
    pos = find(sf <= 0);
    while ~ isempty(pos)
        sf(pos)=0.45+0.1*rand(length(pos), 1);
        pos = find(sf <= 0);
    end
else
    sf = mu_sf + 0.1 * tan(pi * (rand(NP, 1) - 0.5));
    pos = find(sf <= 0);
    while ~ isempty(pos)
        sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
        pos = find(sf <= 0);
    end
end
sf = min(sf, 1);

%% for generating Hybridization Class probability
mut_prop=(Par.memory_1st_class_percentage(mem_rand_index)>=mem_rand_ratio);
if(Par.Hybridization_flag==0)
    mut_prop=or(mut_prop,~mut_prop);%All will be in class#1
end

r0 = [1 : NP];
if(size(archive.Pop,1)~=0)
    Arc_pop=archive.Pop;
    popAll = [Pop; Arc_pop];
else
    popAll = Pop;
end
[r1, r2] = gnR1R2(NP, size(popAll, 1), r0);

pNP = max(round(p_best_rate * NP), 2); %% choose at least two best solutions
randindex = ceil(rand(1, NP) .* pNP); %% select from [1, 2, 3, ..., pNP]
randindex = max(1, randindex); %% to avoid the problem that rand = 0 and thus ceil(rand) = 0
pbest = Pop(sorted_index(randindex), :); %% randomly choose one of the top 100p% solutions

if sum(mut_prop)>0
    X(mut_prop,:) = Pop(mut_prop,:) + sf(mut_prop, ones(1, D)) .* (pbest(mut_prop,:) - Pop(mut_prop,:) + Pop(r1(mut_prop), :) - popAll(r2(mut_prop), :));
end

mut_prop=~mut_prop;

if sum(mut_prop)>0
    temp=[];
    for k=1:sum(mut_prop)
        temp(:,k) = xmean + sigma * B * (diagD .* randn(D,1)); % m + sig * Normal(0,C)
    end
    X(mut_prop,:) = temp';
end

mut_prop=~mut_prop;

if(~isreal(X))
    Par.Hybridization_flag=0;
    Par.memory_size=memory_size;
    Par.memory_sf=memory_sf;
    Par.memory_cr=memory_cr;
    Par.memory_pos=memory_pos;
    Par.CMAES_Par= CMAES_Par;
    return;
end

X = boundConstraint(X, Pop, lu);

mask = rand(NP, D) > cr(:, ones(1, D)); %mask is used to indicate which elements of ui comes from the parent
Rnd=ceil(D* rand(NP, 1)); %choose one position where the element of X doesn't come from the parent
jrand = sub2ind([NP D], (1:NP)', Rnd);
mask(jrand)=false;
X(mask) = Pop(mask);

[temp1,~] = size(X);Child_Fit = zeros(temp1,1);
for it1=1:temp1
    Child_Fit(it1) = fhd(X(it1,:)');
end
%Child_Fit= cec17_func(X',func_num);
%Child_Fit=Child_Fit';

%% Update Archive, Pop, CR, and F
Fit_imp_inf = (Child_Fit<=Fit);
goodCR = cr(Fit_imp_inf);
goodF = sf(Fit_imp_inf);
dif = abs(Fit - Child_Fit);
dif_val = dif(Fit_imp_inf);
dif_val_Class_1 = dif(and(Fit_imp_inf,mut_prop));
dif_val_Class_2 = dif(and(Fit_imp_inf,~mut_prop));

archive = updateArchive(archive, Pop(Fit_imp_inf, :), Fit(Fit_imp_inf));

num_success_params = numel(goodF);
if num_success_params > 0
    sum_dif = sum(dif_val);
    dif_val = dif_val / sum_dif;
    
    %% for updating the memory of scaling factor
    memory_sf(memory_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);
    
    %% for updating the memory of crossover rate
    if max(goodCR) == 0 || memory_cr(memory_pos)  == -1
        memory_cr(memory_pos)  = -1;
    else
        memory_cr(memory_pos) = (dif_val' * (goodCR .^ 2)) / (dif_val' * goodCR);
    end
    
    if(Par.Hybridization_flag==1)
        Par.memory_1st_class_percentage(memory_pos) = Par.memory_1st_class_percentage(memory_pos)*L_Rate+ (1-L_Rate)*(sum(dif_val_Class_1) / (sum(dif_val_Class_1) + sum(dif_val_Class_2)));
        Par.memory_1st_class_percentage(memory_pos) = min(Par.memory_1st_class_percentage(memory_pos),0.8);
        Par.memory_1st_class_percentage(memory_pos) = max(Par.memory_1st_class_percentage(memory_pos),0.2);
    end
    
    memory_pos = memory_pos + 1;
    if memory_pos > memory_size;  memory_pos = 1; end
end

Pop(Fit_imp_inf,:) = X(Fit_imp_inf,:); % replace current by trial
Fit(Fit_imp_inf) = Child_Fit(Fit_imp_inf) ;

if(Par.Hybridization_flag==1)
    [CMAES_Par,flag]= CMAES_update(C_nfes,Fit,CMAES_Par,Pop);
    if(flag==0)
        Par.Hybridization_flag=1;
    end
end

Par.memory_size=memory_size;
Par.memory_sf=memory_sf;
Par.memory_cr=memory_cr;
Par.memory_pos=memory_pos;
Par.CMAES_Par= CMAES_Par;

end

function [r1, r2] = gnR1R2(NP1, NP2, r0)

% gnA1A2 generate two column vectors r1 and r2 of size NP1 & NP2, respectively
%    r1's elements are choosen from {1, 2, ..., NP1} & r1(i) ~= r0(i)
%    r2's elements are choosen from {1, 2, ..., NP2} & r2(i) ~= r1(i) & r2(i) ~= r0(i)
%
% Call:
%    [r1 r2 ...] = gnA1A2(NP1)   % r0 is set to be (1:NP1)'
%    [r1 r2 ...] = gnA1A2(NP1, r0) % r0 should be of length NP1
%
% Version: 2.1  Date: 2008/07/01
% Written by Jingqiao Zhang (jingqiao@gmail.com)

NP0 = length(r0);

r1 = floor(rand(1, NP0) * NP1) + 1;
%for i = 1 : inf
for i = 1 : 99999999
    pos = (r1 == r0);
    if sum(pos) == 0
        break;
    else % regenerate r1 if it is equal to r0
        r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
    end
    if i > 1000, % this has never happened so far
        error('Can not genrate r1 in 1000 iterations');
    end
end

r2 = floor(rand(1, NP0) * NP2) + 1;
%for i = 1 : inf
for i = 1 : 99999999
    pos = ((r2 == r1) | (r2 == r0));
    if sum(pos)==0
        break;
    else % regenerate r2 if it is equal to r0 or r1
        r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
    end
    if i > 1000, % this has never happened so far
        error('Can not genrate r2 in 1000 iterations');
    end
end
end

function vi = boundConstraint (vi, pop, lu)

% if the boundary constraint is violated, set the value to be the middle
% of the previous value and the bound
%
% Version: 1.1   Date: 11/20/2007
% Written by Jingqiao Zhang, jingqiao@gmail.com

[NP, D] = size(pop);  % the population size and the problem's dimension

%% check the lower bound
xl = repmat(lu(1, :), NP, 1);
pos = vi < xl;
vi(pos) = (pop(pos) + xl(pos)) / 2;

%% check the upper bound
xu = repmat(lu(2, :), NP, 1);
pos = vi > xu;
vi(pos) = (pop(pos) + xu(pos)) / 2;
end

function archive = updateArchive(archive, pop, funvalue)
% Update the archive with input solutions
%   Step 1: Add new solution to the archive
%   Step 2: Remove duplicate elements
%   Step 3: If necessary, randomly remove some solutions to maintain the archive size
%
% Version: 1.1   Date: 2008/04/02
% Written by Jingqiao Zhang (jingqiao@gmail.com)

if archive.NP == 0, return; end

if size(pop, 1) ~= size(funvalue,1), error('check it'); end

% Method 2: Remove duplicate elements
popAll = [archive.Pop; pop ];
funvalues = [archive.funvalues; funvalue ];
[~, IX]= unique(popAll, 'rows');
if length(IX) < size(popAll, 1) % There exist some duplicate solutions
  popAll = popAll(IX, :);
  funvalues = funvalues(IX, :);
end

if size(popAll, 1) <= archive.NP   % add all new individuals
  archive.Pop = popAll;
  archive.funvalues = funvalues;
else                % randomly remove some solutions
  rndpos = randperm(size(popAll, 1)); % equivelent to "randperm";
  temp_NP=archive.NP;
  temp_NP=floor(temp_NP);
  rndpos = rndpos(1 : temp_NP);
  
  archive.Pop = popAll  (rndpos, :);
  archive.funvalues = funvalues(rndpos, :);
end

end

function [CMAES_Par,flag]= CMAES_update(nfes,fitness,CMAES_Par,Pop)

try
    %% update CMAES parameters
    flag=1;
    [NP,D]=size(Pop);
    
    sigma=CMAES_Par.sigma;
    xmean=CMAES_Par.xmean;
    cc=CMAES_Par.cc;
    cs=CMAES_Par.cs;
    c1=CMAES_Par.c1;
    cmu=CMAES_Par.cmu;
    damps=CMAES_Par.damps;
    pc=CMAES_Par.pc;
    ps=CMAES_Par.ps;
    B=CMAES_Par.B;
    diagD=CMAES_Par.diagD;
    C=CMAES_Par.C;
    invsqrtC=CMAES_Par.invsqrtC;
    eigeneval=CMAES_Par.eigeneval;
    chiN=CMAES_Par.chiN;
    
    mu = NP/2;               % number of parents/points for recombination
    weights = log(mu+1/2)-log(1:mu)'; % muXone array for weighted recombination
    weights = weights/sum(weights);     % normalize recombination weights array
    mueff=sum(weights)^2/sum(weights.^2); % variance-effectiveness of sum w_i x_i
    
    % Sort by fitness and compute weighted mean into xmean
    [~, popindex] = sort(fitness);  % minimization
    xold = xmean;
    mu=floor(mu);
    xmean = Pop(popindex(1:mu),:)' * weights;  % recombination, new mean value
    
    % Cumulation: Update evolution paths
    ps = (1-cs) * ps ...
        + sqrt(cs*(2-cs)*mueff) * invsqrtC * (xmean-xold) / sigma;
    hsig = sum(ps.^2)/(1-(1-cs)^(2*nfes/NP))/D < 2 + 4/(D+1);
    pc = (1-cc) * pc ...
        + hsig * sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma;
    
    % Adapt covariance matrix C
    mu=floor(mu);
    artmp = (1/sigma) * (Pop(popindex(1:mu),:)' - repmat(xold,1,mu));  % mu difference vectors
    C = (1-c1-cmu) * C ...                   % regard old matrix
        + c1 * (pc * pc' ...                % plus rank one update
        + (1-hsig) * cc*(2-cc) * C) ... % minor correction if hsig==0
        + cmu * artmp * diag(weights) * artmp'; % plus rank mu update
    
    % Adapt step size sigma
    sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1));
    
    % Update B and D from C
    if nfes - eigeneval > NP/(c1+cmu)/D/10  % to achieve O(D^2)
        eigeneval = nfes;
        C = triu(C) + triu(C,1)'; % enforce symmetry
        if(sum(sum(isnan(C)))>0 || sum(sum(~isfinite(C)))>0 || ~isreal(C))
            flag=0;
        end
        [B,diagD] = eig(C);           % eigen decomposition, B==normalized eigenvectors
        diagD = sqrt(diag(diagD));        % D contains standard deviations now
        invsqrtC = B * diag(diagD.^-1) * B';
    end
    
    
    CMAES_Par.sigma=sigma;
    CMAES_Par.xmean=xmean;
    CMAES_Par.cc=cc;
    CMAES_Par.cs=cs;
    CMAES_Par.c1=c1;
    CMAES_Par.cmu=cmu;
    CMAES_Par.damps=damps;
    CMAES_Par.pc=pc;
    CMAES_Par.ps=ps;
    CMAES_Par.B=B;
    CMAES_Par.diagD=diagD;
    CMAES_Par.C=C;
    CMAES_Par.invsqrtC=invsqrtC;
    CMAES_Par.eigeneval=eigeneval;
    CMAES_Par.chiN=chiN;
catch
    flag=0;
end


end


function [Pop,Fit,archive,Par]= EADE_SPA(Pop,Fit,lu,fhd,archive,Par,C_nfes)

[NP,D]=size(Pop);

memory_size=Par.memory_size;
memory_sf=Par.memory_sf;
memory_cr=Par.memory_cr;
memory_pos=Par.memory_pos;


X = zeros(NP,D); % trial vector

mem_rand_index = ceil(memory_size * rand(NP, 1));
mu_sf = memory_sf(mem_rand_index);
mu_cr = memory_cr(mem_rand_index);

%% for generating crossover rate
cr = normrnd(mu_cr, 0.1);
term_pos = mu_cr == -1;
cr(term_pos) = 0;
cr = min(cr, 1);
cr = max(cr, 0);

%% for generating scaling factor
if(C_nfes <= Par.max_nfes/2)
    sf=0.45+.1*rand(NP, 1);
    pos = find(sf <= 0);
    while ~ isempty(pos)
        sf(pos)=0.45+0.1*rand(length(pos), 1);
        pos = find(sf <= 0);
    end
else
    sf = mu_sf + 0.1 * tan(pi * (rand(NP, 1) - 0.5));
    pos = find(sf <= 0);
    while ~ isempty(pos)
        sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
        pos = find(sf <= 0);
    end
end

sf = min(sf, 1);
F=rand(NP,1);
r=genR_EADE(Fit);
X=Pop(r(:,4),:) + F(:, ones(1, D)).*(Pop(r(:,2),:)- Pop(r(:,3),:));
X = boundConstraint(X, Pop, lu);

mask = rand(NP, D) > cr(:, ones(1, D)); %mask is used to indicate which elements of ui comes from the parent
Rnd=ceil(D* rand(NP, 1)); %choose one position where the element of X doesn't come from the parent
jrand = sub2ind([NP D], (1:NP)', Rnd);
mask(jrand)=false;
X(mask) = Pop(mask);

[temp1,~] = size(X);Child_Fit = zeros(temp1,1);
for it1=1:temp1
    Child_Fit(it1) = fhd(X(it1,:)');
end
% Child_Fit= cec17_func(X',func_num);
% Child_Fit=Child_Fit';

%% Update Archive, Pop, CR, and F
Fit_imp_inf = (Child_Fit<=Fit);
goodCR = cr(Fit_imp_inf);
goodF = sf(Fit_imp_inf);
dif = abs(Fit - Child_Fit);
dif_val = dif(Fit_imp_inf);

archive = updateArchive(archive, Pop(Fit_imp_inf, :), Fit(Fit_imp_inf));

num_success_params = numel(goodF);
if num_success_params > 0
    sum_dif = sum(dif_val);
    dif_val = dif_val / sum_dif;
    
    %% for updating the memory of scaling factor
    memory_sf(memory_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);
    
    %% for updating the memory of crossover rate
    if max(goodCR) == 0 || memory_cr(memory_pos)  == -1
        memory_cr(memory_pos)  = -1;
    else
        memory_cr(memory_pos) = (dif_val' * (goodCR .^ 2)) / (dif_val' * goodCR);
    end
    
    memory_pos = memory_pos + 1;
    if memory_pos > memory_size;  memory_pos = 1; end
end

Pop(Fit_imp_inf,:) = X(Fit_imp_inf,:); % replace current by trial
Fit(Fit_imp_inf) = Child_Fit(Fit_imp_inf) ;

Par.memory_size=memory_size;
Par.memory_sf=memory_sf;
Par.memory_cr=memory_cr;
Par.memory_pos=memory_pos;

end

function r=genR_EADE(Fit)

NP=length(Fit);
r(:,1)=1:NP;

[srt, Fit_index]=sort(Fit,'ascend');

T=ceil(length(Fit_index)/10);
Best=Fit_index(1:T);
Mid=Fit_index(T+1:end-T);
Worest=Fit_index(end-T+1:end);

% choose three random individuals from population mutually different
r(:,2) = Best(ceil(length(Best)* rand(NP, 1)));

r(:,3) = Worest(ceil(length(Worest)* rand(NP, 1)));

r(:,4) = Mid(ceil(length(Mid)* rand(NP, 1)));

pos=r(:,2)==r(:,3);

while(sum(pos)~=0)
    r(pos,3) = Worest(ceil(length(Worest)* rand(sum(pos), 1)));
    pos=r(:,2)==r(:,3);
end

pos=r(:,3)==r(:,4);
while(sum(pos)~=0)
    r(pos,4) = Mid(ceil(length(Mid)* rand(sum(pos), 1)));
    pos=r(:,3)==r(:,4);
end

end

