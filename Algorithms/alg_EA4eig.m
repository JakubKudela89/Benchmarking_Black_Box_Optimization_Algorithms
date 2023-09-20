function [minima, xatmin, history] = alg_EA4eig(Problem, opts, bounds)
% original citation: 
% P. Bujok and P. Kolenovsky, “Eigen crossover in cooperative model of
% evolutionary algorithms applied to cec 2022 single objective numerical
% optimisation," in 2022 IEEE Congress on Evolutionary Computation
% (CEC). IEEE, 2022, pp. 1–8.
% modified by Jakub Kudela
% =========================================================================

tic;
a = bounds(:,1)';
b = bounds(:,2)';

fistar = -Inf;

fhd = Problem.f;
DIM = opts.dimension;
Runs = 1;
maxFES = opts.maxevals;

fmin_stage = zeros(16, 1);
for i = 0:15
    fmin_stage(i + 1) = round(DIM ^ ((i / 5) - 3) * maxFES);
end

stage = numel(fmin_stage);
val_2_reach = 1e-8;

results = zeros(1, 6);
for func_no = 1
    fprintf('\n-------------------------------------------------------\n')
    allerrorvals = zeros(stage + 1, Runs);    
    h = 4;
    cni_fun = zeros(Runs, h);
    
    for run_id = 1:Runs
        
        N_init = 100;
        N = N_init;
        Nmin = 10;
        n0 = 2;
        cni = zeros(1, h);
        ni = zeros(1, h) + n0;
        nrst = 0;
        success = zeros(1, h);
        
        Run_RecordFEsFactor = fmin_stage;
        run_funcvals = [];
        %% Initialize the main population
        bsf_fit_var = Inf; xatmin = zeros(DIM,1); history = [];
        FES = 0;
        while bsf_fit_var == Inf
            P = repmat(a, N, 1) + rand(N, DIM) .* (repmat(b - a, N, 1));
            %P(:, DIM + 1) = (feval(fhd, P', func_no))'; 
            for ii=1:N
                P(ii, DIM + 1) = fhd(P(ii,1:DIM)');
            end
            FES = FES + N;
            succ = 0;
            bsf_fit_var = min(P(:, DIM + 1));
            if FES > maxFES
                break;
            end
        end
        iter = 1;
        history(iter,1) = 1;
        history(iter,2) = FES;
        history(iter,3) = bsf_fit_var;
        history(iter,4) = toc;
        if opts.showits
            fprintf("Iter:%5i\tf_min:%15.10f\ttime(s):%10.5f\tfn evals:%9i\n",[history(iter,1),history(iter,3),history(iter,4),history(iter,2)]);
            %Iter:   43   f_min:  -78.9844713606    time(s):    0.34048    fn evals:     1723
        end
        Q = P;
        
        delta = 1 / (5 * h);
        
        %IDE params:
        gmax = round(maxFES / N);
        SRg = zeros(gmax, 1) + 2;
        T = gmax/10;
        GT = fix(gmax / 2);
        gt = GT; % auxiliary count of generation for SRT threshold
        g = 0;
        Tcurr = 0;
        
        % cobide:
        CBps = 0.5; %such part of P is selscetd fo eigenvestors...
        peig = 0.4; %probability to use an eigenvector crossover
        suceig = 0;
        ceig = 0;
        %        
        CBF = zeros(N, 1);
        CBCR = zeros(N, 1);
        for i = 1:N
            if rand < 0.5
                CBF(i) = cauchy_rnd(0.65, 0.1);
            else
                CBF(i) = cauchy_rnd(1, 0.1);
            end
            while CBF(i) < 0
                if rand < 0.5
                    CBF(i) = cauchy_rnd(0.65, 0.1);
                else
                    CBF(i) = cauchy_rnd(1, 0.1);
                end
            end
            if CBF(i) > 1
                CBF(i) = 1;
            end
            if rand < 0.5
                CBCR(i) = cauchy_rnd(0.1, 0.1);
            else
                CBCR(i) = cauchy_rnd(0.95, 0.1);
            end
            if CBCR(i) > 1
                CBCR(i) = 1;
            elseif CBCR(i) < 0
                CBCR(i) = 0;
            end
        end
        
        % CMAES params
        sigma = (b(1) - a(1)) / 2;
        oldPop = P(:, 1:DIM)'; 
        myeps = 1e-6;
        
        % Strategy parameter setting: Selection
        mu = N / 2;               % number of parents/points for recombination
        weights = log(mu + 1 / 2) - log(1:mu)'; % muXone array for weighted recombination
        mu = floor(mu);
        weights = weights / sum(weights);     % normalize recombination weights array
        mueff=sum(weights) ^ 2 / sum(weights .^ 2); % variance-effectiveness of sum w_i x_i
        
        % Strategy parameter setting: Adaptation
        cc = (4 + mueff / DIM) / (DIM + 4 + 2 * mueff / DIM); % time constant for cumulation for C
        cs = (mueff + 2) / (DIM + mueff + 5);  % t-const for cumulation for sigma control
        c1 = 2 / ((DIM + 1.3) ^ 2 + mueff);    % learning rate for rank-one update of C
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((DIM + 2) ^ 2 + mueff));  % and for rank-mu update
        damps = 1 + 2 * max(0, sqrt((mueff - 1) / (DIM + 1)) - 1) + cs; % damping for sigma
        % usually close to 1
        % Initialize dynamic (internal) strategy parameters and constants
        pc = zeros(DIM, 1); ps = zeros(DIM, 1);   % evolution paths for C and sigma
        B = eye(DIM, DIM);                       % B defines the coordinate system
        D = ones(DIM, 1);                      % diagonal D defines the scaling
        CC = B * diag(D .^ 2) * B';            % covariance matrix C
        invsqrtC = B * diag(D .^ -1) * B';    % C^-1/2
        eigeneval = 0;                      % track update of B and D
        chiN = DIM ^ 0.5 * (1 - 1 / (4 * DIM) + 1 / (21 * DIM ^ 2));  % expectation of ||DIM(0,I)|| == norm(randn(DIM,1))
        
        % jSO params
        Asize_max = round(N * 2.6);
        H = 5;
        MF = 0.3 * ones(1, H);
        MCR = 0.8 * ones(1, H);
        MF(1, H) = 0.9;
        MCR(1, H) = 0.9;
        k = 1;
        Asize = 0;
        A = [];
        pmax = 0.25;
        pmin = pmax / 2;
        
        div_stage = [];
        if(FES >= Run_RecordFEsFactor(1))
            run_funcvals = [run_funcvals; bsf_fit_var];
            Run_RecordFEsFactor(1) = [];
        end
        FESterm = 0;
        
        %% main loop
        while (FES < maxFES) && ((bsf_fit_var - fistar(func_no)) >= val_2_reach)
            
            [hh, p_min] = roulete(ni); % select algorithm to be used
            if p_min < delta
                cni = cni + ni - n0;
                ni = zeros(1, h) + n0;
                nrst = nrst + 1;
            end  %reset
            
            switch hh % number of selected heuristic
                case 1 % 1 generation of cobide
                    Q = zeros(N, DIM + 1);
                    if rand < peig % eig. cross will be used for whole pop.
                        ceig = 1;
                        Popeig = sortrows(P, DIM + 1);
                        Popeig(round(N * CBps + 1):N, :) = [];
                        [EigVect, ~] = eig(cov(Popeig(:, 1:DIM)));
                        for i = 1:N  %  in each generation
                            % rand1
                            vyb = nahvyb_expt(N, 3, i); % three random points from P
                            r1 = P(vyb(1), 1:DIM);
                            r2 = P(vyb(2), 1:DIM);
                            r3 = P(vyb(3), 1:DIM);
                            v = r1 + CBF(i) * (r2 - r3);
                            % and mirroring the point by boundaries
                            v = zrcad(v, a, b);
                            %
                            y = P(i, 1:DIM);
                            yeig = EigVect' * y';
                            veig = EigVect' * v';
                            change = find(rand(1, DIM) < CBCR(i));
                            if isempty(change)% at least one element is changed
                                change = 1 + fix(DIM * rand(1));
                            end
                            yeig(change) = veig(change);
                            y = (EigVect * yeig)';
                            
                            y = zrcad(y, a, b);
                            %fy = feval(fhd, y', func_no);
                            fy = fhd(y');
                            Q(i, :) = [y fy];
                            
                            FES = FES + 1;
                        end
                    else
                        for i = 1:N  %  in each generation
                            % rand1
                            vyb = nahvyb_expt(N, 3, i); % three random points from P
                            r1 = P(vyb(1), 1:DIM);
                            r2 = P(vyb(2), 1:DIM);
                            r3 = P(vyb(3), 1:DIM);
                            v = r1 + CBF(i) * (r2 - r3);
                            % and mirroring the point by boundaries
                            v = zrcad(v, a, b);
                            %
                            y = P(i, 1:DIM);
                            change = find(rand(1, DIM) < CBCR(i));
                            if isempty(change)% at least one element is changed
                                change = 1 + fix(DIM * rand(1));
                            end
                            y(change) = v(change);
                            
                            y = zrcad(y, a, b);
                            %fy = feval(fhd, y', func_no);
                            fy = fhd(y');
                            
                            Q(i, :) = [y fy];
                            FES = FES + 1;
                        end
                    end
                    
                    if ceig == 1 %generation with Eig
                        for i = 1:N
                            if Q(i, DIM + 1) <= P(i, DIM + 1)  % trial error y is less than old
                                P(i, :) = Q(i, :);
                                suceig = suceig + 1;
                                success(1, hh) = success(1, hh) + 1;
                                ni(hh) = ni(hh) + 1;
                            else
                                if rand < 0.5
                                    CBF(i) = cauchy_rnd(0.65, 0.1);
                                else
                                    CBF(i) = cauchy_rnd(1, 0.1);
                                end
                                while CBF(i) < 0
                                    if rand < 0.5
                                        CBF(i) = cauchy_rnd(0.65, 0.1);
                                    else
                                        CBF(i) = cauchy_rnd(1, 0.1);
                                    end
                                end
                                if CBF(i) > 1
                                    CBF(i) = 1;
                                end
                                if rand < 0.5
                                    CBCR(i) = cauchy_rnd(0.1, 0.1);
                                else
                                    CBCR(i) = cauchy_rnd(0.95, 0.1);
                                end
                                if CBCR(i) > 1
                                    CBCR(i) = 1;
                                elseif CBCR(i) < 0
                                    CBCR(i) = 0;
                                end
                            end
                        end
                        ceig = 0;
                    else
                        for i = 1:N
                            if Q(i, DIM + 1) <= P(i, DIM + 1)  % trial error y is less than old
                                P(i, :) = Q(i, :);
                                success(1, hh) = success(1, hh) + 1;
                                ni(hh) = ni(hh) + 1;
                            else
                                if rand < 0.5
                                    CBF(i) = cauchy_rnd(0.65, 0.1);
                                else
                                    CBF(i) = cauchy_rnd(1, 0.1);
                                end
                                while CBF(i) < 0
                                    if rand < 0.5
                                        CBF(i) = cauchy_rnd(0.65, 0.1);
                                    else
                                        CBF(i) = cauchy_rnd(1, 0.1);
                                    end
                                end
                                if CBF(i) > 1
                                    CBF(i) = 1;
                                end
                                if rand < 0.5
                                    CBCR(i) = cauchy_rnd(0.1, 0.1);
                                else
                                    CBCR(i) = cauchy_rnd(0.95, 0.1);
                                end
                                if CBCR(i) > 1
                                    CBCR(i) = 1;
                                elseif CBCR(i) < 0
                                    CBCR(i) = 0;
                                end
                            end
                        end
                    end
                    [fmin, ~] = min(P(:, DIM + 1));
                    
                    if fmin < bsf_fit_var
                        bsf_fit_var = fmin;
                        pom = bsf_fit_var;                        
                        if pom == 0
                            disp('cobi')
                            disp(FES)                            
                        end
                    end
                    
                case 2 %1 generation of IDE
                    [P, I] = sortrows(P, DIM + 1);
                    CBF = CBF(I);
                    CBCR = CBCR(I);
                    Q = P;
                    IDEps = 0.1 + 0.9 * 10^(5*(g/gmax - 1)); % ratio of Superior
                    pd = 0.1 * IDEps;  % prob. of perturbation
                    if g < gt
                        SRT = 0;
                    else
                        SRT = 0.1;
                    end
                    for i=1:N  %  mutation in each generation
                        vyb = nahvyb_expt(N,4, i);
                        o = vyb(1);
                        if g <= gt
                            o = i;
                        end
                        r1 = vyb(2);
                        r2 = vyb(3);
                        r3 = vyb(4);
                        xo = P(o,1:DIM);
                        xr1= P(r1,1:DIM);
                        xr2= P(r2,1:DIM);
                        xr3= P(r3,1:DIM);
                        indperturb = find(rand(1,DIM) < pd);
                        pom = a + rand(1, DIM) .* (b - a);
                        xr3(indperturb) = pom(indperturb);
                        
                        Fo = o / N + 0.1 * randn(1);
                        while Fo <= 0 || Fo > 1
                            Fo = o/N + 0.1 * randn(1);
                        end
                        high_ind_S = fix(IDEps * N);
                        if o > high_ind_S  % Inferior
                            if r1 > high_ind_S  %  find "better" from Superior
                                candidates = setdiff(1:high_ind_S, [vyb, i]);
                                r1 = 1 + fix(rand(1) * length(candidates));
                                xr1 =  P(r1,1:DIM);
                            end
                        end
                        if (g > gt) && (rand < 0.5) % with probability 0.1 - best point is used:
                            Q(i, 1:DIM) = P(i, 1:DIM) + Fo*(xr1 - xo) + Fo*(xr2 - xr3);
                        else
                            Q(i, 1:DIM) = xo + Fo*(xr1 - xo) + Fo*(xr2 - xr3);
                        end
                    end % end of mutation in the current generation
                    
                    if rand < peig % eig. cross will be used for whole pop.
                        ceig = 1;
                        Popeig = sortrows(P, DIM + 1);
                        Popeig(round(N * CBps + 1):N, :) = [];
                        [EigVect, ~] = eig(cov(Popeig(:, 1:DIM)));
                        for i=1:N  %  crossover
                            y = P(i, 1:DIM);
                            v = Q(i, 1:DIM);
                            yeig = EigVect' * y';
                            veig = EigVect' * v';
                            change = find(rand(1, DIM) < CBCR(i));
                            if isempty(change)% at least one element is changed
                                change = 1 + fix(DIM * rand(1));
                            end
                            yeig(change) = veig(change);
                            y = (EigVect * yeig)';
                            
                            Q(i, 1:DIM) = zrcad(y, a, b);
                        end % end of mutation in the current generation
                    else
                        ceig = 0;
                        for i=1:N  %  crossover
                            CR = i/N + 0.1 * randn(1);
                            while CR < 0 || CR > 1
                                CR = i/N + 0.1 * randn(1);
                            end
                            jrand = 1 + fix(rand(1)* DIM);
                            for j=1:DIM
                                if ~(rand(1) <= CR || j  == jrand) % take old P
                                    Q(i,j) = P(i,j);
                                end
                                if Q(i,j) < a(j) || Q(i,j) > b(j) % out of boundary
                                    Q(i,j) = a(j) + rand(1)*(b(j) - a(j));
                                end
                            end  % Q is new genaration
                        end % end of mutation in the current generation
                    end
                    [temp1,~] = size(Q);
                    for ii=1:temp1
                        Q(ii, DIM + 1) = fhd(Q(ii,1:DIM)');
                    end
                    %Q(:, DIM + 1) = (feval(fhd, Q(:, 1:DIM)', func_no))';                    
                    FES = FES + N;
                    if length(P(:, 1)) ~= length(Q(:, 1))
                        disp(size(P(:, 1)))
                        disp(size(Q(:, 1)))
                    end
                    indsucc = find(Q(:,DIM+1) <= P(:,DIM+1));
                    success(1, hh) = success(1, hh) + length(indsucc);
                    ni(hh) = ni(hh) + length(indsucc);
                    SR = length(indsucc) / N;
                    if  g < gt  % stage == 1
                        if SR <= SRT
                            Tcurr = Tcurr + 1;
                        else
                            Tcurr = 0;
                        end
                        if Tcurr >= T    %%  stage = 2;
                            gt = g; %;
                        end 
                    end
                    P(indsucc,:) = Q(indsucc,:); % replace by better points
                    [P, I] = sortrows(P, DIM + 1);
                    CBF = CBF(I);
                    CBCR = CBCR(I);
                    
                    fmin= P(1, DIM + 1);
                    if fmin < bsf_fit_var
                        bsf_fit_var = fmin;
                        pom = bsf_fit_var;
                        if pom == 0
                            disp('ide')
                            disp(FES)
                        end
                    end
                    g = g + 1;
                    
                case 3 % 1 generation of CMAES
                    [P, I] = sortrows(P, DIM + 1);
                    CBF = CBF(I);
                    CBCR = CBCR(I);
                    
                    xmean = P(1:mu, 1:DIM)' * weights;  % recombination, new mean value
                    Pop = zeros(DIM, N);
                    PopFit = zeros(1, N);
                    for kk = 1:N
                        Pop(:, kk) = xmean + sigma * B * (D .* randn(DIM, 1)); 
                        %mirroring
                        zrca = find(Pop(:, kk) < a');
                        Pop(zrca, kk) = (oldPop(zrca, kk) + a(zrca)') / 2;
                        zrcb = find(Pop(:, kk) > b');
                        Pop(zrcb, kk) = (oldPop(zrcb, kk) + b(zrcb)') / 2;
                        PopFit(kk) = fhd(Pop(:, kk));
                        %PopFit(kk) = feval(fhd, Pop(:, kk), func_no);
                        FES = FES + 1;
                        % if the solution is better than any of P - replace
                        % and increase success
                        [maxf, maxind] = max(P(:, DIM + 1));
                        if PopFit(kk) < maxf
                            P(maxind, :) = [Pop(:, kk)', PopFit(kk)];
                            success(1, hh) = success(1, hh) + 1;
                            ni(hh) = ni(hh) + 1;
                        end                        
                    end
                    % Sort by fitness and compute weighted mean into xmean
                    [PopFit, FitInd] = sort(PopFit);  % minimization
                    xold = xmean;
                    xmean = Pop(:, FitInd(1:mu)) * weights;  % recombination, new mean value - VAZENY SOUCET!!!
                    fmin = min(P(:, DIM + 1));                    
                    if fmin < bsf_fit_var
                        bsf_fit_var = fmin;
                        pom = bsf_fit_var;
                        if pom == 0
                            disp('cma')
                            disp(FES)
                        end
                    end
                     
                    oldPop = Pop;
                    
                    % Cumulation: Update evolution paths
                    ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * invsqrtC * (xmean - xold) / sigma;
                    hsig = sum(ps .^ 2) / (1 - (1 - cs) ^ (2 * FES / N)) / DIM < 2 + 4 / (DIM + 1);
                    pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma;
                    
                    % Adapt covariance matrix C
                    artmp = (1 / sigma) * (Pop(:, FitInd(1:mu)) - repmat(xold, 1, mu));  % mu difference vectors
                    CC = (1 - c1 - cmu) * CC + c1 * (pc * pc' + (1 - hsig) * cc * (2 - cc) * CC) + cmu * artmp * diag(weights) * artmp';
                    % regard old matrix plus rank one update minor correction if hsig==0 plus rank mu update
                    
                    % Adapt step size sigma
                    sigma = sigma * exp((cs / damps) * (norm(ps) / chiN - 1));
                    if (sigma > 1e+300) || (sigma < 1e-300) || isnan(sigma)
                        sigma = (b(1) - a(1)) / 2;
                    end
                    % Update B and D from C
                    if FES - eigeneval > N / (c1 + cmu) / DIM / 10  % to achieve O(DIM^2)
                        eigeneval = FES;
                        CC = triu(CC) + triu(CC, 1)'; % enforce symmetry
                        [B, D] = eig(CC);           % eigen decomposition, B==normalized eigenvectors
                        D = sqrt(diag(D));        % DIM contains standard deviations now
                        invsqrtC = B * diag(D .^ -1) * B';
                    end
                    
                    % Break, if fitness is good enough or condition exceeds 1e14, better termination methods are advisable
%                     if (PopFit(1)) <= myeps || max(D) > 1e7 * min(D)
%                         break;
%                     end
                    
                case 4 % jSO generation
                    Fpole = -1 * ones(1, N);
                    CRpole = -1 * ones(1, N);
                    SCR = []; SF = [];
                    suc = 0;
                    Q = zeros(N, DIM + 1);
                    deltafce = -1 * ones(1, N);
                    pp = pmax - ((pmax - pmin) * (FES / maxFES));
                    if rand < peig
                        ceig = 1;
                        Popeig = sortrows(P, DIM + 1);
                        Popeig(round(N * CBps + 1):N, :) = [];
                        [EigVect, ~] = eig(cov(Popeig(:, 1:DIM)));
                        for i = 1:N
                            rr = nahvyb(H, 1);
                            CR = MCR(1, rr) + sqrt(0.1) * randn(1);
                            if CR > 1
                                CR = 1;
                            elseif CR < 0
                                CR = 0;
                            end
                            % jSO CR:
                            if FES < (0.25 * maxFES)
                                CR = max([CR 0.7]);
                            elseif FES < (0.5 * maxFES)
                                CR = max([CR 0.6]);
                            end
                            F = -1;
                            while F <= 0
                                F = rand * pi - pi / 2;
                                F = 0.1 * tan(F) + MF(1, rr);
                            end
                            if F > 1
                                F = 1;
                            end
                            %jSO F:
                            if (FES < (0.6 * maxFES)) && (F > 0.7)
                                F = 0.7;
                            end
                            Fpole(1, i) = F;
                            CRpole(1, i) = CR;
                            
                            expt = i;
                            p = max(2, ceil(pp * N));
                            pom = P;
                            pom = sortrows(pom, DIM + 1);
                            pbest = pom(1:p, 1:DIM);
                            ktery = 1 + fix(p * rand(1));
                            xpbest = pbest(ktery, :);
                            
                            xi = P(expt, 1:DIM);
                            
                            vyb = nahvyb_expt(N, 1, expt);
                            r1 = P(vyb, 1:DIM);
                            expt = [expt, vyb];
                            
                            vyb = nahvyb_expt(N + Asize, 1, expt);
                            sjed = [P(:, 1:DIM); A];
                            r2 = sjed(vyb, :);
                            
                            if FES < 0.2 * maxFES
                                Fw = 0.7 * F;
                            elseif FES < 0.4 * maxFES
                                Fw = 0.8 * F;
                            else
                                Fw = 1.2 * F;
                            end
                            
                            v = xi + Fw * (xpbest - xi) + F * (r1 - r2);
                            y = xi;
                            
                            yeig = EigVect' * y';
                            veig = EigVect' * v';
                            change = find(rand(1, DIM) < CBCR(i));
                            if isempty(change)% at least one element is changed
                                change = 1 + fix(DIM * rand(1));
                            end
                            yeig(change) = veig(change);
                            y = (EigVect * yeig)';
                            Q(i, 1:DIM) = y;
                        end
                    else
                        ceig = 0;
                        for i = 1:N
                            rr = nahvyb(H, 1);
                            CR = MCR(1, rr) + sqrt(0.1) * randn(1);
                            if CR > 1
                                CR = 1;
                            elseif CR < 0
                                CR = 0;
                            end
                            % jSO CR:
                            if FES < (0.25 * maxFES)
                                CR = max([CR 0.7]);
                            elseif FES < (0.5 * maxFES)
                                CR = max([CR 0.6]);
                            end
                            F = -1;
                            while F <= 0
                                F = rand * pi - pi / 2;
                                F = 0.1 * tan(F) + MF(1, rr);
                            end
                            if F > 1
                                F = 1;
                            end
                            %jSO F:
                            if (FES < (0.6 * maxFES)) && (F > 0.7)
                                F = 0.7;
                            end
                            Fpole(1, i) = F;
                            CRpole(1, i) = CR;
                            
                            expt = i;
                            p = max(2, ceil(pp * N));
                            pom = P;
                            pom = sortrows(pom, DIM + 1);
                            pbest = pom(1:p, 1:DIM);
                            ktery = 1 + fix(p * rand(1));
                            xpbest = pbest(ktery, :);
                            
                            xi = P(expt, 1:DIM);
                            
                            vyb = nahvyb_expt(N, 1, expt);
                            r1 = P(vyb, 1:DIM);
                            expt = [expt, vyb];
                            
                            vyb = nahvyb_expt(N + Asize, 1, expt);
                            sjed = [P(:, 1:DIM); A];
                            r2 = sjed(vyb, :);
                            
                            if FES < 0.2 * maxFES
                                Fw = 0.7 * F;
                            elseif FES < 0.4 * maxFES
                                Fw = 0.8 * F;
                            else
                                Fw = 1.2 * F;
                            end
                            
                            v = xi + Fw * (xpbest - xi) + F * (r1 - r2);
                            y = xi;
                            
                            change = find(rand(1, DIM) < CR);
                            if isempty(change) % at least one element is changed
                                change = 1 + fix(DIM * rand(1));
                            end
                            y(change) = v(change);
                            y = zrcad(y, a, b);
                            Q(i, 1:DIM) = y;
                        end
                    end
                    [temp1,~] = size(Q);
                    for ii=1:temp1
                        Q(ii, DIM + 1) = fhd(Q(ii,1:DIM)');
                    end
                    %Q(:, DIM + 1) = (feval(fhd, Q(:, 1:DIM)', func_no))';
                    for i = 1:N
                        if Q(i, DIM + 1) < P(i, DIM + 1)
                            deltafce(1, i) = P(i, DIM + 1) - Q(i, DIM + 1);
                            suc = suc + 1;
                            if Asize < Asize_max
                                A = [A; P(i, 1:DIM)];
                                Asize = Asize + 1;
                            else
                                ktere = nahvyb(Asize, 1);
                                A(ktere, :) = P(i, 1:DIM);
                            end
                            SCR = [SCR, CRpole(1, i)];
                            SF = [SF, Fpole(1, i)];
                        end
                        if Q(i, DIM + 1) <= P(i, DIM + 1)
                            P(i, :) = Q(i, :);
                        end
                    end
                    
                    if suc > 0
                        MCR_old = MCR(1, k);
                        MF_old = MF(1, k);
                        platne = find(deltafce ~= -1);
                        delty = deltafce(1, platne);
                        sum_delta = sum(delty);
                        vahyw = 1 / sum_delta * delty;
                        mSCR = max(SCR);
                        if (MCR(1, k) == -1) || (mSCR == 0)
                            MCR(1, k) = -1;
                        else
                            meanSCRpomjm = vahyw .* SCR;
                            meanSCRpomci = meanSCRpomjm .* SCR;
                            MCR(1, k) = sum(meanSCRpomci) / sum(meanSCRpomjm);
                        end
                        meanSFpomjm = vahyw .* SF;
                        meanSFpomci = meanSFpomjm .* SF;
                        MF(1, k) = sum(meanSFpomci) / sum(meanSFpomjm);
                        MCR(1, k) = (MCR(1, k) + MCR_old) / 2;
                        MF(1, k) = (MF(1, k) + MF_old) / 2; 
                        k = k + 1;
                        if k >= H 
                            k = 1;
                        end
                    end
                    FES = FES + N;
                    fmin = min(P(:, DIM + 1));
                    
                    if fmin < bsf_fit_var
                        bsf_fit_var = fmin;
                        pom = bsf_fit_var;                        
                        if pom == 0
                            disp('jso')
                            disp(FES)                            
                        end
                    end
 
                    success(1, hh) = success(1, hh) + suc;
                    ni(hh) = ni(hh) + suc;
            end
            
            if min(P(:, DIM + 1)) < bsf_fit_var
                bsf_fit_var = min(P(:, DIM + 1));
                if (bsf_fit_var - fistar(func_no)) < val_2_reach
                    FESterm = FES; %for early stopped alg. to publish
                end
            end 
            
            if(FES >= Run_RecordFEsFactor(1))
                run_funcvals = [run_funcvals; bsf_fit_var];
                prum = mean(P(:, 1:DIM));
                prum_mat = repmat(prum, N, 1);
                div_stage = [div_stage; sqrt(sum(sum((P(:, 1:DIM) - prum_mat) .* (P(:, 1:DIM) - prum_mat))) / N)];
                Run_RecordFEsFactor(1) = [];
            end
            
            optN = round((((Nmin - N_init) / maxFES) * FES) + N_init);
            
            if N > optN
                diffPop = N - optN;
                if N - diffPop < Nmin
                    diffPop = N - Nmin;
                end
                
                N = N - diffPop;
                [P, I] = sortrows(P, DIM + 1);
                CBF = CBF(I);
                CBCR = CBCR(I);
                
                P(N + 1:end, :) = [];
                Q(N + 1:end, :) = [];
                CBF(N + 1:end) = [];
                CBCR(N + 1:end) = [];
                Asize_max = round(N * 2.6);
                while Asize > Asize_max
                    index_v_arch = nahvyb(Asize, 1);
                    A(index_v_arch, :) = [];
                    Asize = Asize - 1;
                end
                % CMAES
                mu = floor(N / 2);
                weights = log(mu + 1 / 2) - log(1:mu)'; % muXone array for weighted recombination
                weights = weights / sum(weights);     % normalize recombination weights array
                mueff = sum(weights) ^ 2 / sum(weights .^ 2);
            end
            iter = iter + 1;
            history(iter,1) = iter;
            history(iter,2) = FES;
            history(iter,3) = bsf_fit_var;
            history(iter,4) = toc;        
            if opts.showits && ~mod(iter,50) %only print every 50 iterations
                fprintf("Iter:%5i\tf_min:%15.10f\ttime(s):%10.5f\tfn evals:%9i\n",[history(iter,1),history(iter,3),history(iter,4),history(iter,2)]);
                %Iter:   43   f_min:  -78.9844713606    time(s):    0.34048    fn evals:     1723
            end
            if bsf_fit_var-opts.globalmin < opts.tolabs || toc > 600
                break
            end
        end
    
    end %% end 1 run
    
%     fclose(fid);
            
end %% end 1 function run

minima = bsf_fit_var;
xatmin = P(1,1:DIM)';
end


function  y = cauchy_rnd(x0, gamma)
% nah. cis. z Cauchyho rozdeleni a parametry x0, gamma
y = x0 + gamma * tan (pi * (rand(1) - 1 / 2));
end


% random sample, k of N without repetiotion
%
function nahv = nahvyb(N,k)
opora = 1:N;
nahv = zeros(1,k);
for i = 1:k
	index = 1+fix(rand(1)*length(opora));
	nahv(i) = opora(index);
	opora(index) = [];
end
end

% random sample, k of N without repetition, 
% numbers given in vector expt are not included
%
function vyb = nahvyb_expt(N,k,expt)
opora = 1:N;
if nargin==3 opora(expt)=[]; end
vyb = zeros(1,k);
for i = 1:k
	index = 1+fix(rand(1)*length(opora));
	vyb(i) = opora(index);
	opora(index) = [];
end
end

function [res, p_min] = roulete(cutpoints)
%
% returns an integer from [1, length(cutpoints)] with probability proportional
% to cutpoints(i)/ summa cutpoints
%
h = length(cutpoints);
ss = sum(cutpoints);
p_min = min(cutpoints)/ss;
cp(1) = cutpoints(1);
for i = 2:h
    cp(i) = cp(i-1)+cutpoints(i);
end
cp = cp/ss;
res = 1+fix(sum(cp<rand(1)));
end

% mirroring, Perturbation y into <a,b>
% function result = zrcad(y,a,b)
% zrc = find(y<a|y>b); 
% for i = zrc
% 	while (y(i)<a(i) || y(i)>b(i))
% 		if y(i) > b(i)
% 		    y(i) = 2*b(i)-y(i);
% 		elseif y(i) < a(i)
% 		    y(i) = 2*a(i)-y(i);
%       end
% 	end
% end
% result=y;
% end

%new version
function result = zrcad(y,a,b)
y = real(y); zrc = find(y<a|y>b); y_in = y; y_orig = y;
b_mod = b; b(b == 0) = 1; a_mod = a; a(a == 0) = -1;
y_in(y_in > b_mod) = mod(y_in(y_in > b_mod), 1.5*b_mod(y_in > b_mod))+1.5*b_mod(y_in > b_mod);
y_in(y_in < a_mod) = mod(y_in(y_in < a_mod), 1.5*a_mod(y_in < a_mod))+1.5*a_mod(y_in < a_mod);
y = y_in;
for i = zrc
	while (y(i)<a(i) || y(i)>b(i))
		if y(i) > b(i)
		    y(i) = 2*b(i)-y(i);
		elseif y(i) < a(i)
		    y(i) = 2*a(i)-y(i);
        end
	end
end
result=y;
end