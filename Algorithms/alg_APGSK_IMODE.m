function [minima, xatmin, history] = alg_APGSK_IMODE(Problem, opts, bounds)
% original citation: 
% A. W. Mohamed, A. A. Hadi, P. Agrawal, K. M. Sallam, and A. K.
% Mohamed, "Gaining-sharing knowledge based algorithm with adaptive
% parameters hybrid with imode algorithm for solving cec 2021 benchmark
% problems," in 2021 IEEE Congress on Evolutionary Computation (CEC).
% IEEE, 2021, pp. 841–848.
% modified by Jakub Kudela
% =========================================================================

tic;
fhd=Problem.f; 
lu = bounds';
D = opts.dimension;

max_nfes = opts.maxevals;


[Par] = Introd_Par(D);
Par.max_nfes = max_nfes ;
C = 1;

NP=Par.PopSize;
max_NP = NP;
min_NP = 12.0;
min_NP1=4;

bsf_fit_var = Inf;
nfes = 0;
while bsf_fit_var == Inf && nfes < max_nfes
    Pop = repmat(lu(1, :), NP, 1) + rand(NP, D) .* (repmat(lu(2, :) - lu(1, :), NP, 1));
    
    [temp,~] = size(Pop); Fit = zeros(temp,1);
    for it1=1:temp
        Fit(it1,1) = fhd(Pop(it1,:)');
    end
    nfes = nfes + NP;
%         fitness = feval(fhd, pop', func, C(m,:));
%         fitness = fitness';

    [bsf_fit_var,I_best]=min(Fit);
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

% Fit=feval(fhd,Pop',func_num,C);
% nfes = NP;
% Fit=Fit';

X_best=Pop(I_best,:);

minima = bsf_fit_var;
xatmin = X_best';

res_det= min(repmat(min(Fit),Par.PopSize,1), Fit); %% used to record the convergence

PS2=NP/4;
PS1=NP-PS2;
% PS2=NP-PS1;
max_NP1=PS1;
max_NP2=PS2;
%% ================== fill in for each  Algorithm ===================================
%% IMODE
EA_1= Pop(1:PS1,:);    EA_obj1= Fit(1:PS1);
%% APGSK
EA_2= Pop(PS1+1:size(Pop,1),:);    EA_obj2= Fit(PS1+1:size(Pop,1));

%% ===================== archive data ====================================
arch_rate=1.4;
archive.NP = arch_rate * PS1; % the maximum size of the archive
archive.pop = zeros(0, Par.n); % the solutions stored in te archive
archive.funvalues = zeros(0, 1); % the function value of the archived solutions



%% APGSK setting
%%POSSIBLE VALUES FOR KNOWLEDGE RATE K%%%%
EA_1old=EA_1;
hist_pos=1;
memory_size=15*Par.n;
archive_f= ones(1,memory_size).*0.5;
archive_Cr= ones(1,memory_size).*0.5;
archive_T = ones(1,memory_size).*0.1;
archive_freq = ones(1, memory_size).*0.5;
F = normrnd(0.5,0.15,1,NP);
cr= normrnd(0.5,0.15,1,NP);
probDE1=1./Par.n_opr .* ones(1,Par.n_opr);
[bestold, bes_l]=min(Fit);     bestx= Pop(bes_l,:);
Probs=[1 1];
it=0;
cy=0;
stop_con=0;
while stop_con==0 % nfes<max_nfes
    
    it=it+1;
    cy=cy+1; % to control CS
    %  ================ determine the best phase ===========================
    if(cy==ceil(Par.CS+1))
        
        %%calc normalized qualit -- NQual
        qual(1) = min(EA_obj1);
        qual(2) = min(EA_obj2);
        norm_qual = qual./sum(qual);
        norm_qual=1-norm_qual; %% to satisfy the bigger is the better
        Probs=norm_qual;
        %%Update Prob_MODE and Prob_CMAES
        Probs = max(0.1, min(0.9,Probs./sum(Probs)));
        
        [~,indx]=max(Probs);
        if Probs(1)==Probs(2)
            indx=0;%% no sharing of information
        end
        if indx>0
            Probs=[0 0];
            Probs(indx)=1;
        end
    elseif cy==2*ceil(Par.CS)
        
        %% share information
        if indx==1
            EA_2(PS2,:)=EA_1(1,:);
            EA_obj2(PS2)=EA_obj1(1);
            [EA_obj2, ind]=sort(EA_obj2);
            EA_2=EA_2(ind,:);
        else
            if (min (EA_2(1,:)))> -100 && (max(EA_2(1,:)))<100 %% share best sol. in EA_2 if it is feasible
                EA_1(PS1,:)= EA_2(1,:);
                EA_obj1(PS1)= EA_obj2(1);
                [EA_obj1, ind]=sort(EA_obj1);
                EA_1=EA_1(ind,:);
            end
            
        end
        %% reset cy and Probs
        cy=1;   Probs=ones(1,2);
    end
    
    if (rand<=Probs(1))
        %% apply IMODE
        [EA_1, EA_1old, EA_obj1,probDE1,bestold,bestx,archive,hist_pos,memory_size, archive_f,archive_Cr,archive_T,archive_freq, nfes,F,cr, res_det] = ...
            IMODE( EA_1,EA_1old, EA_obj1,probDE1,bestold,bestx,archive,hist_pos,memory_size, archive_f,archive_Cr,archive_T,....
            archive_freq, Par.xmin, Par.xmax,  Par.n,  PS1,  nfes, fhd,Par.Printing,Par.max_nfes, Par.Gmax,F,cr,C,fhd,res_det);
%         if(~isempty(Par.Record_FEs))
%             if (nfes>=Par.Record_FEs(1))
%                 Par.All_fit=[Par.All_fit;min(EA_obj1)];
%                 Par.Record_FEs(1)=[];
%             end
%         end
        if nfes >= max_nfes
            [Best_fit,I_best]= min(EA_obj1);
            X_best=EA_1(I_best,:);
            All_fit=Par.All_fit;
            return;
        end
        
        plan_NP1 = round((((min_NP1 - max_NP1) / (max_nfes)) * nfes) + max_NP1);
        %                     UpdPopSize = round((((Par.MinPopSize - InitPop) / Par.Max_FES) * current_eval) + InitPop);
        
        Par.p_best_rate = (((Par.p_best_rate_min - Par.p_best_rate_max) / (max_nfes)) * nfes) + Par.p_best_rate_max;
        
        if PS1 > plan_NP1
            reduction_ind_num = PS1 - plan_NP1;
            if PS1 - reduction_ind_num <  min_NP1
                reduction_ind_num = PS1 - min_NP1;
            end
            PS1 = PS1 - reduction_ind_num;
            for r = 1 : reduction_ind_num
                [valBest, indBest] = sort(EA_obj1, 'ascend');
                worst_ind = indBest(end);
                EA_1(worst_ind,:) = [];
                EA_obj1(worst_ind,:) = [];
            end
            archive.NP = PS1;
            if size(archive.NP, 1) > archive.NP
                rndpos = randperm(size(archive.NP, 1));
                rndpos = rndpos(1 : archive.NP);
                archive.Pop = archive.Pop(rndpos, :);
            end
        end
        
    end
    
    if(rand<Probs(2))
        %% Apply APGSK
        [EA_2,EA_obj2,Par,nfes,res_det]= APGSK_fun(EA_2,EA_obj2,lu,fhd,Par,nfes,fhd,C,res_det);

        if nfes >= max_nfes
            [Best_fit,I_best]= min(EA_obj2);
            X_best=EA_2(I_best,:);
            All_fit=Par.All_fit;
            break;
        end
        plan_NP2 = round((((min_NP - max_NP2) / (max_nfes)) * nfes) + max_NP2);
        Par.p_best_rate = (((Par.p_best_rate_min - Par.p_best_rate_max) / (max_nfes)) * nfes) + Par.p_best_rate_max;
        
        if PS2 > plan_NP2
            reduction_ind_num = PS2 - plan_NP2;
            if PS2 - reduction_ind_num <  min_NP
                reduction_ind_num = PS2 - min_NP;
            end
            PS2 = PS2 - reduction_ind_num;
            for r = 1 : reduction_ind_num
                [valBest, indBest] = sort(EA_obj2, 'ascend');
                worst_ind = indBest(end);
                EA_2(worst_ind,:) = [];
                EA_obj2(worst_ind,:) = [];
                Par.K(worst_ind,:)=[];
            end
        end
    end
    
        Fit=[EA_obj1;EA_obj2];
        Pop=[EA_1;EA_2];
        [Best_fit,I_best]= min(Fit);
        X_best=Pop(I_best,:);
        All_fit=Par.All_fit;
        PopSize=size(Pop,1);
%         res_det= [res_det; repmat(Best_fit,PopSize,1)];
    
    
    % fprintf('current_eval\t %d fitness\t %d \n', nfes, Best_fit);
    iter = iter + 1;
    history(iter,1) = iter;
    history(iter,2) = nfes;
    history(iter,3) = Best_fit;
    history(iter,4) = toc;       
    minima = Best_fit;
    xatmin = X_best';
    if opts.showits && ~mod(iter,50) 
        fprintf("Iter:%5i\tf_min:%15.10f\ttime(s):%10.5f\tfn evals:%9i\n",[history(iter,1),history(iter,3),history(iter,4),history(iter,2)]);
        %Iter:   43   f_min:  -78.9844713606    time(s):    0.34048    fn evals:     1723
    end
    if Best_fit-opts.globalmin < opts.tolabs
        break
    end


    if (nfes>=Par.max_nfes)
        stop_con=1;
    end
%     if ( (abs (Par.f_optimal - Best_fit)<= 1e-8))
%         stop_con=1;
%     end
end



end

function [Par] = Introd_Par(n)

%% loading

Par.n_opr=3;  %% number of operators
Par.n=n;     %% number of decision vriables


if Par.n<=10
    Par.CS=50; %% cycle
    Par.max_nfes=200000;
else
    Par.CS=50; %% cycle
    Par.max_nfes=1000000;
end
Par.Gmax=1000;

% if C(1)==1
%     opt= [100, 1100,700,1900,1700,1600,2100,2200,2400,2500];      %% define the optimal solution as shown in the TR
% else
%     opt=[0,0,0,0,0,0,0,0,0,0]; %% define the optimal solution as shown in the TR
% end
Par.xmin= -100*ones(1,Par.n);
Par.xmax= 100*ones(1,Par.n);

%Par.f_optimal=opt(I_fno);
Par.PopSize=30*Par.n; %% population size


%% from Ali Code
Par.p_best_rate = 0.5;
Par.p_best_rate_max = Par.p_best_rate;
Par.p_best_rate_min = 0.15;

Par.All_fit=[];
%Par.Record_FEs=Par.max_nfes.*[0.01, 0.02, 0.03, 0.05, 0.1:0.1:1];
Par.First_calss_percentage=0.5;
Par.Hybridization_flag=1;

% Par.memory_size=15*Par.n;
% Par.memory_sf = 0.5 .* ones(Par.memory_size, 1);
% Par.memory_cr = 0.5 .* ones(Par.memory_size, 1);
% Par.memory_pos = 1;

% Par.memory_1st_class_percentage = Par.First_calss_percentage.* ones(Par.memory_size, 1); % Class#1 probability for Hybridization

% [CMAES_Par]= CMAES_Init(Par.PopSize,Par.n);
% Par.CMAES_Par=CMAES_Par;

K=[];

Kind=rand(round(Par.PopSize/4), 1);
%%%%%%%%%%%%%%%%%%%K uniform rand (0,1) with prob 0.5 and unifrom integer [1,20] with prob 0.5
K(Kind<0.5,:)= rand(sum(Kind<0.5), 1);
K(Kind>=0.5,:)=ceil(20 * rand(sum(Kind>=0.5), 1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Par.All_Imp=zeros(1,4);
Par.K=K;
Par.KW_ind=[];
%% printing the detailed results- this will increase the computational time
Par.Printing=1; %% 1 to print; 0 otherwise

end



%% ============ Improved Multi-operator Differential Evolution Algorithm (IMODE) ============
% Should you have any queries, please contact
% Dr. Karam Sallam. Zagazig University
% karam_sallam@zu.edu.eg
% =========================================================================
function [x, xold, fitx,prob,bestold,bestx,archive,hist_pos,memory_size, archive_f,archive_Cr,archive_T,archive_freq,current_eval,F,cr,res_det ] = ...
    IMODE( x,xold, fitx,prob,bestold,bestx,archive,hist_pos,memory_size, archive_f,archive_Cr,archive_T,archive_freq, xmin, xmax,  n,...
    PopSize,  current_eval, I_fno,Printing,Max_FES, G_Max,F,cr,C,fhd,res_det)
PopSize = floor(PopSize);
vi=zeros(PopSize,n);

%% calc CR and F
mem_rand_index = ceil(memory_size * rand(PopSize, 1));
mu_sf = archive_f(mem_rand_index);
mu_cr = archive_Cr(mem_rand_index);

%% ========================= generate CR ==================================

cr = normrnd(mu_cr, 0.1);
term_pos = find(mu_cr == -1);
cr(term_pos) = 0;
cr = min(cr, 1);
cr = max(cr, 0);
%         cr=0.1.*ones(1,PopSize);
%% for generating scaling factor


F = mu_sf + 0.1 * tan(pi * (rand(1,PopSize) - 0.5));
pos = find(F <= 0);

while ~ isempty(pos)
    F(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(1,length(pos)) - 0.5));
    pos = find(F <= 0);
end

F = min(F, 1);
F=F';
[fitx,inddd]=sort(fitx);
x=x(inddd,:);
[cr,~]=sort(cr);


%% ======================== generate new x =================================
popAll = [x;archive.pop]; %% set archive
r0 = 1 : PopSize;
%% generate random integer numbers
[r1, r2,r3] = gnR1R2(PopSize, size(popAll, 1), r0);

%% mutation
bb= rand(PopSize, 1);
probiter = prob(1,:);
l2= sum(prob(1:2));
op_1 = bb <=  probiter(1)*ones(PopSize, 1);
op_2 = bb > probiter(1)*ones(PopSize, 1) &  bb <= (l2*ones(PopSize, 1)) ;
op_3 = bb > l2*ones(PopSize, 1) &  bb <= (ones(PopSize, 1)) ;

[~,inddd]=sort(fitx);

pNP = max(round(0.25* PopSize), 1); %% choose at least two best solutions
randindex = ceil(rand(1, PopSize) .* pNP); %% select from [1, 2, 3, ..., pNP]
randindex = max(1, randindex); %% to avoid the problem that rand = 0 and thus ceil(rand) = 0
phix = x(randindex, :);
vi(op_1==1,:) = x(op_1==1,:)+ F(op_1==1, ones(1, n)) .*(phix(op_1==1,:) - x(op_1==1,:) + x(r1(op_1==1), :) - popAll(r2(op_1==1), :));
vi(op_2==1,:) =  x(op_2==1,:)+ F(op_2==1, ones(1, n)) .*(phix(op_2==1,:) - x(op_2==1,:) + x(r1(op_2==1), :) - x(r3(op_2==1), :));
%% DE3
pNP = max(round(0.5 * PopSize), 2); %% choose at least two best solutions
randindex = ceil(rand(1, PopSize) .* pNP); %% select from [1, 2, 3, ..., pNP]
randindex = max(1, randindex); %% to avoid the problem that rand = 0 and thus ceil(rand) = 0
phix = x(randindex, :);
vi(op_3==1,:) = F(op_3==1, ones(1, n)).* x(r1(op_3==1), :) +F(op_3==1, ones(1, n)).* (phix(op_3==1,:) - x(r3(op_3==1), :));

%% handle boundaries
vi = han_boun(vi, xmax, xmin, x,PopSize,2);
%% crossover
if rand<0.3
    mask = rand(PopSize, n) > cr(:, ones(1, n));
    % mask is used to indicate which elements of ui comes from the parent
    rows = (1 : PopSize)'; cols = floor(rand(PopSize, 1) * n)+1; % choose one position where the element of ui doesn't come from the parent
    jrand = sub2ind([PopSize n], rows, cols); mask(jrand) = false;
    ui = vi; ui(mask) = x(mask);
else
    ui=x;
    startLoc= randi(n,PopSize,1);
    for i=1:PopSize
        l=startLoc(i);
        while (rand<cr(i) && l<n)
            l=l+1;
        end
        for j=startLoc(i) : l
            ui(i,j)= vi(i,j);
        end
    end
end
% ui = x; ui(mask) = vi(mask);
%% evaluate
% fitx_new = cec20_func(ui',I_fno);
[temp,~] = size(ui); fitx_new = zeros(temp,1);
for it1=1:temp
    fitx_new(it1) = fhd(ui(it1,:)');
end
% fitx_new = feval(fhd,ui',I_fno,C);
% fitx_new=fitx_new';
%% update FITNESS EVALUATIONS
for i = 1 : PopSize
    current_eval = current_eval + 1;
    if current_eval > Max_FES
        break;
    end
    if fitx_new(i) < bestold
        bestold = fitx_new(i);
        bestx = ui(i, :);
    end
    
end
% current_eval =current_eval+PopSize;

%% calc. imprv. for Cr and F
diff = abs(fitx - fitx_new);
I =(fitx_new < fitx);
goodCR = cr(I == 1);
goodF = F(I == 1);

%% ========================= update archive ===============================
archive = updateArchive(archive, x(I == 1, :), fitx(I == 1));
%% ==================== update Prob. of each DE ===========================
diff2 = max(0,(fitx - fitx_new))./abs(fitx);
count_S(1)=max(0,mean(diff2(op_1==1)));
count_S(2)=max(0,mean(diff2(op_2==1)));
count_S(3)=max(0,mean(diff2(op_3==1)));

%% update probs.
if count_S~=0
    prob= max(0.1,min(0.9,count_S./(sum(count_S))));
else
    prob=1/3 * ones(1,3);
end
%% ==================== update x and fitx =================================
fitx(I==1)= fitx_new(I==1);
xold(I == 1, :) = x(I == 1, :);
x(I == 1, :) = ui(I == 1, :);

%% =================== update memory cr and F =============================

if size(goodF,1)==1
    goodF=goodF';
end
if size(goodCR,1)==1
    goodCR=goodCR';
end
num_success_params = numel(goodCR);
if num_success_params > 0
    weightsDE = diff(I == 1)./ sum(diff(I == 1));
    %% for updating the memory of scaling factor
    archive_f(hist_pos) = (weightsDE' * (goodF .^ 2))./ (weightsDE' * goodF);
    
    %% for updating the memory of crossover rate
    if max(goodCR) == 0 || archive_Cr(hist_pos)  == -1
        archive_Cr(hist_pos)  = -1;
    else
        archive_Cr(hist_pos) = (weightsDE' * (goodCR .^ 2)) / (weightsDE' * goodCR);
    end
    
    hist_pos= hist_pos+1;
    if hist_pos > memory_size;  hist_pos = 1; end
else
    archive_Cr(hist_pos)=0.2;
    archive_f(hist_pos)=0.2;
    % end
end

%% sort new x, fitness
[fitx, ind]=sort(fitx);
x=x(ind,:);
xold = xold(ind,:);

%% record the best value after checking its feasiblity status
if fitx(1)<bestold  && min(x(ind(1),:))>=-100 && max(x(ind(1),:))<=100
    bestold=fitx(1);
    bestx= x(1,:);
end

if bestold<res_det(end)
res_det= [res_det ;repmat(bestold,PopSize,1)];
else
    res_det= [res_det ;repmat(res_det(end),PopSize,1)];
end
    

end
%% check to print
% if Par.Printing==1
% end

function [r1, r2,r3] = gnR1R2(NP1, NP2, r0)

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
% r1 = randperm(NP1,NP0);

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

r3= floor(rand(1, NP0) * NP1) + 1;
%for i = 1 : inf
for i = 1 : 99999999
    pos = ((r3 == r0) | (r3 == r1) | (r3==r2));
    if sum(pos)==0
        break;
    else % regenerate r2 if it is equal to r0 or r1
         r3(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
    end
    if i > 1000, % this has never happened so far
        error('Can not genrate r2 in 1000 iterations');
    end
end
end

%% ============United Multi-Operator Evolutionary AlgorithmsII ============
% Should you have any queries, please contact
% Dr. Saber Elsayed. University of New South Wales at Canberra
% s.elsayed@adfa.edu.au
% www.saberelsayd.net or
% https://sites.google.com/site/saberelsayed3/home
% =========================================================================

function x = han_boun (x, xmax, xmin, x2, PopSize,hb)
hb=randi(3);
%  hb=3;
switch hb
    case 1 % for DE
        x_L = repmat(xmin, PopSize, 1);
        pos = x < x_L;
        x(pos) = (x2(pos) + x_L(pos)) / 2;
        
        x_U = repmat(xmax, PopSize, 1);
        pos = x > x_U;
        x(pos) = (x2(pos) + x_U(pos)) / 2;
        
    case 2 
        x_L = repmat(xmin, PopSize, 1);
        pos = x < x_L;
        x_U = repmat(xmax, PopSize, 1);
        x(pos) = min(x_U(pos),max(x_L(pos),2*x_L(pos)-x2(pos)))  ;
        pos = x > x_U;
        x(pos) = max(x_L(pos),min(x_U(pos),2*x_L(pos)-x2(pos)));
        
   case 3 
        x_L = repmat(xmin, PopSize, 1);
        pos = x < x_L;
        x_U = repmat(xmax, PopSize, 1);
        x(pos) = x_L(pos)+ rand*(x_U(pos)-x_L(pos) ) ;
        pos = x > x_U;
        x(pos) = x_L(pos)+ rand*(x_U(pos)-x_L(pos));
        
end  
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
popAll = [archive.pop; pop ];
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

function [pop,fitness,Par,nfes,res_det]= APGSK_fun(pop,fitness,lu,func,Par,nfes,fhd,C,res_det)

[pop_size,problem_size]=size(pop);

KF_pool = [0.1 1.0 0.5 1.0];
KF_poool= [-0.1 -0.1 -0.1 -0.1];
KR_pool = [0.2 0.1 0.9 0.9];

max_nfes=Par.max_nfes;
All_Imp=Par.All_Imp;
KW_ind=Par.KW_ind;



if  (nfes < 0.1*max_nfes)% intial probability values
    KW_ind=[0.85 0.05 0.05 0.05];
    K_rand_ind=rand(pop_size, 1);
    K_rand_ind(K_rand_ind>sum(KW_ind(1:3))&K_rand_ind<=sum(KW_ind(1:4)))=4;
    K_rand_ind(K_rand_ind>sum(KW_ind(1:2))&K_rand_ind<=sum(KW_ind(1:3)))=3;
    K_rand_ind(K_rand_ind>KW_ind(1)&K_rand_ind<=sum(KW_ind(1:2)))=2;
    K_rand_ind(K_rand_ind>0&K_rand_ind<=KW_ind(1))=1;
    KF=KF_pool(K_rand_ind)';
    KR=KR_pool(K_rand_ind)';
else %% updaing probability values
    if isempty(KW_ind)
        KW_ind=[0.85 0.05 0.05 0.05];
     end
    KW_ind=0.95*KW_ind+0.05*All_Imp;
    KW_ind=KW_ind./sum(KW_ind);
    K_rand_ind=rand(pop_size, 1);
    K_rand_ind(K_rand_ind>sum(KW_ind(1:3))&K_rand_ind<=sum(KW_ind(1:4)))=4;
    K_rand_ind(K_rand_ind>sum(KW_ind(1:2))&K_rand_ind<=sum(KW_ind(1:3)))=3;
    K_rand_ind(K_rand_ind>KW_ind(1)&K_rand_ind<=sum(KW_ind(1:2)))=2;
    K_rand_ind(K_rand_ind>0&K_rand_ind<=KW_ind(1))=1;
    % KF=KF_pool(K_rand_ind)';
    KR=KR_pool(K_rand_ind)';
    if rand>=0.1 && nfes>0.5*max_nfes
        KF=KF_pool(K_rand_ind)';
    else
        KF=KF_poool(K_rand_ind)';
    end
    
end

%%% Junior and Senior Gaining-Sharing phases %%%%%
%D_Gained_Shared_Junior=ceil((problem_size)*(1-nfes / max_nfes).^K);
if rand >(nfes / max_nfes)
    D_Gained_Shared_Junior=ceil((1)* round((problem_size)* ((1-nfes / max_nfes).^((0.5)))));
    
else
    D_Gained_Shared_Junior=ceil((1)* round((problem_size)* ((1-nfes / max_nfes).^((2)))));
end

D_Gained_Shared_Senior=problem_size-D_Gained_Shared_Junior;

[valBest, indBest] = sort(fitness, 'ascend');
[Rg1, Rg2, Rg3] = Gained_Shared_Junior_R1R2R3(indBest);

[R1, R2, R3] = Gained_Shared_Senior_R1R2R3(indBest);
R01=1:pop_size;
Gained_Shared_Junior=zeros(pop_size, problem_size);
ind1=fitness(R01)>fitness(Rg3);

if(sum(ind1)>0)
    Gained_Shared_Junior (ind1,:)= pop(ind1,:) + KF(ind1, ones(1,problem_size)).* (pop(Rg1(ind1),:) - pop(Rg2(ind1),:)+pop(Rg3(ind1), :)-pop(ind1,:)) ;
end
ind1=~ind1;
if(sum(ind1)>0)
    Gained_Shared_Junior(ind1,:) = pop(ind1,:) + KF(ind1, ones(1,problem_size)) .* (pop(Rg1(ind1),:) - pop(Rg2(ind1),:)+pop(ind1,:)-pop(Rg3(ind1), :)) ;
end
R0=1:pop_size;
Gained_Shared_Senior=zeros(pop_size, problem_size);
ind=fitness(R0)>fitness(R2);
if(sum(ind)>0)
    Gained_Shared_Senior(ind,:) = pop(ind,:) + KF(ind, ones(1,problem_size)) .* (pop(R1(ind),:) - pop(ind,:) + pop(R2(ind),:) - pop(R3(ind), :)) ;
end
ind=~ind;
if(sum(ind)>0)
    Gained_Shared_Senior(ind,:) = pop(ind,:) + KF(ind, ones(1,problem_size)) .* (pop(R1(ind),:) - pop(R2(ind),:) + pop(ind,:) - pop(R3(ind), :)) ;
end
Gained_Shared_Junior = boundConstraint(Gained_Shared_Junior, pop, lu);
Gained_Shared_Senior = boundConstraint(Gained_Shared_Senior, pop, lu);


D_Gained_Shared_Junior_mask=rand(pop_size, problem_size)<=(D_Gained_Shared_Junior(:, ones(1, problem_size))./problem_size); % mask is used to indicate which elements of will be gained
D_Gained_Shared_Senior_mask=~D_Gained_Shared_Junior_mask;

D_Gained_Shared_Junior_rand_mask=rand(pop_size, problem_size)<=KR(:,ones(1, problem_size));
D_Gained_Shared_Junior_mask=and(D_Gained_Shared_Junior_mask,D_Gained_Shared_Junior_rand_mask);

D_Gained_Shared_Senior_rand_mask=rand(pop_size, problem_size)<=KR(:,ones(1, problem_size));
D_Gained_Shared_Senior_mask=and(D_Gained_Shared_Senior_mask,D_Gained_Shared_Senior_rand_mask);
ui=pop;

ui(D_Gained_Shared_Junior_mask) = Gained_Shared_Junior(D_Gained_Shared_Junior_mask);
ui(D_Gained_Shared_Senior_mask) = Gained_Shared_Senior(D_Gained_Shared_Senior_mask);

[temp,~] = size(ui); children_fitness = zeros(temp,1);
for it1=1:temp
    children_fitness(it1) = fhd(ui(it1,:)');
end
% 
% children_fitness = feval(fhd, ui', func,C);
% children_fitness = children_fitness';
for i = 1 : pop_size
    nfes = nfes + 1;
    if nfes > max_nfes;
        break;
    end
    if children_fitness(i) < valBest(1)
        valBest(1) = children_fitness(i);
        bsf_solution = ui(i, :);
    end
    
end

%%%%  Calculate the improvemnt of each settings %%%
dif = abs(fitness - children_fitness);
%% I == 1: the parent is better; I == 2: the offspring is better
Child_is_better_index = (fitness > children_fitness);
dif_val = dif(Child_is_better_index == 1);
All_Imp=zeros(1,4);% (1,4) delete for 4 cases
for i=1:4
    if(sum(and(Child_is_better_index,K_rand_ind==i))>0)
        All_Imp(i)=sum(dif(and(Child_is_better_index,K_rand_ind==i)));
    else
        All_Imp(i)=0;
    end
end

if(sum(All_Imp)~=0)
    All_Imp=All_Imp./sum(All_Imp);
    [temp_imp,Imp_Ind] = sort(All_Imp);
    for imp_i=1:length(All_Imp)-1
        All_Imp(Imp_Ind(imp_i))=max(All_Imp(Imp_Ind(imp_i)),0.05);
    end
    All_Imp(Imp_Ind(end))=1-sum(All_Imp(Imp_Ind(1:end-1)));
else
    Imp_Ind=1:length(All_Imp);
    All_Imp(:)=1/length(All_Imp);
end
[fitness, Child_is_better_index] = min([fitness, children_fitness], [], 2);

pop(Child_is_better_index == 2, :) = ui(Child_is_better_index == 2, :);



Par.All_Imp=All_Imp;
Par.KW_ind=KW_ind;

bestold=min(fitness);
if bestold<res_det(end)
res_det= [res_det ;repmat(bestold,pop_size,1)];
else
    res_det= [res_det ;repmat(res_det(end),pop_size,1)];
end
end

function [R1, R2, R3] = Gained_Shared_Junior_R1R2R3(indBest)


pop_size = length(indBest);
R0=1:pop_size;
R1=[];
R2=[];
R3=[];

for i=1:pop_size
    ind=find(indBest==i);
    if(ind==1)% best
    R1(i)=indBest(2);
    R2(i)=indBest(3);
    elseif(ind==pop_size)% worst
    R1(i)=indBest(pop_size-2);
    R2(i)=indBest(pop_size-1);
    else
    R1(i)=indBest(ind-1);
    R2(i)=indBest(ind+1);
    end
end

R3 = floor(rand(1, pop_size) * pop_size) + 1;

for i = 1 : 99999999
    pos = ((R3 == R2) | (R3 == R1) | (R3 == R0));
    if sum(pos)==0
        break;
    else % regenerate r2 if it is equal to r0 or r1
        R3(pos) = floor(rand(1, sum(pos)) * pop_size) + 1;
    end
    if i > 1000, % this has never happened so far
        error('Can not genrate R3 in 1000 iterations');
    end
end

end

function [R1, R2, R3] = Gained_Shared_Senior_R1R2R3(indBest)


pop_size = length(indBest);

R1=indBest(1:round(pop_size*0.1));
R1rand = ceil(length(R1) * rand(pop_size, 1));
R1 = R1(R1rand);

R2=indBest(round(pop_size*0.1)+1:round(pop_size*0.9));
R2rand = ceil(length(R2) * rand(pop_size, 1));
R2 = R2(R2rand);

R3=indBest(round(pop_size*0.9)+1:end);
R3rand = ceil(length(R3) * rand(pop_size, 1));
R3 = R3(R3rand);

end


%This function is used for bound checking
function vi = boundConstraint (vi, pop, lu)

% if the boundary constraint is violated, set the value to be the middle
% of the previous value and the bound
%

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
