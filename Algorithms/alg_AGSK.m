function [minima, xatmin, history] = alg_AGSK(Problem, opts, bounds)
% original citation: 
% A. W. Mohamed, A. A. Hadi, A. K. Mohamed, and N. H. Awad,
% "Evaluating the performance of adaptive gainingsharing knowledge
% based algorithm on cec 2020 benchmark problems," in 2020 IEEE
% Congress on Evolutionary Computation (CEC). IEEE, 2020, pp. 1–8.
% modified by Jakub Kudela
% =========================================================================
tic;
fhd=Problem.f; 
lu = bounds';
problem_size = opts.dimension;

max_nfes = opts.maxevals;

% switch problem_size
%     case 10
%         max_nfes=200000;
%     case 20
%         max_nfes=1000000;
%     otherwise
%         disp('Error..')
% end

% for n=0:15
%     RecordFEsFactor(n+1) = round(problem_size^((n/5)-3)*max_nfes);
% end
% progress = numel(RecordFEsFactor);
% val_2_reach = 10^(-8);
% max_region = 100.0;
% min_region = -100.0;
% lu = [-100 * ones(1, problem_size); 100 * ones(1, problem_size)];
analysis= zeros(10,6);

KF_pool = [0.1 1.0 0.5 1.0];
KR_pool = [0.2 0.1 0.9 0.9];

for func = 1
    %% Record the best results
%     outcome = [];
%     fprintf('\n-------------------------------------------------------\n')
%     fprintf('Function = %d, Transformation = (%d%d%d), Dimension size = %d\n', func,  C(m,:), problem_size)
%     allerrorvals = zeros(progress, Runs);
%     allsolutions = zeros(problem_size, Runs);
    
    %     you can use parfor if you have MATLAB Parallel Computing Toolbox
    for run_id = 1 : 1
        pop_size=100;
        max_pop_size = pop_size;
        min_pop_size = 12;

        %% Initialize the main population
        nfes = 0;
        bsf_fit_var = Inf;
        while bsf_fit_var == Inf && nfes < max_nfes
            popold = repmat(lu(1, :), pop_size, 1) + rand(pop_size, problem_size) .* (repmat(lu(2, :) - lu(1, :), pop_size, 1));
            pop = popold; % the old population becomes the current population
            [temp,~] = size(pop); fitness = zeros(temp,1);
            for it1=1:temp
                fitness(it1,1) = fhd(pop(it1,:)');
            end
    %         fitness = feval(fhd, pop', func, C(m,:));
    %         fitness = fitness';
    
    
            bsf_solution = zeros(1, problem_size);
    
            %%%%%%%%%%%%%%%%%%%%%%%% for out
            for i = 1 : pop_size
                nfes = nfes + 1;
                if fitness(i) <= bsf_fit_var
                    bsf_fit_var = fitness(i);
                    bsf_solution = pop(i, :);            
                end
                if nfes == max_nfes
                    break
                end
            end
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
%         if(nfes>=Run_RecordFEsFactor(1))
%             run_funcvals = [run_funcvals; bsf_fit_var];
%             Run_RecordFEsFactor(1)=[];
%         end

        %%POSSIBLE VALUES FOR KNOWLEDGE RATE K%%%%
        K=[];
        KF=[];
        KR=[];
        Kind=rand(pop_size, 1);
        %%%%%%%%%%%%%%%%%%%K uniform rand (0,1) with prob 0.5 and unifrom integer [1,20] with prob 0.5
                           K(Kind<0.5,:)= rand(sum(Kind<0.5), 1);
                           K(Kind>=0.5,:)=ceil(20 * rand(sum(Kind>=0.5), 1));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        g=0;
        %% main loop

        KW_ind=[];
        All_Imp=zeros(1,4);

        while nfes < max_nfes
            g=g+1;

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
                KF=KF_pool(K_rand_ind)';
                KR=KR_pool(K_rand_ind)';

            end

            %%% Junior and Senior Gaining-Sharing phases %%%%%
            D_Gained_Shared_Junior=ceil((problem_size)*(1-nfes / max_nfes).^K);
            D_Gained_Shared_Senior=problem_size-D_Gained_Shared_Junior;
            pop = popold; % the old population becomes the current population

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
                children_fitness(it1,1) = fhd(ui(it1,:)');
            end
%             children_fitness = feval(fhd, ui', func, C(m,:));
%             children_fitness = children_fitness';
            
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

            popold = pop;
            popold(Child_is_better_index == 2, :) = ui(Child_is_better_index == 2, :);
            
            %%% Updating the record for best solutions %%%
            for i = 1 : pop_size
                nfes = nfes + 1;
                if fitness(i) <= bsf_fit_var
                    bsf_fit_var = fitness(i);
                    bsf_solution = popold(i, :);
                end
                if nfes == max_nfes; break; end
            end
            
%             if(nfes>=Run_RecordFEsFactor(1))
%                 run_funcvals = [run_funcvals; bsf_fit_var];
%                 Run_RecordFEsFactor(1)=[];
%             end
            
            %% for resizing the population size %%%%
            plan_pop_size = round((min_pop_size - max_pop_size)* ((nfes / max_nfes).^((1-nfes / max_nfes)))  + max_pop_size);

            if pop_size > plan_pop_size
                reduction_ind_num = pop_size - plan_pop_size;
                if pop_size - reduction_ind_num <  min_pop_size; reduction_ind_num = pop_size - min_pop_size;end

                pop_size = pop_size - reduction_ind_num;
                for r = 1 : reduction_ind_num
                    [valBest indBest] = sort(fitness, 'ascend');
                    worst_ind = indBest(end);
                    popold(worst_ind,:) = [];
                    pop(worst_ind,:) = [];
                    fitness(worst_ind,:) = [];
                    K(worst_ind,:)=[];
                end
            end
            iter = iter + 1;
            history(iter,1) = iter;
            history(iter,2) = nfes;
            history(iter,3) = bsf_fit_var;
            history(iter,4) = toc;  
            if opts.showits && ~mod(iter,50) 
                fprintf("Iter:%5i\tf_min:%15.10f\ttime(s):%10.5f\tfn evals:%9i\n",[history(iter,1),history(iter,3),history(iter,4),history(iter,2)]);
                %Iter:   43   f_min:  -78.9844713606    time(s):    0.34048    fn evals:     1723
            end
            if bsf_fit_var-opts.globalmin < opts.tolabs
                break
            end
        end     
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end %% end 1 run
    
end %% end 1 function run

minima = bsf_fit_var;
xatmin = bsf_solution';

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

R1=indBest(1:round(pop_size*0.05));
R1rand = ceil(length(R1) * rand(pop_size, 1));
R1 = R1(R1rand);

R2=indBest(round(pop_size*0.05)+1:round(pop_size*0.95));
R2rand = ceil(length(R2) * rand(pop_size, 1));
R2 = R2(R2rand);

R3=indBest(round(pop_size*0.95)+1:end);
R3rand = ceil(length(R3) * rand(pop_size, 1));
R3 = R3(R3rand);

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