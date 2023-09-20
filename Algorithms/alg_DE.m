function [minima, xatmin, history] = alg_DE(Problem, opts, bounds)
tic;
pop_size=opts.population;
problem_size=opts.dimension;
max_nfes = opts.maxevals;
fhd = Problem.f;

lb = bounds(:,1)';
ub = bounds(:,2)';
C = 1;
F =0.50*ones(pop_size,problem_size);
Cr=0.90*ones(pop_size,problem_size);
Runs = 1;
%lu = [lb * ones(1, problem_size); ub * ones(1, problem_size)];
lu = [lb;ub];

for func_no = 1
    for run_id = 1 : Runs
        
        %% Initialize the main population
        bsf_fit_var = Inf; xatmin = zeros(problem_size,1); history = []; nfes = 0;
        while bsf_fit_var == Inf && nfes < max_nfes
            popold = repmat(lu(1, :), pop_size, 1) + rand(pop_size, problem_size) .* (repmat(lu(2, :) - lu(1, :), pop_size, 1));
            %popold = init_val;
            pop = popold; % the old population becomes the current population
            
            %fitness = feval(fhd,pop',func_no,C);
            %fitness = fhd(pop');
            for i=1:pop_size
                fitness(i,1)=feval(fhd,pop(i,:)');
            end
            %fitness = init_val;
            %fitness = fitness';
            nfes = nfes + pop_size;
            [bsf_fit_var,minpos] = min(fitness);
            best_ind = pop(minpos,:);
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

        %% main loop
        while nfes < max_nfes
            pop = popold; % the old population becomes the current population
            R = Gen_R(pop_size,3);
            r1=R(:,2);
            r2=R(:,3);
            r3=R(:,4);
            
            vi = pop(r1, :) + F .* (pop(r2, :) - pop(r3, :));
            
            vi = boundConstraint(vi, pop, lu);
            
            mask = rand(pop_size, problem_size) > Cr; % mask is used to indicate which elements of ui comes from the parent
            rows = (1 : pop_size)'; cols = floor(rand(pop_size, 1) * problem_size)+1; % choose one position where the element of ui doesn't come from the parent
            jrand = sub2ind([pop_size problem_size], rows, cols); mask(jrand) = false;
            ui = vi; ui(mask) = pop(mask);
            for i=1:pop_size
                children_fitness(i,1) = fhd(ui(i,:)');
            end
            for i = 1 : pop_size
                nfes = nfes + 1;
                if children_fitness(i) < bsf_fit_var
                    bsf_fit_var = children_fitness(i);
                    best_ind = ui(i,:);
                end
                if nfes > max_nfes; break; end
            end
            
            [fitness, I] = min([fitness, children_fitness], [], 2);
            
            popold = pop;
            popold(I == 2, :) = ui(I == 2, :);
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
        

    end %% end 1 run
   
end %% end 1 function run
minima = bsf_fit_var;
xatmin = best_ind';
end


function R = Gen_R(NP_Size,N)

% Gen_R generate N column vectors r1, r2, ..., rN of size NP_Size
%    R's elements are choosen from {1, 2, ..., NP_Size} & R(j,i) are unique per row

% Call:
%    [R] = Gen_R(NP_Size)   % N is set to be 1;
%    [R] = Gen_R(NP_Size,N) 
%
% Version: 0.1  Date: 2018/02/01
% Written by Anas A. Hadi (anas1401@gmail.com)


R(1,:)=1:NP_Size;

for i=2:N+1
    
    R(i,:) = ceil(rand(NP_Size,1) * NP_Size);
    
    flag=0;
    while flag ~= 1
        pos = (R(i,:) == R(1,:));
        for w=2:i-1
            pos=or(pos,(R(i,:) == R(w,:)));
        end
        if sum(pos) == 0
            flag=1;
        else
            R(i,pos)= floor(rand(sum(pos),1 ) * NP_Size) + 1;
        end
    end
end

R=R';

end

%This function is used for bound checking
function vi = boundConstraint (vi, pop, lu)

% if the boundary constraint is violated, set the value to be the middle
% of the previous value and the bound

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
