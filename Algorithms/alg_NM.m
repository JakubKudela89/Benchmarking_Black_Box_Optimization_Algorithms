function [minima, xatmin, history] = alg_NM(Problem, opts, bounds)
tic;
minima = []; xatmin = []; history = [];
pop_size=opts.population;
problem_size=opts.dimension;
max_nfes = opts.maxevals;
fhd = Problem.f;

lb = bounds(:,1);
ub = bounds(:,2);
optimopts = optimset('TolFun', 1e-8 , 'TolX', 1e-8, 'MaxFunEvals', 1e3*problem_size,'MaxIter', 1e3*problem_size, 'Display','off');
nfes = 0; bsf_fit_var = Inf; iter = 0;
while nfes < max_nfes && bsf_fit_var-opts.globalmin > opts.tolabs
    x0 = lb + (ub-lb).*rand(problem_size,1);
    [x,fval,exitflag,output] = fminsearch(fhd,x0,optimopts);
    nfes = nfes + output.funcCount;
    if fval < bsf_fit_var
        bsf_fit_var = fval;
        xatmin = x;
    end
    iter = iter + 1;
    history(iter,1) = iter;
    history(iter,2) = nfes;
    history(iter,3) = bsf_fit_var;
    history(iter,4) = toc;
    if opts.showits && (~mod(iter,50) || iter == 1)
        fprintf("Iter:%5i\tf_min:%15.10f\ttime(s):%10.5f\tfn evals:%9i\n",[history(iter,1),history(iter,3),history(iter,4),history(iter,2)]);
        %Iter:   43   f_min:  -78.9844713606    time(s):    0.34048    fn evals:     1723
    end
end
minima = bsf_fit_var;

end
