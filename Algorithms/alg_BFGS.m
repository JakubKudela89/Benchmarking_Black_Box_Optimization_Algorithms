function [minima, xatmin, history] = alg_BFGS(Problem, opts, bounds)
tic;
minima = []; xatmin = []; history = [];
pop_size=opts.population;
problem_size=opts.dimension;
max_nfes = opts.maxevals;
fhd = Problem.f;

lb = bounds(:,1);
ub = bounds(:,2);
optimopts = optimset('TolX', 1e-8, 'MaxFunEvals', 1e3*problem_size,'MaxIter', 1e3*problem_size, 'Display','off','Algorithm','interior-point','Hessian','bfgs');
nfes = 0; bsf_fit_var = Inf; iter = 0;
while nfes < max_nfes && bsf_fit_var-opts.globalmin > opts.tolabs && opts.time > toc
    x0 = lb + (ub-lb).*rand(problem_size,1);
    iter = iter + 1;
    nfes = nfes + 1;
    fx0 = fhd(x0);
    if fx0 < bsf_fit_var
        bsf_fit_var = fx0;
        xatmin = x0;
    end
    history(iter,1) = iter;
    history(iter,2) = nfes;
    history(iter,3) = bsf_fit_var;
    history(iter,4) = toc;
    while fx0 == Inf && nfes <= max_nfes
        x0 = lb + (ub-lb).*rand(problem_size,1);
        fx0 = fhd(x0);
        if fx0 < bsf_fit_var
            bsf_fit_var = fx0;
            xatmin = x0;
        end
        nfes = nfes + 1;
        if mod(nfes,1000) == 0
            iter = iter + 1;
            history(iter,1) = iter;
            history(iter,2) = nfes;
            history(iter,3) = bsf_fit_var;
            history(iter,4) = toc;
        end
    end

    if nfes > max_nfes
        break;
    end

    [x,fval,exitflag,output] = fmincon(fhd,x0,[],[],[],[],lb,ub,[],optimopts);
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
