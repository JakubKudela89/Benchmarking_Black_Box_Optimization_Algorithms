function [minima, xatmin, history] = alg_cobyla(Problem, opts, bounds)
    tic;
    minima = inf; xatmin = nan(opts.dimension, 1); fcount = 0; iter = 0;
    options = struct();
    options.ftarget = opts.globalmin + opts.tolabs;
    history = [-1, 0, inf, 0];
    while fcount < opts.maxevals && minima-opts.globalmin > opts.tol
        x0 = bounds(:, 1) + (bounds(:, 2) - bounds(:, 1)).*rand(opts.dimension, 1);
        [x, fval, ~, output] = cobyla(Problem.f, x0, [], [], [], [], bounds(:, 1), bounds(:, 2), options);
        fcount = fcount + output.funcCount;
        if fval < minima
            minima = fval;
            xatmin = x;
        end
        iter = iter + 1;
        if history(end, 3) > minima || history(end, 1) == -1
            history(end + 1, :) = [iter, fcount, minima,  toc]; %#ok<*AGROW>
            fprintf("Iter:%5i\tf_min:%15.10f\ttime(s):%10.5f\tfn evals:%9i\n",[history(end,1),history(end,3),history(end,4),history(end,2)]);
        end
    end
    if history(end, 1) ~= iter
        history(end + 1, :) = [iter, fcount, minima,  toc];
    end
    history(1, :) = [];
end