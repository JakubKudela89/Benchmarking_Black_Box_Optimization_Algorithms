function [Fmin, Xmin, historys] = alg_surrogateopt(ObjFun, opts, bounds)
    historys = [-1, inf, 0, 0];
    
    options = optimoptions('surrogateopt', 'MaxFunctionEvaluations', ...
        opts.maxevals, 'MinSampleDistance', 1e-15, 'Display', 'off',...
        'MaxTime', 600, 'ObjectiveLimit', opts.globalmin + 1e-8, 'PlotFcn',...
        [], 'OutputFcn', @listout);
    [Xmin, Fmin, ~, output] = surrogateopt(@(x) feval(ObjFun.f, x'), bounds(:, 1), bounds(:, 2), options);
    historys(end + 1, :) = [historys(end, 1) + 1, output.funccount, Fmin,  output.elapsedtime];
    historys(1, :) = [];
    function stop = listout(~, optimValues, state)
    stop = false;
    switch state
        case 'iter'
            if isempty(optimValues.fval)
                TT = inf;
            else
                TT = optimValues.fval;
            end
            if historys(end, 3) > TT || historys(end, 1) == -1
                fprintf('Iter: %4i   f_min: %15.10f    time(s): %10.05f    fn evals: %8i\n', optimValues.iteration, optimValues.fval, optimValues.elapsedtime, optimValues.funccount);
                historys(end + 1, :) = [optimValues.iteration, optimValues.funccount, TT,  optimValues.elapsedtime];
            end
    end
    end
end