function [Fmin, Xmin, h] = alg_PRS(ObjFun, opts, Bounds)
    tic
    [it, fcount] = deal(0);
    [Fmin, Fmin_old, Xmin] = deal(inf);
    for i = 1:opts.maxevals
        x_new = abs(Bounds(:, 2) - Bounds(:, 1)).*rand(opts.dimension, 1) + Bounds(:, 1);
        fcount = fcount + 1; Fmin = min([feval(ObjFun.f, x_new), Fmin]);
    
        if Fmin < Fmin_old
            it = it + 1;
            fprintf('Iter: %4i   f_min: %15.10f    time(s): %10.05f    fn evals: %8i\n', it, Fmin, toc, fcount);
            h(it, 1) = it; h(it, 2) = fcount; h(it, 3) = Fmin; h(it, 4) = toc; %#ok<*AGROW>
            Fmin_old = Fmin;
            Xmin = x_new;
        end
        if abs(opts.globalmin - Fmin) <= opts.tol || fcount >= opts.maxevals || toc > 600
            it = it + 1;
            fprintf('Iter: %4i   f_min: %15.10f    time(s): %10.05f    fn evals: %8i\n', it, Fmin, toc, fcount);
            h(it, 1) = it; h(it, 2) = fcount; h(it, 3) = Fmin; h(it, 4) = toc;
            break
        end
    end
return