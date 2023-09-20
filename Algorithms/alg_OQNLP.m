function [Fmin, Xmin, h] = alg_OQNLP(ObjFun, opts, Bounds) %#ok<*INUSD>
    Fmin = []; Xmin = []; h = [];
    disp('The "OQNLP" algorithm requires a separate installation and a licence. ');
    disp('Please contact: https://tomopt.com/')
return