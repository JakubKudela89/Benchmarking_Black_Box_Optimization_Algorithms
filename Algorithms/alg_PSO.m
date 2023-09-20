function [minima, xatmin, history] = alg_PSO(Problem, opts, bounds)
tic;
ps=opts.population;
D=opts.dimension;
maxFES = opts.maxevals;
fhd = Problem.f;

VRmin = bounds(:,1)';
VRmax = bounds(:,2)';
MaxGen = ceil(maxFES/ps);
me=MaxGen;
Particle_Number = ps;

cc=[2 2];   %acceleration constants
iwt=0.9-(1:me).*(0.5./me);
% iwt=0.5.*ones(1,me);
if length(VRmin)==1
    VRmin=repmat(VRmin,1,D);
    VRmax=repmat(VRmax,1,D);
end
mv=0.5*(VRmax-VRmin);
VRmin=repmat(VRmin,ps,1);
VRmax=repmat(VRmax,ps,1);
Vmin=repmat(-mv,ps,1);
Vmax=-Vmin;

current_eval = 0; run_funcvals = []; 
bsf_fit_var = Inf; xatmin = zeros(D,1); history = [];
while bsf_fit_var == Inf && current_eval < maxFES
    pos=VRmin+(VRmax-VRmin).*rand(ps,D);
    
    e = zeros(1,ps);
    %e = [fhd(pos(1:ps,:))]';
    for i=1:Particle_Number
        e(i)=feval(fhd,pos(i,:)');
    end
    
    fitcount=ps;
    vel=Vmin+2.*Vmax.*rand(ps,D);%initialize the velocity of the particles
    pbest=pos;
    pbestval=e; %initialize the pbest and the pbest's fitness value
    [gbestval,gbestid]=min(pbestval);
    gbest=pbest(gbestid,:);%initialize the gbest and the gbest's fitness value
    gbestrep=repmat(gbest,ps,1);
    current_eval = current_eval + Particle_Number;
    bsf_fit_var = gbestval;
end
iter = 1;
history(iter,1) = 1;
history(iter,2) = current_eval;
history(iter,3) = bsf_fit_var;
history(iter,4) = toc;
if opts.showits
    fprintf("Iter:%5i\tf_min:%15.10f\ttime(s):%10.5f\tfn evals:%9i\n",[history(iter,1),history(iter,3),history(iter,4),history(iter,2)]);
    %Iter:   43   f_min:  -78.9844713606    time(s):    0.34048    fn evals:     1723
end

for i=2:me
        aa=cc(1).*rand(ps,D).*(pbest-pos)+cc(2).*rand(ps,D).*(gbestrep-pos);
        vel=iwt(i).*vel+aa;
        vel=(vel>Vmax).*Vmax+(vel<=Vmax).*vel;
        vel=(vel<Vmin).*Vmin+(vel>=Vmin).*vel;
        pos=pos+vel;
        pos=((pos>=VRmin)&(pos<=VRmax)).*pos...
            +(pos<VRmin).*(VRmin+0.25.*(VRmax-VRmin).*rand(ps,D))+(pos>VRmax).*(VRmax-0.25.*(VRmax-VRmin).*rand(ps,D));
        for ii=1:Particle_Number
            e(ii)=feval(fhd,pos(ii,:)');
        end        
       %e = zeros(1,ps);
        %e = [fhd(pos(1:ps,:)')];

        fitcount=fitcount+ps;
        tmp=(pbestval<e);
        temp=repmat(tmp',1,D);
        pbest=temp.*pbest+(1-temp).*pos;
        pbestval=tmp.*pbestval+(1-tmp).*e;%update the pbest
        [gbestval,tmp]=min(pbestval);
        gbest=pbest(tmp,:);
        gbestrep=repmat(gbest,ps,1);%update the gbest
        current_eval = current_eval + Particle_Number;
        
        bsf_fit_var = gbestval;
        iter = iter + 1;
        history(iter,1) = iter;
        history(iter,2) = current_eval;
        history(iter,3) = bsf_fit_var;
        history(iter,4) = toc;       
        if current_eval > maxFES
            break
        end
        if opts.showits && ~mod(iter,50) %only print every 50 iterations
            fprintf("Iter:%5i\tf_min:%15.10f\ttime(s):%10.5f\tfn evals:%9i\n",[history(iter,1),history(iter,3),history(iter,4),history(iter,2)]);
            %Iter:   43   f_min:  -78.9844713606    time(s):    0.34048    fn evals:     1723
        end
        if bsf_fit_var-opts.globalmin < opts.tolabs
            break
        end
end

minima = bsf_fit_var;
xatmin = gbest';

end


