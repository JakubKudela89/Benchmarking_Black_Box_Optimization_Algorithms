function [minima, xatmin, history] = alg_HSES(Problem, opts, bounds)
% original citation: 
% G. Zhang and Y. Shi, "Hybrid sampling evolution strategy for solving
% single objective bound constrained problems," in 2018 IEEE Congress
% on Evolutionary Computation (CEC). IEEE, 2018, pp. 1–7.
% modified by Jakub Kudela
% =========================================================================
tic;
VRmin = bounds(:,1)';
VRmax = bounds(:,2)';

D=opts.dimension;
Max_FEs = opts.maxevals;
fhd = Problem.f;

runs=1;

bsf_fit_var = Inf; xatmin = zeros(D,1); history = []; FEs = 0;

for i=1:1
    for jj=1:runs
        total=200;
         mu=100;
%          VRmin=repmat(Xmin,total,1);
%          VRmax=repmat(Xmax,total,1);
         while bsf_fit_var == Inf && FEs < Max_FEs
         posinitial=VRmin+(VRmax-VRmin).*rand(total,D);
         e = zeros(total,1);
         for it1=1:total
            e(it1) = fhd(posinitial(it1,:)');
         end
         %e=fhd(posinitial);
         [bsf_fit_var,minpos] = min(e);
         FEs=FEs+total;
         end
         iter = 1;
        history(iter,1) = 1;
        history(iter,2) = FEs;
        history(iter,3) = bsf_fit_var;
        history(iter,4) = toc;
        if opts.showits
            fprintf("Iter:%5i\tf_min:%15.10f\ttime(s):%10.5f\tfn evals:%9i\n",[history(iter,1),history(iter,3),history(iter,4),history(iter,2)]);
            %Iter:   43   f_min:  -78.9844713606    time(s):    0.34048    fn evals:     1723
        end
         xatmin = posinitial(minpos,:)';
         weights=log(mu+1/2)-log(1:mu)';
         weights=weights/sum(weights);
         [a1,a2]=sort(e);
         bestval=a1(1);
          bestvec=posinitial(a2(1),:);
         pos=posinitial(a2(1:mu),:);
         meanval=mean(pos);
         stdval=std(pos);
         for k=1:total
          pos(k,:)=meanval+stdval.*randn(1,D);
         end
         for k=1:total
           for j=1:D
             if pos(k,j)>100
                 pos(k,j)=meanval(j)+stdval(j).*randn;
             elseif pos(k,j)<-100
                 pos(k,j)=meanval(j)+stdval(j).*randn;
             end
           end
         end
         cc1=0;
        for kk=1:100
             %e=feval(fhd,pos',vara(:));
             %e=fhd(pos);
             e = zeros(total,1);
             for it1=1:total
                e(it1) = fhd(pos(it1,:)');
             end
             FEs=FEs+total;        
             [a1,a2]=sort(e);
             if a1(1)<bestval
                 bestval=a1(1);
                 bestvec=pos(a2(1),:);
             end
             xatmin = bestvec;
            iter = iter + 1;
            bsf_fit_var = bestval;
            history(iter,1) = iter;
            history(iter,2) = FEs;
            history(iter,3) = bsf_fit_var;
            history(iter,4) = toc;        
            if opts.showits && ~mod(iter,50) 
                fprintf("Iter:%5i\tf_min:%15.10f\ttime(s):%10.5f\tfn evals:%9i\n",[history(iter,1),history(iter,3),history(iter,4),history(iter,2)]);
                %Iter:   43   f_min:  -78.9844713606    time(s):    0.34048    fn evals:     1723
            end
            if bsf_fit_var-opts.globalmin < opts.tolabs
                break
            end

             newpos=pos(a2(1:mu),:);
             meanval=(newpos(:,1:D)'*weights)';
             stdval=1*std(newpos);
             FV(kk)=a1(1);
             if kk>30 
               if mod(kk,20)==0
                  [aa1,aa2]=min(FV);
                  if aa2<kk-20
                     cc1=1;
                  end
               end
             end
             for k=1:total
                 if cc1==1      %kk>300
                    a=0.96*randn(1,D);
                 else
                     a=randn(1,D);
                 end
                 pos(k,:)=meanval+stdval.*a;
             end
              for k=1:total
                 for j=1:D
                    if pos(k,j)>100
                       pos(k,j)=mod(pos(k,j),100);
                    elseif pos(k,j)<-100
                       pos(k,j)=mod(pos(k,j),-100);
                    end
                 end
              end
        end
        
        previousbest=a1(1);
              if D<=30
                  Times=2;
             else
                  Times=1;
              end
             arfitnessbest=bestval.*ones(1,Times);
             xvalbest=repmat(bestvec',1,Times);
              N=D;
             for kkk=1:Times
             sigma=0.2;
             stopfitness=1e-8;
             if D<=30
                   stopeval=Max_FEs/1;
             else
                  stopeval=Max_FEs/2;
             end
             lambda=floor(3*log(N))+80;
             mu=lambda/2;
             weights=log(mu+1/2)-log(1:mu)';
             mu=floor(mu);
             weights=weights/sum(weights);
             mueff=sum(weights)^2/sum(weights.^2);
             cc=(4+mueff/N)/(N+4+2*mueff/N);
             cs=(mueff+2)/(N+mueff+5);
             c1=2/((N+1.3)^2+mueff);
             cmu=2*(mueff-2+1/mueff)/((N+2)^2+2*mueff/2);
             damps=1+2*max(0,sqrt((mueff-1)/(N+1))-1)+cs;           
              pc=zeros(N,1);
              ps=zeros(N,1);
              B=eye(N);
              DD=eye(N);
              C=B*DD*(B*DD)';
              eigenval=0;
             chiN=N^0.5*(1-1/(4*N)+1/(21*N^2));                 %unknown
             counteval=0;
             xmean=bestvec';
             while counteval<stopeval && FEs < Max_FEs
                 for k=1:lambda
                     arz(:,k)=randn(N,1);
                     arxx(:,k)=xmean+1*sigma*B*DD*arz(:,k);
                  for jjj=1:N
                      if real(arxx(jjj,k))>100
                         arxx(jjj,k)=mod(real(arxx(jjj,k)),100);
                    elseif real(arxx(jjj,k))<-100
                         arxx(jjj,k)=mod(real(arxx(jjj,k)),-100);
                      end
                  end
                     %arfitness(k)=feval(fhd,arxx(:,k),vara(:));
                     arfitness(k)=fhd([arxx(:,k)]);
                     counteval=counteval+1;
                      FEs=FEs+1;
                      if arfitness(k) < bsf_fit_var
                        bsf_fit_var = arfitness(k);
                        xatmin = arxx(:,k)';
                      end
                 end
                                  
                     [arfitness, arindex]=sort(arfitness);
                     xval=arxx(:,arindex(1)); 
                     
                     if abs(arfitness(1)-previousbest)<1*10^(-11)
                         break;
                     else
                         previousbest=arfitness(1);
                     end
                     
                     
                     if arfitnessbest(kkk)>arfitness(1)
                        arfitnessbest(kkk)=arfitness(1);
                        xvalbest(:,kkk)=arxx(:,arindex(1));
                     end
                     xmean=arxx(:,arindex(1:mu))*weights;
                     zmean=arz(:,arindex(1:mu))*weights;
                     ps=(1-cs)*ps+(sqrt(cs*(2-cs)*mueff))*(B*zmean);
                     hsig=norm(ps)/sqrt(1-(1-cs)^(2*counteval/lambda))/chiN<1.4+2/(N+1);    
                     pc=(1-cc)*pc+hsig*sqrt(cc*(2-cc)*mueff)*(B*DD*zmean);
                     C=(1-c1-cmu)*C+c1*(pc*pc'+(1-hsig)*cc*(2-cc)*C)+cmu*(B*DD*arz(:,arindex(1:mu)))*diag(weights)*(B*DD*arz(:,arindex(1:mu)))';
                     sigma=sigma*exp((cs/damps)*(norm(ps)/chiN-1));
                     xx(counteval/lambda)=sigma;
                     if counteval-eigenval>lambda/(cmu)/N/10
                         eigenval=counteval;
                         C=triu(C)+triu(C,1)';
                         [B,DD]=eig(C);
                          DD=diag(sqrt(diag(DD)));
                     end
                     
                     if arfitness(1)==arfitness(ceil(0.7*lambda))
                        sigma=sigma*exp(0.2+cs/damps);
                        xx(counteval/lambda)=sigma;
                     end  
                  %  disp([num2str(counteval) ':' num2str(arfitness(1))]);
                  iter = iter + 1;
                bsf_fit_var = bestval;
                history(iter,1) = iter;
                history(iter,2) = FEs;
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
             end
             %univariate sampling 
             
             if D<=30
               total=200;
               mu=160;
             elseif D==50
               total=450;
               mu=360;
             else
                  total=600;
               mu=480;
             end
            if D>=50
                 if FEs<=0.3*Max_FEs
                     total=total+200;
                     mu=mu+160;
                 end
             end
             weights=log(mu+1/2)-log(1:mu)';
             weights=weights/sum(weights);
             if D<=30
                 ppp1=std(xvalbest');
              ppp2=sort(std(xvalbest'));
                if ppp2(1)>0.2
                   dividevalue=0;
                elseif max(ppp2)<0.01
                    dividevalue=1;
                else
                   for dd=2:D
                       indicatorppp(dd)=(ppp2(dd)-ppp2(dd-1))/ppp2(dd-1);
                   end
                       indicatorppp(1)=min(indicatorppp)-0.001;
                      [value1,value2]=sort(indicatorppp,'descend');
                   for dd=1:D
                      if ppp2(value2(dd))<10
                      if ppp2(value2(dd))>0.1
                      dividevalue=ppp2(value2(dd))-0.001;
                      break;
                      end
                     elseif ppp2(value2(dd)-1)<0.01
                     dividevalue=ppp2(value2(dd))-0.001;
                    break;
                      end
                      if dd==D
                           dividevalue=ppp2(value2(dd))-0.001;
                      end
                    end     
                end
             else
                  for kkk2=1:total/5
                      spos(kkk2,:)=xvalbest(:);
                 end
             for d=1:D
                for k=1:total/5
                      spos(k,d)=xvalbest(d)-0.1*total+1*k;
                 end
                 %e=feval(fhd,spos',vara(:));
                 e = zeros(total/5,1);
                 for it1=1:total/5
                    e(it) = fhd(spos(it1,:)');
                 end
                 FEs=FEs+total/5;
                 bbpbbp(d)=abs(max(e)/arfitnessbest);
                 for k=1:total/5
                     spos(k,d)=xvalbest(d);
                 end
             end
             
              if max(bbpbbp)<3.1
                 for d=1:D
                     bbpbb(d)=1;
                 end
              else
                  [aaa1,aaa2]=sort(bbpbbp);
                  for d=1:D-1
                      diaaa1(d)=aaa1(d+1)/aaa1(d);
                  end
                      [aab1,aab2]=sort(diaaa1,'descend');
                  if aaa1(D/2)<=2
                    for d=1:D-1
                       if aaa1(aab2(d))<1.8
                          division=aaa1(aab2(d))+0.01;
                          break;
                       end
                    end
                    for d=1:D
                       if bbpbbp(d)<=division
                          bbpbb(d)=1;
                       else
                           bbpbb(d)=0;
                       end
                    end
                  else
                      for d=1:D-1
                        if aaa1(aab2(d))<4
                          division=aaa1(aab2(d))+0.01;
                          break;
                        else division=0;
                       end
                      end
                  for d=1:D
                    if bbpbbp(d)<=division
                       bbpbb(d)=1;
                    else
                        bbpbb(d)=0;
                    end
                  end
                  end
              end
             end
              kk=1;
              cc2=0;
             
               %VRmin=repmat(Xmin,total,1);
               %VRmax=repmat(Xmax,total,1);
               pos=VRmin+(VRmax-VRmin).*rand(total,D);
               FEs=FEs+total;

            while FEs<Max_FEs-total
%            e1=feval(fhd,pos',vara(:));
            %e1=fhd(pos);
            e1 = zeros(total,1);
            for it1=1:total
                e1(it1) = fhd(pos(it1,:)');
            end
            FEs=FEs+total;
            [a1,a2]=sort(e1);
         % a1(1)
          xmin(kk)=a1(1);
            if kk==1
            [arfitnessbest,seq]=min(arfitnessbest);
            end
            if a1(1)<arfitnessbest
                xy(kk)=a1(1);
                xyvector=pos(a2(1),:);
            else
                xy(kk)=arfitnessbest;
                 xyvector=xvalbest(:,seq(1));
            end      
            if a1 < bsf_fit_var
                xatmin = pos(a2(1),:);
                bsf_fit_var = a1(1);
            end
            iter = iter + 1;
            history(iter,1) = iter;
            history(iter,2) = FEs;
            history(iter,3) = bsf_fit_var;
            history(iter,4) = toc;        
            if opts.showits && ~mod(iter,50) 
                fprintf("Iter:%5i\tf_min:%15.10f\ttime(s):%10.5f\tfn evals:%9i\n",[history(iter,1),history(iter,3),history(iter,4),history(iter,2)]);
                %Iter:   43   f_min:  -78.9844713606    time(s):    0.34048    fn evals:     1723
            end
            if bsf_fit_var-opts.globalmin < opts.tolabs
                break
            end
            newpos=pos(a2(1:mu),:);
            meanval=(newpos(:,1:D)'*weights)';
             stdval=1*std(newpos);
%              if max(stdval)<0.0001
%                  break;
%              end
            if kk==1
                if D>=50
                   for jjj=1:D
                     if bbpbb(jjj)==0
                     stdval(jjj)=0.001;
                     meanval(jjj)=xvalbest(jjj);
                     end
                   end
                else
                    for jjj=1:D
                      if ppp1(jjj)<dividevalue
                          stdval(jjj)=0.001;
                     meanval(jjj)=xvalbest(jjj,seq(1));
                      end
                    end
                end
            end
            kk=kk+1;
            if kk>30 
               if mod(kk,20)==0
                   [aaa,bbb]=min(xmin);
               if bbb<kk-20
                   cc2=1;
               else
                   cc2=0;
                end
               end
            end
            for k=1:total
            if cc2==1      
                   a=0.96*randn(1,D);
            else 
                   a=1*randn(1,D);
            end
             pos(k,:)=meanval+stdval.*a;
            end           
             for k=1:total
                 for j=1:D
                    if pos(k,j)>100
                        pos(k,j)=meanval(j)+stdval(j).*randn;
                   elseif pos(k,j)<-100
                        pos(k,j)=meanval(j)+stdval(j).*randn;
                    end
                 end
              end
%              pos=pos(1:total,:);
%              pos(1,:)=xyvector(:);
            

            end
         if exist('xy')
             processvalue(14)=min(xy);
             clear xy;
             clear spos;
             meanval;
         end

    end 
end
xatmin = xatmin';
minima = bsf_fit_var;


end
