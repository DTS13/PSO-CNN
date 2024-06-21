clc;
 clear all;
 close all;
 %Reading image
 Img   = imread('w.jpg');
 I     = imresize(Img,[256 256]); 
 I     = rgb2gray(I);
 [m,n] = size(I);
 %PSO implementation
 P       = 5;        %swarm size
 iter    = 6;        %number of iterations
 c1      = 2;
 c2      = 2;        %usually c1=c2=2
 r1      = rand;
 r2      = rand;
 fit_val = [];       %matrix for storing fitness values
 P_best  = [];       %matrix for storing pbest values
 pbest   = 0;
 gbest   = [];
 %particle initialization
 for i = 1:P
 %updating particle position
       a(i) = (1.5).*rand(1,1);
       b(i) = (0.5).*rand(1,1);
       c(i) = rand(1,1);
       k(i) = 0.5+1.*rand(1,1);
       fprintf('particle  %d:: a=%f ; b=%f ; c=%f ;k=%f \n',i,a(i),b(i),c(i),k(i));
 %updating particle velocity
       v1(i)  =  0.15*a(i);
       v2(i)  =  0.5*b(i);
       v3(i)  =  0.15*c(i);
       v4(i)  =  0.5*k(i);
       f_v    =  []; 
       P_best =  [];
       pbest  =  [];
  end
  %iterations
   for it=1:iter
       IG=I;
       fprintf('\n');
       fprintf('--> ITERATION %d <--\n',it)   
       for i=1:P
            fprintf('\n.....PARTICLE (%d)....',i);
            figure
            imshow(IG);
            pause(1);
           IG              = uint8(trans_fcn(IG,a(i),b(i),c(i),k(i)));
           [fitness,out1]  = f_fcn(m,n,IG);
           f_v(it,i)       = fitness;
           pbest(i)        = max(f_v(:,i));
           P_best(it,i)    = pbest(i); 
           fprintf('FITNESS VALUE of (%d) iteration :\t\t\n',it); 
           disp(f_v()); 
           fprintf('p_best_values: \n'); 
           disp(P_best(it,:));
           gbest(it) = max(f_v(:)); %calculating gbest
          
           v1(i) = (1/it).*v1(i) + c1.*r1.*(gbest(it)-a(i)) + c2.*r2.*(pbest(i)-a(i));
           a(i)  = a(i) + v1(i);
%            a(i) = ((a(i)-0.1139724)/(7.7605725-0.113924))*1.5;      % 0 to 1.5
           fprintf('a = %d \n',a(i));
           
           v2(i)= (1/it).*v2(i) + c1.*r1.*(gbest(it)-b(i)) + c2.*r2.*(pbest(i)-b(i));
           b(i) = b(i) + v2(i);
%            b(i) = ((b(i)-(0.31926))/(0.9866753-(0.31926)))*0.5;    % 0 to 0.5                  
           fprintf('b = %d \n',b(i));

           v3(i) = (1/it).*v3(i) + c1.*r1.*(gbest(it)-c(i)) + c2.*r2.*(pbest(i)-c(i));
           c(i)  = c(i) + v3(i);
%            c(i)  = (c(i)- 0.1117058)/(2.78389-0.7249451);          % 0 to 1                    
           fprintf('c = %d \n',c(i));

           v4(i) = (1/it).*v4(i) + c1.*r1.*(gbest(it)-k(i)) + c2.*r2.*(pbest(i)-k(i));
           k(i)  = k(i) + v4(i);
%            k(i) = 0.5+(((k(i)-(0.193213))/(0.9783839-(0.193213)))* 1);          
           fprintf('k = %d',k(i));                                % 0.5 to 1.5
           fprintf('\n');
           
%    if (out1.n_edgels<500||out1.E<100000||out1.H<0.2123)
%                break;
%        end
       end
       fprintf('gbest of iteration %d :-> %f', it, gbest(it));
       fprintf('\n');
       fprintf('.......Iteration %d stops......', it)
end
   figure
       x= [1:iter];
       y= gbest ;
       plot(x,y)
%     figure
%     bar(gbest)
%     imshow(IG);