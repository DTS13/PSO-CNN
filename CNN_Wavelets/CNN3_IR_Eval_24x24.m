%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%
%%%%%%%%%%%%  CNN IR  %%%%%%%%%%%%


testpreds = classify(M0mnist,test);
plotconfusion(testpreds,test.Labels)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

im = imread('D:\Documents\MATLAB\THESIS\Datasets\IR Database\FLIR_ADAS_1_3\train\thermal_8_bit\FLIR_00016.jpeg');

actvn1 = activations(M0mnist,im,'Odd1');
actvn2 = activations(M0mnist,im,'Even1');
actvn3 = activations(M0mnist,im,'ShiftV1');
actvn4 = activations(M0mnist,im,'Add2');
actvn5 = activations(M0mnist,im,'Sub2');
%actvn6 = activations(M0mnist,im,'PoolD1');
%actvn7 = activations(M0mnist,im,'Batch1D1');
%actvn8 = activations(M0mnist,im,'Batch2D1');
%actvn9 = activations(M0mnist,im,'Batch3D1');
%actvn10 = activations(M0mnist,im,'Batch4D1');

figure(11); montage(rescale(actvn1))
figure(12); montage(rescale(actvn2))
figure(13); montage(rescale(actvn3))
figure(14); montage(rescale(actvn4))
figure(15); montage(rescale(actvn5))
%figure(8); montage(rescale(actvn6))
%figure(9); montage(rescale(actvn7))
%figure(10); montage(rescale(actvn8))
%figure(11); montage(rescale(actvn9))
%figure(12); montage(rescale(actvn10))

%figure(); imshow(actvn1(:,:,1))
%figure(); imshow(actvn2(:,:,1))
%figure(); imshow(actvn3(:,:,1))

%%%%%%%% Reference %%%%%%%%
%%%% Ref1 Average Pooling: 98.5%
%%%% Ref2 Max Pooling:     99.0%;V:99.0%
%%%% Ref3 Mix Pooling:     97.2%
%%%% Ref4 Stochastic Pooling:  __._%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Ref5 Average Pooling and LeakyRelu: 99.2%
%%%% Ref6 Max Pooling and LeakyRelu:     99.2%
%%%% Ref7 Mix Pooling and LeakyRelu:     97.2%
%%%% Ref8 Stochastic Pooling and LeakyRelu:  __._%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%% MODELO 1 %%%%%%%%%%
%%%%%%%% Lifting Haar Only %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% a sin batch:     98.3%
%%%% b con batch:     98.6%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% c con relu:      99.0%
%%%% d con leakyrelu: 99.2%;V:98.7%;V:99.0%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%% MODELO 2 %%%%%%%%%%
%%%%%%%% Lifting Haar and Average Pooling Parallel %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% a suma sin batch: 98.4%
%%%% b suma con batch: 98.3%, 98.5% (batch despues de suma)
%%%% c suma con relu:  99.2% (relu despues de suma)
%%%% h suma con leakyrelu: 99.3% (leaky relu despues de suma);V:99.1%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% d cat sin batch:  98.2%
%%%% e cat con batch:  98.4% (batch despues de suma)
%%%% f cat con relu:   99.1% (relu despues de suma)
%%%% g cat con leakyrelu:  99.3% (leaky relu despues de suma);V:99.3%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% i suma con batch: 98.5%, 98.6% (batch antes de suma)
%%%% j suma con relu:  99.2% (relu antes de suma)
%%%% k suma con leakyrelu: 99.3% (leakyrelu antes de suma);V:99.1%;V:99.1%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% l cat con batch:  98.4% (batch antes de suma)
%%%% m cat con relu:   99.1% (relu antes de suma)
%%%% n cat con leakyrelu: 99.2% (leakyrelu antes de suma);V:99.2%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%% MODELO 3 %%%%%%%%%%
%%%%%%%% Lifting Haar (LL) and Max Pooling Parallel %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% a suma con relu: 99.3% (despues); V:99.4%
% b suma con leakyrelu: 99.1% (despues)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% c cat con relu: 99.1% (despues)
% d cat w/leakyrelu:99.4%;99.2% (despues); V:99.3%;V:99.1%;V:99.3%;V:99.5%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% e cat w/leakyrelu,Norm: 99.1%
% f cat w/leakyrelu,Norm,sinBatch1,sinFC: 99.3%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%% MODELO 4 %%%%%%%%%%
%%%%%%%% Lifting Haar and Mix Pooling Parallel %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% a suma con relu: 99.1% (despues)
%%%% b suma con leakyrelu: 99.0% (despues)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% c cat con relu: 99.1% (despues)
%%%% d cat con leakyrelu: 99.0% (despues)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%% MODELO 5 %%%%%%%%%%
%%%%%%%% Average Pooling and Max Pooling Parallel %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% a suma con relu: 99.3% (despues);V:99.1%
%%%% b suma con leakyrelu: 99.3% (despues);V:99.0%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% c cat con relu: 99.2% (despues);V:99.2%
%%%% d cat con leakyrelu: 99.2% (despues);V:99.3%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%% MODELO 6 %%%%%%%%%%
%%%%%%%% Lifting Haar, Average Pooling and Max Pooling Parallel %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% a suma con relu: 99.2% (despues)
%%%% b suma con leakyrelu: 99.1% (despues)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% c cat con relu: 99.3% (despues);V:99.4%
%%%% d cat con leakyrelu: 99.2% (despues);V:99.1%;V:99.2%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%% MODELO 7 %%%%%%%%%%
%%%%%%%% Lifting Haar, All Coefficients %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% a suma, sin relu (antes), sin relu (despues): 98.3%
%%%% b suma, con relu solo LL (antes), sin relu otros (antes), sin relu (despues): 98.7%
%%%% c suma, con relu todos (antes), sin relu (despues): 99.0%
%%%% d suma, sin relu (antes), con relu (despues): 98.7%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% f suma, con leakyrelu solo LL (antes), sin leakyrelu otros (antes), sin relu (despues): 98.5%
%%%% g suma, con leakyrelu todos (antes), sin leakyrelu (despues): 98.9%
%%%% h suma, sin leakyrelu (antes), con leakyrelu (despues): 98.7%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% i cat, sin relu (antes), sin relu (despues): 98.1%
%%%% j cat, con relu solo LL (antes), sin relu otros (antes), sin relu (despues): 98.8%
%%%% k cat, con relu todos (antes), sin relu (despues): 99.0%
%%%% l cat, sin relu (antes), con relu (despues): 99.0%
%%%% m cat, con relu (antes), con relu (despues): 98.9%;V:99.0%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% o cat, con leakyrelu solo LL (antes), sin leakyrelu otros (antes), sin relu (despues): 98.9%
%%%% p cat, con leakyrelu todos (antes), sin leakyrelu (despues): 99.0%
%%%% q cat, sin leakyrelu (antes), con leakyrelu (despues): 99.0%
%%%% r cat,con leakyrelu (antes),con leakyrelu(despues):98.7%;V:98.8%;V:99.1%;N:99.0%;N:99.1%
%%%% rr cat,con leakyrelu (antes),con leakyrelu(despues): __._%
%%%% s cat, con batch (antes),con leakyrelu(despues):98.9%;V:99.0%;V:99.1%;N:98.2%
%%%%%%%%%%%%%%%%%%%%%%% Iterations: ---; Epocs: 20;
%%%%%%%%%%%%%%%%%%%%%%% Val.Freq:50; Val.Patience:20
%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%% MODELO 71 %%%%%%%%%%
%%%%%%%% Lifting DB4, All Coefficients %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% a cat,con leakyrelu (antes),con leakyrelu(despues): __._%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%% MODELO 72 %%%%%%%%%%
%%%%%%%% Lifting DB6, All Coefficients %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% a cat,con leakyrelu (antes),con leakyrelu(despues): __._%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%% MODELO 8 %%%%%%%%%%
%%%%%%%% Lifting Haar, All Coefficients and Max pool %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% a cat,w/batch (antes cat),w/leakyrelu(despues):99.1%;V:99.2%,98.9%,99.18%
% b cat,w/batch (dentro),con leakyrelu(despues):99.1.%;V:98.9%
% c cat,w/batch (solo dentro),w/leakyrelu(despues):99.1%;V:98.8,98.9,99.1%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%% MODELO 9 %%%%%%%%%%
%%%%%%%% Lifting Haar, LL and HH Coefficients Only %%%%%%%%


%%% PROBAR PUNTO OPTIMO DE ENTRENAMIENTO CON LA VALIDACION %%%

%%%%%%%% Lifting Haar and Stochastic Pooling %%%%%%%%
%%%%%%%% Lifting cdf53 and Max Pooling %%%%%%%%
%%%%%%%% Lifting cdf53 and Mix Pooling %%%%%%%%
%%%%%%%% Lifting cdf53 and Stochastic Pooling %%%%%%%%
%%%%%%%% Lifting cdf97 and Max Pooling %%%%%%%%
%%%%%%%% Lifting cdf97 and Mix Pooling %%%%%%%%
%%%%%%%% Lifting cdf97 and Stochastic Pooling %%%%%%%%
