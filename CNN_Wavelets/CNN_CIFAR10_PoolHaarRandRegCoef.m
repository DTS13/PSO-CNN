%%%%%%%%%%%%%%%%%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  CNN CIFAR10 Pool Random Region Haar Coefficients (LH, HL & HH)  %%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
In = imageInputLayer([32 32 1],'Name','In');
Batch1 = batchNormalizationLayer('Name','Batch1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Conv1 = convolution2dLayer([5,5],32,'Padding',[2 2 2 2],'Name','Conv1');
Batch2 = batchNormalizationLayer('Name','Batch2');
Relu1 = reluLayer('Name','Relu1');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pool1 = maxPooling2dLayer([2 2],'Stride',[2 2],'Name','Pool1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftV1D1 = ShiftLayerV('ShiftV1D1');
Odd1D1 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Odd1D1'); %Odd
Even1D1 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Even1D1'); %Even
MultP1D1 = MultLayerP('MultP1D1');
MultU1D1 = MultLayerU('MultU1D1');
Add1D1 = additionLayer(2,'Name','Add1D1');
Sub1D1 = additionLayer(2,'Name','Sub1D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH2D1 = ShiftLayerH('ShiftH2D1'); 
Odd2D1 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd2D1'); %Odd
Even2D1 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even2D1'); %Even
MultP2D1 = MultLayerP('MultP2D1');
MultU2D1 = MultLayerU('MultU2D1');
Add2D1 = additionLayer(2,'Name','Add2D1');
Sub2D1 = additionLayer(2,'Name','Sub2D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH3D1 = ShiftLayerH('ShiftH3D1'); 
Odd3D1 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd3D1'); %Odd
Even3D1 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even3D1'); %Even
MultP3D1 = MultLayerP('MultP3D1');
MultU3D1 = MultLayerU('MultU3D1');
Add3D1 = additionLayer(2,'Name','Add3D1');
Sub3D1 = additionLayer(2,'Name','Sub3D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MultNA1D1 = MultHaarNormAprox('MultNA1D1');
MultND1D1 = MultHaarNormDetall('MultND1D1');
MultNA2D1 = MultHaarNormAprox('MultNA2D1');
MultND2D1 = MultHaarNormDetall('MultND2D1');
MultNA3D1 = MultHaarNormAprox('MultNA3D1');
MultND3D1 = MultHaarNormDetall('MultND3D1');
Relu11D1 = leakyReluLayer('Name','Relu11D1');
Relu12D1 = leakyReluLayer('Name','Relu12D1');
Relu21D1 = leakyReluLayer('Name','Relu21D1');
Relu22D1 = leakyReluLayer('Name','Relu22D1');
Relu31D1 = leakyReluLayer('Name','Relu31D1');
Relu32D1 = leakyReluLayer('Name','Relu32D1');
Selec1 = randRegWavpoolLayer('Selec1');
Cat1 = concatenationLayer(3,2,'Name','Cat1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% END  POOL 1 %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Conv2 = convolution2dLayer([5,5],32,'Padding',[2 2 2 2],'Name','Conv2');
Batch3 = batchNormalizationLayer('Name','Batch3');
Relu2 = reluLayer('Name','Relu2');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pool2 = maxPooling2dLayer([2 2],'Stride',[2 2],'Name','Pool2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftV1D2 = ShiftLayerV('ShiftV1D2'); 
Odd1D2 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Odd1D2'); %Odd
Even1D2 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Even1D2'); %Even
MultP1D2 = MultLayerP('MultP1D2');
MultU1D2 = MultLayerU('MultU1D2');
Add1D2 = additionLayer(2,'Name','Add1D2');
Sub1D2 = additionLayer(2,'Name','Sub1D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH2D2 = ShiftLayerH('ShiftH2D2'); 
Odd2D2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd2D2'); %Odd
Even2D2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even2D2'); %Even
MultP2D2 = MultLayerP('MultP2D2');
MultU2D2 = MultLayerU('MultU2D2');
Add2D2 = additionLayer(2,'Name','Add2D2');
Sub2D2 = additionLayer(2,'Name','Sub2D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH3D2 = ShiftLayerH('ShiftH3D2'); 
Odd3D2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd3D2'); %Odd
Even3D2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even3D2'); %Even
MultP3D2 = MultLayerP('MultP3D2');
MultU3D2 = MultLayerU('MultU3D2');
Add3D2 = additionLayer(2,'Name','Add3D2');
Sub3D2 = additionLayer(2,'Name','Sub3D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
MultNA1D2 = MultHaarNormAprox('MultNA1D2');
MultND1D2 = MultHaarNormDetall('MultND1D2');
MultNA2D2 = MultHaarNormAprox('MultNA2D2');
MultND2D2 = MultHaarNormDetall('MultND2D2');
MultNA3D2 = MultHaarNormAprox('MultNA3D2');
MultND3D2 = MultHaarNormDetall('MultND3D2');
Relu11D2 = leakyReluLayer('Name','Relu11D2');
Relu12D2 = leakyReluLayer('Name','Relu12D2');
Relu21D2 = leakyReluLayer('Name','Relu21D2');
Relu22D2 = leakyReluLayer('Name','Relu22D2');
Relu31D2 = leakyReluLayer('Name','Relu31D2');
Relu32D2 = leakyReluLayer('Name','Relu32D2');
Selec2 = randRegWavpoolLayer('Selec2');
Cat2 = concatenationLayer(3,2,'Name','Cat2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% END  POOL 2 %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Conv3 = convolution2dLayer([5,5],32,'Padding',[2 2 2 2],'Name','Conv3');
Batch4 = batchNormalizationLayer('Name','Batch4');
Relu3 = reluLayer('Name','Relu3');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pool3 = maxPooling2dLayer([2 2],'Stride',[2 2],'Name','Pool3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftV1D3 = ShiftLayerV('ShiftV1D3'); 
Odd1D3 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Odd1D3'); %Odd
Even1D3 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Even1D3'); %Even
MultP1D3 = MultLayerP('MultP1D3');
MultU1D3 = MultLayerU('MultU1D3');
Add1D3 = additionLayer(2,'Name','Add1D3');
Sub1D3 = additionLayer(2,'Name','Sub1D3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH2D3 = ShiftLayerH('ShiftH2D3'); 
Odd2D3 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd2D3'); %Odd
Even2D3 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even2D3'); %Even
MultP2D3 = MultLayerP('MultP2D3');
MultU2D3 = MultLayerU('MultU2D3');
Add2D3 = additionLayer(2,'Name','Add2D3');
Sub2D3 = additionLayer(2,'Name','Sub2D3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH3D3 = ShiftLayerH('ShiftH3D3'); 
Odd3D3 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd3D3'); %Odd
Even3D3 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even3D3'); %Even
MultP3D3 = MultLayerP('MultP3D3');
MultU3D3 = MultLayerU('MultU3D3');
Add3D3 = additionLayer(2,'Name','Add3D3');
Sub3D3 = additionLayer(2,'Name','Sub3D3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
MultNA1D3 = MultHaarNormAprox('MultNA1D3');
MultND1D3 = MultHaarNormDetall('MultND1D3');
MultNA2D3 = MultHaarNormAprox('MultNA2D3');
MultND2D3 = MultHaarNormDetall('MultND2D3');
MultNA3D3 = MultHaarNormAprox('MultNA3D3');
MultND3D3 = MultHaarNormDetall('MultND3D3');
Relu11D3 = leakyReluLayer('Name','Relu11D3');
Relu12D3 = leakyReluLayer('Name','Relu12D3');
Relu21D3 = leakyReluLayer('Name','Relu21D3');
Relu22D3 = leakyReluLayer('Name','Relu22D3');
Relu31D3 = leakyReluLayer('Name','Relu31D3');
Relu32D3 = leakyReluLayer('Name','Relu32D3');
Selec3 = randRegWavpoolLayer('Selec3');
Cat3 = concatenationLayer(3,2,'Name','Cat3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% END  POOL 3 %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Drop1 = dropoutLayer(0.1,'Name','Drop1');
Conv4 = convolution2dLayer([4,4],64,'Name','Conv4');
Relu4 = reluLayer('Name','Relu4');
Drop2 = dropoutLayer(0.2,'Name','Drop2');
Conv5 = convolution2dLayer([1,1],4,'Name','Conv5');
%FC1 = fullyConnectedLayer(4,'Name','FC1');
Soft = softmaxLayer('Name','Soft');
Out = classificationLayer('Name','Out');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%}


%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = layerGraph;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,In);
netM0 = addLayers(netM0,Batch1);
netM0 = addLayers(netM0,Conv1);
netM0 = addLayers(netM0,Batch2);
netM0 = addLayers(netM0,Relu1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% netM0 = addLayers(netM0,Pool1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftV1D1);
netM0 = addLayers(netM0,Odd1D1);
netM0 = addLayers(netM0,Even1D1);
netM0 = addLayers(netM0,MultP1D1);
netM0 = addLayers(netM0,MultU1D1);
netM0 = addLayers(netM0,Add1D1);
netM0 = addLayers(netM0,Sub1D1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftH2D1);
netM0 = addLayers(netM0,Odd2D1);
netM0 = addLayers(netM0,Even2D1);
netM0 = addLayers(netM0,MultP2D1);
netM0 = addLayers(netM0,MultU2D1);
netM0 = addLayers(netM0,Add2D1);
netM0 = addLayers(netM0,Sub2D1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftH3D1);
netM0 = addLayers(netM0,Odd3D1);
netM0 = addLayers(netM0,Even3D1);
netM0 = addLayers(netM0,MultP3D1);
netM0 = addLayers(netM0,MultU3D1);
netM0 = addLayers(netM0,Add3D1);
netM0 = addLayers(netM0,Sub3D1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,MultNA1D1);
netM0 = addLayers(netM0,MultND1D1);
netM0 = addLayers(netM0,MultNA2D1);
netM0 = addLayers(netM0,MultND2D1);
netM0 = addLayers(netM0,MultNA3D1);
netM0 = addLayers(netM0,MultND3D1);
netM0 = addLayers(netM0,Relu11D1);
netM0 = addLayers(netM0,Relu12D1);
netM0 = addLayers(netM0,Relu21D1);
netM0 = addLayers(netM0,Relu22D1);
netM0 = addLayers(netM0,Relu31D1);
netM0 = addLayers(netM0,Relu32D1);
netM0 = addLayers(netM0,Selec1);
netM0 = addLayers(netM0,Cat1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END POOL 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%

netM0 = addLayers(netM0,Conv2);
netM0 = addLayers(netM0,Batch3);
netM0 = addLayers(netM0,Relu2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% netM0 = addLayers(netM0,Pool2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftV1D2);
netM0 = addLayers(netM0,Odd1D2);
netM0 = addLayers(netM0,Even1D2);
netM0 = addLayers(netM0,MultP1D2);
netM0 = addLayers(netM0,MultU1D2);
netM0 = addLayers(netM0,Add1D2);
netM0 = addLayers(netM0,Sub1D2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftH2D2);
netM0 = addLayers(netM0,Odd2D2);
netM0 = addLayers(netM0,Even2D2);
netM0 = addLayers(netM0,MultP2D2);
netM0 = addLayers(netM0,MultU2D2);
netM0 = addLayers(netM0,Add2D2);
netM0 = addLayers(netM0,Sub2D2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftH3D2);
netM0 = addLayers(netM0,Odd3D2);
netM0 = addLayers(netM0,Even3D2);
netM0 = addLayers(netM0,MultP3D2);
netM0 = addLayers(netM0,MultU3D2);
netM0 = addLayers(netM0,Add3D2);
netM0 = addLayers(netM0,Sub3D2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,MultNA1D2);
netM0 = addLayers(netM0,MultND1D2);
netM0 = addLayers(netM0,MultNA2D2);
netM0 = addLayers(netM0,MultND2D2);
netM0 = addLayers(netM0,MultNA3D2);
netM0 = addLayers(netM0,MultND3D2);
netM0 = addLayers(netM0,Relu11D2);
netM0 = addLayers(netM0,Relu12D2);
netM0 = addLayers(netM0,Relu21D2);
netM0 = addLayers(netM0,Relu22D2);
netM0 = addLayers(netM0,Relu31D2);
netM0 = addLayers(netM0,Relu32D2);
netM0 = addLayers(netM0,Selec2);
netM0 = addLayers(netM0,Cat2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END POOL 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%

netM0 = addLayers(netM0,Conv3);
netM0 = addLayers(netM0,Batch4);
netM0 = addLayers(netM0,Relu3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% netM0 = addLayers(netM0,Pool3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftV1D3);
netM0 = addLayers(netM0,Odd1D3);
netM0 = addLayers(netM0,Even1D3);
netM0 = addLayers(netM0,MultP1D3);
netM0 = addLayers(netM0,MultU1D3);
netM0 = addLayers(netM0,Add1D3);
netM0 = addLayers(netM0,Sub1D3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftH2D3);
netM0 = addLayers(netM0,Odd2D3);
netM0 = addLayers(netM0,Even2D3);
netM0 = addLayers(netM0,MultP2D3);
netM0 = addLayers(netM0,MultU2D3);
netM0 = addLayers(netM0,Add2D3);
netM0 = addLayers(netM0,Sub2D3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftH3D3);
netM0 = addLayers(netM0,Odd3D3);
netM0 = addLayers(netM0,Even3D3);
netM0 = addLayers(netM0,MultP3D3);
netM0 = addLayers(netM0,MultU3D3);
netM0 = addLayers(netM0,Add3D3);
netM0 = addLayers(netM0,Sub3D3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,MultNA1D3);
netM0 = addLayers(netM0,MultND1D3);
netM0 = addLayers(netM0,MultNA2D3);
netM0 = addLayers(netM0,MultND2D3);
netM0 = addLayers(netM0,MultNA3D3);
netM0 = addLayers(netM0,MultND3D3);
netM0 = addLayers(netM0,Relu11D3);
netM0 = addLayers(netM0,Relu12D3);
netM0 = addLayers(netM0,Relu21D3);
netM0 = addLayers(netM0,Relu22D3);
netM0 = addLayers(netM0,Relu31D3);
netM0 = addLayers(netM0,Relu32D3);
netM0 = addLayers(netM0,Selec3);
netM0 = addLayers(netM0,Cat3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END POOL 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%

netM0 = addLayers(netM0,Drop1);
netM0 = addLayers(netM0,Conv4);
netM0 = addLayers(netM0,Relu4);
netM0 = addLayers(netM0,Drop2);
netM0 = addLayers(netM0,Conv5);
%netM0 = addLayers(netM0,FC1);
netM0 = addLayers(netM0,Soft);
netM0 = addLayers(netM0,Out);
%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'In','Batch1');
netM0 = connectLayers(netM0,'Batch1','Conv1');
netM0 = connectLayers(netM0,'Conv1','Batch2');
netM0 = connectLayers(netM0,'Batch2','Relu1');

%netM0 = connectLayers(netM0,'Relu1','Pool1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Relu1','Odd1D1');
netM0 = connectLayers(netM0,'Relu1','ShiftV1D1');
netM0 = connectLayers(netM0,'ShiftV1D1','Even1D1');
netM0 = connectLayers(netM0,'Even1D1','MultP1D1');
netM0 = connectLayers(netM0,'Even1D1','Add1D1/in1');
netM0 = connectLayers(netM0,'Odd1D1','Sub1D1/in1');
netM0 = connectLayers(netM0,'Sub1D1','MultU1D1');
netM0 = connectLayers(netM0,'MultU1D1','Add1D1/in2');
netM0 = connectLayers(netM0,'MultP1D1','Sub1D1/in2');
netM0 = connectLayers(netM0,'Add1D1','MultNA1D1');
netM0 = connectLayers(netM0,'Sub1D1','MultND1D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'MultNA1D1','Relu11D1');
netM0 = connectLayers(netM0,'MultND1D1','Relu12D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Relu11D1','Odd2D1');
netM0 = connectLayers(netM0,'Relu11D1','ShiftH2D1');
netM0 = connectLayers(netM0,'ShiftH2D1','Even2D1');
netM0 = connectLayers(netM0,'Even2D1','MultP2D1');
netM0 = connectLayers(netM0,'Even2D1','Add2D1/in1');
netM0 = connectLayers(netM0,'Odd2D1','Sub2D1/in1');
netM0 = connectLayers(netM0,'Sub2D1','MultU2D1');
netM0 = connectLayers(netM0,'MultU2D1','Add2D1/in2');
netM0 = connectLayers(netM0,'MultP2D1','Sub2D1/in2');
netM0 = connectLayers(netM0,'Add2D1','MultNA2D1');
netM0 = connectLayers(netM0,'Sub2D1','MultND2D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Relu12D1','Odd3D1');
netM0 = connectLayers(netM0,'Relu12D1','ShiftH3D1');
netM0 = connectLayers(netM0,'ShiftH3D1','Even3D1');
netM0 = connectLayers(netM0,'Even3D1','MultP3D1');
netM0 = connectLayers(netM0,'Even3D1','Add3D1/in1');
netM0 = connectLayers(netM0,'Odd3D1','Sub3D1/in1');
netM0 = connectLayers(netM0,'Sub3D1','MultU3D1');
netM0 = connectLayers(netM0,'MultU3D1','Add3D1/in2');
netM0 = connectLayers(netM0,'MultP3D1','Sub3D1/in2');
netM0 = connectLayers(netM0,'Add3D1','MultNA3D1');
netM0 = connectLayers(netM0,'Sub3D1','MultND3D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'MultNA2D1','Relu21D1');
netM0 = connectLayers(netM0,'MultND2D1','Relu22D1');
netM0 = connectLayers(netM0,'MultNA3D1','Relu31D1');
netM0 = connectLayers(netM0,'MultND3D1','Relu32D1');
netM0 = connectLayers(netM0,'Relu21D1','Cat1/in1');
netM0 = connectLayers(netM0,'Relu22D1','Selec1/in1');
netM0 = connectLayers(netM0,'Relu31D1','Selec1/in2');
netM0 = connectLayers(netM0,'Relu32D1','Selec1/in3');
netM0 = connectLayers(netM0,'Selec1','Cat1/in2');
netM0 = connectLayers(netM0,'Cat1','Conv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = connectLayers(netM0,'Pool1','Conv2');

netM0 = connectLayers(netM0,'Conv2','Batch3');
netM0 = connectLayers(netM0,'Batch3','Relu2');

%netM0 = connectLayers(netM0,'Relu2','Pool2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Relu2','Odd1D2');
netM0 = connectLayers(netM0,'Relu2','ShiftV1D2');
netM0 = connectLayers(netM0,'ShiftV1D2','Even1D2');
netM0 = connectLayers(netM0,'Even1D2','MultP1D2');
netM0 = connectLayers(netM0,'Even1D2','Add1D2/in1');
netM0 = connectLayers(netM0,'Odd1D2','Sub1D2/in1');
netM0 = connectLayers(netM0,'Sub1D2','MultU1D2');
netM0 = connectLayers(netM0,'MultU1D2','Add1D2/in2');
netM0 = connectLayers(netM0,'MultP1D2','Sub1D2/in2');
netM0 = connectLayers(netM0,'Add1D2','MultNA1D2');
netM0 = connectLayers(netM0,'Sub1D2','MultND1D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'MultNA1D2','Relu11D2');
netM0 = connectLayers(netM0,'MultND1D2','Relu12D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Relu11D2','Odd2D2');
netM0 = connectLayers(netM0,'Relu11D2','ShiftH2D2');
netM0 = connectLayers(netM0,'ShiftH2D2','Even2D2');
netM0 = connectLayers(netM0,'Even2D2','MultP2D2');
netM0 = connectLayers(netM0,'Even2D2','Add2D2/in1');
netM0 = connectLayers(netM0,'Odd2D2','Sub2D2/in1');
netM0 = connectLayers(netM0,'Sub2D2','MultU2D2');
netM0 = connectLayers(netM0,'MultU2D2','Add2D2/in2');
netM0 = connectLayers(netM0,'MultP2D2','Sub2D2/in2');
netM0 = connectLayers(netM0,'Add2D2','MultNA2D2');
netM0 = connectLayers(netM0,'Sub2D2','MultND2D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Relu12D2','Odd3D2');
netM0 = connectLayers(netM0,'Relu12D2','ShiftH3D2');
netM0 = connectLayers(netM0,'ShiftH3D2','Even3D2');
netM0 = connectLayers(netM0,'Even3D2','MultP3D2');
netM0 = connectLayers(netM0,'Even3D2','Add3D2/in1');
netM0 = connectLayers(netM0,'Odd3D2','Sub3D2/in1');
netM0 = connectLayers(netM0,'Sub3D2','MultU3D2');
netM0 = connectLayers(netM0,'MultU3D2','Add3D2/in2');
netM0 = connectLayers(netM0,'MultP3D2','Sub3D2/in2');
netM0 = connectLayers(netM0,'Add3D2','MultNA3D2');
netM0 = connectLayers(netM0,'Sub3D2','MultND3D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'MultNA2D2','Relu21D2');
netM0 = connectLayers(netM0,'MultND2D2','Relu22D2');
netM0 = connectLayers(netM0,'MultNA3D2','Relu31D2');
netM0 = connectLayers(netM0,'MultND3D2','Relu32D2');
netM0 = connectLayers(netM0,'Relu21D2','Cat2/in1');
netM0 = connectLayers(netM0,'Relu22D2','Selec2/in1');
netM0 = connectLayers(netM0,'Relu31D2','Selec2/in2');
netM0 = connectLayers(netM0,'Relu32D2','Selec2/in3');
netM0 = connectLayers(netM0,'Selec2','Cat2/in2');
netM0 = connectLayers(netM0,'Cat2','Conv3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = connectLayers(netM0,'Pool2','Conv3');

netM0 = connectLayers(netM0,'Conv3','Batch4');
netM0 = connectLayers(netM0,'Batch4','Relu3');

% netM0 = connectLayers(netM0,'Relu3','Pool3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Relu3','Odd1D3');
netM0 = connectLayers(netM0,'Relu3','ShiftV1D3');
netM0 = connectLayers(netM0,'ShiftV1D3','Even1D3');
netM0 = connectLayers(netM0,'Even1D3','MultP1D3');
netM0 = connectLayers(netM0,'Even1D3','Add1D3/in1');
netM0 = connectLayers(netM0,'Odd1D3','Sub1D3/in1');
netM0 = connectLayers(netM0,'Sub1D3','MultU1D3');
netM0 = connectLayers(netM0,'MultU1D3','Add1D3/in2');
netM0 = connectLayers(netM0,'MultP1D3','Sub1D3/in2');
netM0 = connectLayers(netM0,'Add1D3','MultNA1D3');
netM0 = connectLayers(netM0,'Sub1D3','MultND1D3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'MultNA1D3','Relu11D3');
netM0 = connectLayers(netM0,'MultND1D3','Relu12D3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Relu11D3','Odd2D3');
netM0 = connectLayers(netM0,'Relu11D3','ShiftH2D3');
netM0 = connectLayers(netM0,'ShiftH2D3','Even2D3');
netM0 = connectLayers(netM0,'Even2D3','MultP2D3');
netM0 = connectLayers(netM0,'Even2D3','Add2D3/in1');
netM0 = connectLayers(netM0,'Odd2D3','Sub2D3/in1');
netM0 = connectLayers(netM0,'Sub2D3','MultU2D3');
netM0 = connectLayers(netM0,'MultU2D3','Add2D3/in2');
netM0 = connectLayers(netM0,'MultP2D3','Sub2D3/in2');
netM0 = connectLayers(netM0,'Add2D3','MultNA2D3');
netM0 = connectLayers(netM0,'Sub2D3','MultND2D3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Relu12D3','Odd3D3');
netM0 = connectLayers(netM0,'Relu12D3','ShiftH3D3');
netM0 = connectLayers(netM0,'ShiftH3D3','Even3D3');
netM0 = connectLayers(netM0,'Even3D3','MultP3D3');
netM0 = connectLayers(netM0,'Even3D3','Add3D3/in1');
netM0 = connectLayers(netM0,'Odd3D3','Sub3D3/in1');
netM0 = connectLayers(netM0,'Sub3D3','MultU3D3');
netM0 = connectLayers(netM0,'MultU3D3','Add3D3/in2');
netM0 = connectLayers(netM0,'MultP3D3','Sub3D3/in2');
netM0 = connectLayers(netM0,'Add3D3','MultNA3D3');
netM0 = connectLayers(netM0,'Sub3D3','MultND3D3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'MultNA2D3','Relu21D3');
netM0 = connectLayers(netM0,'MultND2D3','Relu22D3');
netM0 = connectLayers(netM0,'MultNA3D3','Relu31D3');
netM0 = connectLayers(netM0,'MultND3D3','Relu32D3');
netM0 = connectLayers(netM0,'Relu21D3','Cat3/in1');
netM0 = connectLayers(netM0,'Relu22D3','Selec3/in1');
netM0 = connectLayers(netM0,'Relu31D3','Selec3/in2');
netM0 = connectLayers(netM0,'Relu32D3','Selec3/in3');
netM0 = connectLayers(netM0,'Selec3','Cat3/in2');
netM0 = connectLayers(netM0,'Cat3','Drop1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% netM0 = connectLayers(netM0,'Pool3','Drop1');

netM0 = connectLayers(netM0,'Drop1','Conv4');
netM0 = connectLayers(netM0,'Conv4','Relu4');
netM0 = connectLayers(netM0,'Relu4','Drop2');
netM0 = connectLayers(netM0,'Drop2','Conv5');
netM0 = connectLayers(netM0,'Conv5','Soft');
%netM0 = connectLayers(netM0,'Conv5','FC1');
%netM0 = connectLayers(netM0,'FC1','Soft');
netM0 = connectLayers(netM0,'Soft','Out');
%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0.Layers
figure()
plot(netM0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%