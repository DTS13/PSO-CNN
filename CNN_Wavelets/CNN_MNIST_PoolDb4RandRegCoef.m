%%%%%%%%%%%%%%%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  CNN MNIST Pool Random Region Db4 Coefficients (LH, HL & HH)  %%%

%clear all;
%close all;
%clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
In = imageInputLayer([28 28 1],'Name','In');
Batch1 = batchNormalizationLayer('Name','Batch1');
Conv1 = convolution2dLayer([5,5],20,'Name','Conv1');
Batch2 = batchNormalizationLayer('Name','Batch2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POOL 1 Definition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftV1D1 = ShiftLayerV('ShiftV1D1'); 
Odd1D1 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Odd1D1'); %Odd
Even1D1 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Even1D1'); %Even
MultPA1D1 = MultLayerDb4P1('MultPA1D1');
MultUA1D1 = MultLayerDb4U11('MultUA1D1');
MultUB1D1 = MultLayerDb4U12('MultUB1D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftVnZ1D1 = ShiftLayerDb4VnegZ('ShiftVnZ1D1');
ShiftVpZ1D1 = ShiftLayerDb4VposZ('ShiftVpZ1D1');
ShiftHnZ2D1 = ShiftLayerDb4HnegZ('ShiftHnZ2D1');
ShiftHpZ2D1 = ShiftLayerDb4HposZ('ShiftHpZ2D1');
ShiftHnZ3D1 = ShiftLayerDb4HnegZ('ShiftHnZ3D1');
ShiftHpZ3D1 = ShiftLayerDb4HposZ('ShiftHpZ3D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Add1aD1 = additionLayer(2,'Name','Add1aD1');
Add1bD1 = additionLayer(2,'Name','Add1bD1');
MultNA1D1 = MultDb4NormAprox('MultNA1D1');
Sub1aD1 = additionLayer(2,'Name','Sub1aD1');
Sub1bD1 = additionLayer(2,'Name','Sub1bD1');
MultND1D1 = MultDb4NormDetail('MultND1D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH2D1 = ShiftLayerH('ShiftH2D1'); 
Odd2D1 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd2D1'); %Odd
Even2D1 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even2D1'); %Even
MultPA2D1 = MultLayerDb4P1('MultPA2D1');
MultUA2D1 = MultLayerDb4U11('MultUA2D1');
MultUB2D1 = MultLayerDb4U12('MultUB2D1');
Add2aD1 = additionLayer(2,'Name','Add2aD1');
Add2bD1 = additionLayer(2,'Name','Add2bD1');
MultNA2D1 = MultDb4NormAprox('MultNA2D1');
Sub2aD1 = additionLayer(2,'Name','Sub2aD1');
Sub2bD1 = additionLayer(2,'Name','Sub2bD1');
MultND2D1 = MultDb4NormDetail('MultND2D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH3D1 = ShiftLayerH('ShiftH3D1'); 
Odd3D1 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd3D1'); %Odd
Even3D1 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even3D1'); %Even
MultPA3D1 = MultLayerDb4P1('MultPA3D1');
MultUA3D1 = MultLayerDb4U11('MultUA3D1');
MultUB3D1 = MultLayerDb4U12('MultUB3D1');
Add3aD1 = additionLayer(2,'Name','Add3aD1');
Add3bD1 = additionLayer(2,'Name','Add3bD1');
MultNA3D1 = MultDb4NormAprox('MultNA3D1');
Sub3aD1 = additionLayer(2,'Name','Sub3aD1');
Sub3bD1 = additionLayer(2,'Name','Sub3bD1');
MultND3D1 = MultDb4NormDetail('MultND3D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Relu11D1 = leakyReluLayer('Name','Relu11D1');
Relu12D1 = leakyReluLayer('Name','Relu12D1');
Relu21D1 = leakyReluLayer('Name','Relu21D1');
Relu22D1 = leakyReluLayer('Name','Relu22D1');
Relu31D1 = leakyReluLayer('Name','Relu31D1');
Relu32D1 = leakyReluLayer('Name','Relu32D1');
Selec1 = randRegWavpoolLayer('Selec1');
Cat1 = concatenationLayer(3,2,'Name','Cat1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POOL 1 End
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Conv2 = convolution2dLayer([5,5],50,'Name','Conv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POOL 2 Definition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftV1D2 = ShiftLayerV('ShiftV1D2'); 
Odd1D2 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Odd1D2'); %Odd
Even1D2 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Even1D2'); %Even
MultPA1D2 = MultLayerDb4P1('MultPA1D2');
MultUA1D2 = MultLayerDb4U11('MultUA1D2');
MultUB1D2 = MultLayerDb4U12('MultUB1D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftVnZ1D2 = ShiftLayerDb4VnegZ('ShiftVnZ1D2');
ShiftVpZ1D2 = ShiftLayerDb4VposZ('ShiftVpZ1D2');
ShiftHnZ2D2 = ShiftLayerDb4HnegZ('ShiftHnZ2D2');
ShiftHpZ2D2 = ShiftLayerDb4HposZ('ShiftHpZ2D2');
ShiftHnZ3D2 = ShiftLayerDb4HnegZ('ShiftHnZ3D2');
ShiftHpZ3D2 = ShiftLayerDb4HposZ('ShiftHpZ3D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Add1aD2 = additionLayer(2,'Name','Add1aD2');
Add1bD2 = additionLayer(2,'Name','Add1bD2');
MultNA1D2 = MultDb4NormAprox('MultNA1D2');
Sub1aD2 = additionLayer(2,'Name','Sub1aD2');
Sub1bD2 = additionLayer(2,'Name','Sub1bD2');
MultND1D2 = MultDb4NormDetail('MultND1D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH2D2 = ShiftLayerH('ShiftH2D2'); 
Odd2D2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd2D2'); %Odd
Even2D2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even2D2'); %Even
MultPA2D2 = MultLayerDb4P1('MultPA2D2');
MultUA2D2 = MultLayerDb4U11('MultUA2D2');
MultUB2D2 = MultLayerDb4U12('MultUB2D2');
Add2aD2 = additionLayer(2,'Name','Add2aD2');
Add2bD2 = additionLayer(2,'Name','Add2bD2');
MultNA2D2 = MultDb4NormAprox('MultNA2D2');
Sub2aD2 = additionLayer(2,'Name','Sub2aD2');
Sub2bD2 = additionLayer(2,'Name','Sub2bD2');
MultND2D2 = MultDb4NormDetail('MultND2D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH3D2 = ShiftLayerH('ShiftH3D2'); 
Odd3D2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd3D2'); %Odd
Even3D2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even3D2'); %Even
MultPA3D2 = MultLayerDb4P1('MultPA3D2');
MultUA3D2 = MultLayerDb4U11('MultUA3D2');
MultUB3D2 = MultLayerDb4U12('MultUB3D2');
Add3aD2 = additionLayer(2,'Name','Add3aD2');
Add3bD2 = additionLayer(2,'Name','Add3bD2');
MultNA3D2 = MultDb4NormAprox('MultNA3D2');
Sub3aD2 = additionLayer(2,'Name','Sub3aD2');
Sub3bD2 = additionLayer(2,'Name','Sub3bD2');
MultND3D2 = MultDb4NormDetail('MultND3D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Relu11D2 = leakyReluLayer('Name','Relu11D2');
Relu12D2 = leakyReluLayer('Name','Relu12D2');
Relu21D2 = leakyReluLayer('Name','Relu21D2');
Relu22D2 = leakyReluLayer('Name','Relu22D2');
Relu31D2 = leakyReluLayer('Name','Relu31D2');
Relu32D2 = leakyReluLayer('Name','Relu32D2');
Selec2 = randRegWavpoolLayer('Selec2');
Cat2 = concatenationLayer(3,2,'Name','Cat2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POOL 2 End
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Batch3 = batchNormalizationLayer('Name','Batch3');
Conv3 = convolution2dLayer([4,4],500,'Name','Conv3');
Relu1 = reluLayer('Name','Relu1');
Conv4 = convolution2dLayer([1,1],4,'Name','Conv4'); % Number of Classes: 4
Batch4 = batchNormalizationLayer('Name','Batch4');
Soft = softmaxLayer('Name','Soft');
Out = classificationLayer('Name','Out');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = layerGraph;
netM0 = addLayers(netM0,In);
netM0 = addLayers(netM0,Conv1);
netM0 = addLayers(netM0,Conv2);
netM0 = addLayers(netM0,Conv3);
netM0 = addLayers(netM0,Conv4);
netM0 = addLayers(netM0,Batch1);
netM0 = addLayers(netM0,Batch2);
netM0 = addLayers(netM0,Batch3);
netM0 = addLayers(netM0,Batch4);
netM0 = addLayers(netM0,Relu1);
netM0 = addLayers(netM0,Soft);
netM0 = addLayers(netM0,Out);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POOL 1 Declared into the Net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftV1D1);
netM0 = addLayers(netM0,Odd1D1);
netM0 = addLayers(netM0,Even1D1);
netM0 = addLayers(netM0,MultPA1D1);
netM0 = addLayers(netM0,MultUA1D1);
netM0 = addLayers(netM0,MultUB1D1);
netM0 = addLayers(netM0,ShiftVnZ1D1);
netM0 = addLayers(netM0,ShiftVpZ1D1);
netM0 = addLayers(netM0,ShiftHnZ2D1);
netM0 = addLayers(netM0,ShiftHpZ2D1);
netM0 = addLayers(netM0,ShiftHnZ3D1);
netM0 = addLayers(netM0,ShiftHpZ3D1);
netM0 = addLayers(netM0,Add1aD1);
netM0 = addLayers(netM0,Add1bD1);
netM0 = addLayers(netM0,MultNA1D1);
netM0 = addLayers(netM0,Sub1aD1);
netM0 = addLayers(netM0,Sub1bD1);
netM0 = addLayers(netM0,MultND1D1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftH2D1);
netM0 = addLayers(netM0,Odd2D1);
netM0 = addLayers(netM0,Even2D1);
netM0 = addLayers(netM0,MultPA2D1);
netM0 = addLayers(netM0,MultUA2D1);
netM0 = addLayers(netM0,MultUB2D1);
netM0 = addLayers(netM0,Add2aD1);
netM0 = addLayers(netM0,Add2bD1);
netM0 = addLayers(netM0,MultNA2D1);
netM0 = addLayers(netM0,Sub2aD1);
netM0 = addLayers(netM0,Sub2bD1);
netM0 = addLayers(netM0,MultND2D1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftH3D1);
netM0 = addLayers(netM0,Odd3D1);
netM0 = addLayers(netM0,Even3D1);
netM0 = addLayers(netM0,MultPA3D1);
netM0 = addLayers(netM0,MultUA3D1);
netM0 = addLayers(netM0,MultUB3D1);
netM0 = addLayers(netM0,Add3aD1);
netM0 = addLayers(netM0,Add3bD1);
netM0 = addLayers(netM0,MultNA3D1);
netM0 = addLayers(netM0,Sub3aD1);
netM0 = addLayers(netM0,Sub3bD1);
netM0 = addLayers(netM0,MultND3D1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Relu11D1);
netM0 = addLayers(netM0,Relu12D1);
netM0 = addLayers(netM0,Relu21D1);
netM0 = addLayers(netM0,Relu22D1);
netM0 = addLayers(netM0,Relu31D1);
netM0 = addLayers(netM0,Relu32D1);
netM0 = addLayers(netM0,Selec1);
netM0 = addLayers(netM0,Cat1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POOL 1 End
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POOL 2 Declared into the Net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftVnZ1D2);
netM0 = addLayers(netM0,ShiftVpZ1D2);
netM0 = addLayers(netM0,ShiftHnZ2D2);
netM0 = addLayers(netM0,ShiftHpZ2D2);
netM0 = addLayers(netM0,ShiftHnZ3D2);
netM0 = addLayers(netM0,ShiftHpZ3D2);
netM0 = addLayers(netM0,ShiftV1D2);
netM0 = addLayers(netM0,Odd1D2);
netM0 = addLayers(netM0,Even1D2);
netM0 = addLayers(netM0,MultPA1D2);
netM0 = addLayers(netM0,MultUA1D2);
netM0 = addLayers(netM0,MultUB1D2);
netM0 = addLayers(netM0,Add1aD2);
netM0 = addLayers(netM0,Add1bD2);
netM0 = addLayers(netM0,MultNA1D2);
netM0 = addLayers(netM0,Sub1aD2);
netM0 = addLayers(netM0,Sub1bD2);
netM0 = addLayers(netM0,MultND1D2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftH2D2);
netM0 = addLayers(netM0,Odd2D2);
netM0 = addLayers(netM0,Even2D2);
netM0 = addLayers(netM0,MultPA2D2);
netM0 = addLayers(netM0,MultUA2D2);
netM0 = addLayers(netM0,MultUB2D2);
netM0 = addLayers(netM0,Add2aD2);
netM0 = addLayers(netM0,Add2bD2);
netM0 = addLayers(netM0,MultNA2D2);
netM0 = addLayers(netM0,Sub2aD2);
netM0 = addLayers(netM0,Sub2bD2);
netM0 = addLayers(netM0,MultND2D2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftH3D2);
netM0 = addLayers(netM0,Odd3D2);
netM0 = addLayers(netM0,Even3D2);
netM0 = addLayers(netM0,MultPA3D2);
netM0 = addLayers(netM0,MultUA3D2);
netM0 = addLayers(netM0,MultUB3D2);
netM0 = addLayers(netM0,Add3aD2);
netM0 = addLayers(netM0,Add3bD2);
netM0 = addLayers(netM0,MultNA3D2);
netM0 = addLayers(netM0,Sub3aD2);
netM0 = addLayers(netM0,Sub3bD2);
netM0 = addLayers(netM0,MultND3D2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Relu11D2);
netM0 = addLayers(netM0,Relu12D2);
netM0 = addLayers(netM0,Relu21D2);
netM0 = addLayers(netM0,Relu22D2);
netM0 = addLayers(netM0,Relu31D2);
netM0 = addLayers(netM0,Relu32D2);
netM0 = addLayers(netM0,Selec2);
netM0 = addLayers(netM0,Cat2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POOL 2 End
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'In','Batch1');
netM0 = connectLayers(netM0,'Batch1','Conv1');
netM0 = connectLayers(netM0,'Conv1','Batch2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POOL 1 Connections
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Batch2','Odd1D1');           % Odd11
netM0 = connectLayers(netM0,'Batch2','ShiftV1D1');
netM0 = connectLayers(netM0,'ShiftV1D1','Even1D1');       % Even11
netM0 = connectLayers(netM0,'Even1D1','MultPA1D1');       % P1*E11
netM0 = connectLayers(netM0,'Even1D1','Add1aD1/in1');     % s1 = E11 + __
netM0 = connectLayers(netM0,'Odd1D1','Sub1aD1/in1');      % d1 = O11 + __
netM0 = connectLayers(netM0,'Sub1aD1','MultUA1D1');       % U11*d1
netM0 = connectLayers(netM0,'MultUA1D1','Add1aD1/in2');   % s1 = E11 + U11*d1
netM0 = connectLayers(netM0,'MultPA1D1','Sub1aD1/in2');   % d1 = O11 + P1*E11
netM0 = connectLayers(netM0,'Sub1aD1','ShiftVnZ1D1');     % d11 = d1*Z^(-1)
netM0 = connectLayers(netM0,'ShiftVnZ1D1','MultUB1D1');   % U12*d11
netM0 = connectLayers(netM0,'MultUB1D1','Add1bD1/in1');   % s2 = __ + U12*d11
netM0 = connectLayers(netM0,'Add1aD1','Add1bD1/in2');     % s2 = s1 + U12*d11
netM0 = connectLayers(netM0,'Add1bD1','ShiftVpZ1D1');     % s22 = s2*Z^(+1)
netM0 = connectLayers(netM0,'ShiftVpZ1D1','Sub1bD1/in1'); % d2 = __ + s22
netM0 = connectLayers(netM0,'Sub1aD1','Sub1bD1/in2');     % d2 = d1 + s22
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Add1bD1','MultNA1D1');       % NormAprox 11 * s2
netM0 = connectLayers(netM0,'MultNA1D1','Relu11D1');      
netM0 = connectLayers(netM0,'MultND1D1','Relu12D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Relu11D1','Odd2D1');         % Odd21
netM0 = connectLayers(netM0,'Relu11D1','ShiftH2D1');
netM0 = connectLayers(netM0,'ShiftH2D1','Even2D1');       % Even21
netM0 = connectLayers(netM0,'Even2D1','MultPA2D1');       % P1*E21
netM0 = connectLayers(netM0,'Even2D1','Add2aD1/in1');     % s1 = E21 + _
netM0 = connectLayers(netM0,'Odd2D1','Sub2aD1/in1');      % d1 = O21 + _
netM0 = connectLayers(netM0,'Sub2aD1','MultUA2D1');       % U11*d1
netM0 = connectLayers(netM0,'MultUA2D1','Add2aD1/in2');   % s1 = E21 + U11*d1
netM0 = connectLayers(netM0,'MultPA2D1','Sub2aD1/in2');   % d1 = O21 + P1*E21
netM0 = connectLayers(netM0,'Sub2aD1','ShiftHnZ2D1');     % d11 = d1*Z^(-1)
netM0 = connectLayers(netM0,'ShiftHnZ2D1','MultUB2D1');   % U12*d11
netM0 = connectLayers(netM0,'MultUB2D1','Add2bD1/in1');   % s2 = _ + U12*d11
netM0 = connectLayers(netM0,'Add2aD1','Add2bD1/in2');     % s2 = s1 + U12*d11
netM0 = connectLayers(netM0,'Add2bD1','ShiftHpZ2D1');     % s22 = s2*Z^(+1)
netM0 = connectLayers(netM0,'ShiftHpZ2D1','Sub2bD1/in1'); % d2 = __ + s22
netM0 = connectLayers(netM0,'Sub2aD1','Sub2bD1/in2');     % d2 = d1 + s22
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Sub1bD1','MultND1D1');       % NormDetail 11 * d2
netM0 = connectLayers(netM0,'Relu12D1','Odd3D1');         % Odd31
netM0 = connectLayers(netM0,'Relu12D1','ShiftH3D1');
netM0 = connectLayers(netM0,'ShiftH3D1','Even3D1');       % Even31
netM0 = connectLayers(netM0,'Even3D1','MultPA3D1');       % P1*E31
netM0 = connectLayers(netM0,'Even3D1','Add3aD1/in1');     % s1 = E31 + _
netM0 = connectLayers(netM0,'Odd3D1','Sub3aD1/in1');      % d1 = O31 + _
netM0 = connectLayers(netM0,'Sub3aD1','MultUA3D1');       % U11*d1
netM0 = connectLayers(netM0,'MultUA3D1','Add3aD1/in2');   % s1 = E31 + U11*d1 
netM0 = connectLayers(netM0,'MultPA3D1','Sub3aD1/in2');   % d1 = O31 + P1*E31
netM0 = connectLayers(netM0,'Sub3aD1','ShiftHnZ3D1');     % d11 = d1*Z^(-1)
netM0 = connectLayers(netM0,'ShiftHnZ3D1','MultUB3D1');   % U12*d11
netM0 = connectLayers(netM0,'MultUB3D1','Add3bD1/in1');   % s2 = _ + U12*d11
netM0 = connectLayers(netM0,'Add3aD1','Add3bD1/in2');     % s2 = s1 + U12*d11
netM0 = connectLayers(netM0,'Add3bD1','ShiftHpZ3D1');     % s22 = s2*Z^(+1)
netM0 = connectLayers(netM0,'ShiftHpZ3D1','Sub3bD1/in1'); % d2 = __ + s22
netM0 = connectLayers(netM0,'Sub3aD1','Sub3bD1/in2');     % d2 = d1 + s22
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Add2bD1','MultNA2D1');       % NormAprox 21 * s2
netM0 = connectLayers(netM0,'MultNA2D1','Relu21D1'); % LL
netM0 = connectLayers(netM0,'Relu21D1','Cat1/in1');
netM0 = connectLayers(netM0,'Sub2bD1','MultND2D1');       % NormDetail 21 * d2
netM0 = connectLayers(netM0,'MultND2D1','Relu22D1'); % LH
netM0 = connectLayers(netM0,'Relu22D1','Selec1/in1');
netM0 = connectLayers(netM0,'Add3bD1','MultNA3D1');       % NormAprox 31 * s2
netM0 = connectLayers(netM0,'MultNA3D1','Relu31D1'); % HL
netM0 = connectLayers(netM0,'Relu31D1','Selec1/in2');
netM0 = connectLayers(netM0,'Sub3bD1','MultND3D1');       % NormDetail 31 * d2
netM0 = connectLayers(netM0,'MultND3D1','Relu32D1'); % HH
netM0 = connectLayers(netM0,'Relu32D1','Selec1/in3');
netM0 = connectLayers(netM0,'Selec1','Cat1/in2');
netM0 = connectLayers(netM0,'Cat1','Conv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POOL 1 End
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POOL 2 Connections
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Conv2','Odd1D2');            % Odd12
netM0 = connectLayers(netM0,'Conv2','ShiftV1D2');
netM0 = connectLayers(netM0,'ShiftV1D2','Even1D2');       % Even12
netM0 = connectLayers(netM0,'Even1D2','MultPA1D2');       % P1*E12
netM0 = connectLayers(netM0,'Even1D2','Add1aD2/in1');     % s1 = E12 + _
netM0 = connectLayers(netM0,'Odd1D2','Sub1aD2/in1');      % d1 = O12 + _
netM0 = connectLayers(netM0,'Sub1aD2','MultUA1D2');       % U11*d1
netM0 = connectLayers(netM0,'MultUA1D2','Add1aD2/in2');   % s1 = E12 + U11*d1
netM0 = connectLayers(netM0,'MultPA1D2','Sub1aD2/in2');   % d1 = O12 + P1*E12
netM0 = connectLayers(netM0,'Sub1aD2','ShiftVnZ1D2');     % d11 = d1*Z^(-1)
netM0 = connectLayers(netM0,'ShiftVnZ1D2','MultUB1D2');   % U12*d11
netM0 = connectLayers(netM0,'MultUB1D2','Add1bD2/in1');   % s2 = _ + U12*d11
netM0 = connectLayers(netM0,'Add1aD2','Add1bD2/in2');     % s2 = s1 + U12*d11
netM0 = connectLayers(netM0,'Add1bD2','ShiftVpZ1D2');     % s22 = s2*Z^(+1)
netM0 = connectLayers(netM0,'ShiftVpZ1D2','Sub1bD2/in1'); % d2 = __ + s22
netM0 = connectLayers(netM0,'Sub1aD2','Sub1bD2/in2');     % d2 = d1 + s22
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Add1bD2','MultNA1D2');       % NormAprox 11 * s2
netM0 = connectLayers(netM0,'MultNA1D2','Relu11D2');      % Odd22
netM0 = connectLayers(netM0,'MultND1D2','Relu12D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Relu11D2','Odd2D2');         % Odd22
netM0 = connectLayers(netM0,'Relu11D2','ShiftH2D2');
netM0 = connectLayers(netM0,'ShiftH2D2','Even2D2');       % Even22
netM0 = connectLayers(netM0,'Even2D2','MultPA2D2');       % P1*E22
netM0 = connectLayers(netM0,'Even2D2','Add2aD2/in1');     % s1 = E22 + _
netM0 = connectLayers(netM0,'Odd2D2','Sub2aD2/in1');      % d1 = O22 + _
netM0 = connectLayers(netM0,'Sub2aD2','MultUA2D2');       % U11*d1
netM0 = connectLayers(netM0,'MultUA2D2','Add2aD2/in2');   % s1 = E22 + U11*d1
netM0 = connectLayers(netM0,'MultPA2D2','Sub2aD2/in2');   % d1 = O22 + P1*E22
netM0 = connectLayers(netM0,'Sub2aD2','ShiftHnZ2D2');     % d11 = d1*Z^(-1)
netM0 = connectLayers(netM0,'ShiftHnZ2D2','MultUB2D2');   % U12*d11
netM0 = connectLayers(netM0,'MultUB2D2','Add2bD2/in1');   % s2 = _ + U12*d11
netM0 = connectLayers(netM0,'Add2aD2','Add2bD2/in2');     % s2 = s1 + U12*d11
netM0 = connectLayers(netM0,'Add2bD2','ShiftHpZ2D2');     % s22 = s2*Z^(+1)
netM0 = connectLayers(netM0,'ShiftHpZ2D2','Sub2bD2/in1'); % d2 = __ + s22
netM0 = connectLayers(netM0,'Sub2aD2','Sub2bD2/in2');     % d2 = d1 + s22
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Sub1bD2','MultND1D2');        % NormDetail 12 * d2
netM0 = connectLayers(netM0,'Relu12D2','Odd3D2');         % Odd32
netM0 = connectLayers(netM0,'Relu12D2','ShiftH3D2');
netM0 = connectLayers(netM0,'ShiftH3D2','Even3D2');        % Even32
netM0 = connectLayers(netM0,'Even3D2','MultPA3D2');        % P1*E32
netM0 = connectLayers(netM0,'Even3D2','Add3aD2/in1');      % s1 = E32 + _
netM0 = connectLayers(netM0,'Odd3D2','Sub3aD2/in1');       % d1 = O32 + _
netM0 = connectLayers(netM0,'Sub3aD2','MultUA3D2');        % U11*d1
netM0 = connectLayers(netM0,'MultUA3D2','Add3aD2/in2');    % s1 = E32 + U11*d1
netM0 = connectLayers(netM0,'MultPA3D2','Sub3aD2/in2');    % d1 = O32 + P1*E32
netM0 = connectLayers(netM0,'Sub3aD2','ShiftHnZ3D2');      % d11 = d1*Z^(-1)
netM0 = connectLayers(netM0,'ShiftHnZ3D2','MultUB3D2');    % U12*d11
netM0 = connectLayers(netM0,'MultUB3D2','Add3bD2/in1');   % s2 = _ + U12*d11
netM0 = connectLayers(netM0,'Add3aD2','Add3bD2/in2');     % s2 = s1 + U12*d11
netM0 = connectLayers(netM0,'Add3bD2','ShiftHpZ3D2');     % s22 = s2*Z^(+1)
netM0 = connectLayers(netM0,'ShiftHpZ3D2','Sub3bD2/in1'); % d2 = __ + s22
netM0 = connectLayers(netM0,'Sub3aD2','Sub3bD2/in2');     % d2 = d1 + s22
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Add2bD2','MultNA2D2');       % NormAprox 22 * s2
netM0 = connectLayers(netM0,'MultNA2D2','Relu21D2'); % LL
netM0 = connectLayers(netM0,'Relu21D2','Cat2/in1');
netM0 = connectLayers(netM0,'Sub2bD2','MultND2D2');       % NormDetail 22 * d2
netM0 = connectLayers(netM0,'MultND2D2','Relu22D2'); % LH
netM0 = connectLayers(netM0,'Relu22D2','Selec2/in1');
netM0 = connectLayers(netM0,'Add3bD2','MultNA3D2');       % NormAprox 32 * s2
netM0 = connectLayers(netM0,'MultNA3D2','Relu31D2'); % HL
netM0 = connectLayers(netM0,'Relu31D2','Selec2/in2');
netM0 = connectLayers(netM0,'Sub3bD2','MultND3D2');       % NormDetail 32 * d2
netM0 = connectLayers(netM0,'MultND3D2','Relu32D2'); % HH
netM0 = connectLayers(netM0,'Relu32D2','Selec2/in3');
netM0 = connectLayers(netM0,'Selec2','Cat2/in2');
netM0 = connectLayers(netM0,'Cat2','Batch3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POOL 2 End
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Batch3','Conv3');
netM0 = connectLayers(netM0,'Conv3','Relu1');
netM0 = connectLayers(netM0,'Relu1','Conv4');
netM0 = connectLayers(netM0,'Conv4','Batch4');
netM0 = connectLayers(netM0,'Batch4','Soft');
netM0 = connectLayers(netM0,'Soft','Out');
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0.Layers
%figure()
%plot(netM0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%
