%%%%%%%%%%%%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%%%%%%%%%%%%
%%%  CNN MNIST Pool Db6 All Coefficients (LL, LH, HL & HH) %%%%

%clear all;
%close all;
%clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
In = imageInputLayer([28 28 1],'Name','In');
Conv1 = convolution2dLayer([5,5],20,'Name','Conv1');
Batch1 = batchNormalizationLayer('Name','Batch1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftV1D1 = ShiftLayerV('ShiftV1D1'); 
Odd1D1 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Odd1D1'); %Odd
Even1D1 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Even1D1'); %Even
MultPA1D1 = MultLayerDb6P11('MultPA1D1');
MultPB1D1 = MultLayerDb6P12('MultPB1D1');
MultPC1D1 = MultLayerDb6P2('MultPC1D1');
MultUA1D1 = MultLayerDb6U1('MultUA1D1');
MultUB1D1 = MultLayerDb6U21('MultUB1D1');
MultUC1D1 = MultLayerDb6U22('MultUC1D1');
ShiftVZp1D1 = ShiftLayerDb6VposZ('ShiftVZp1D1');
ShiftVZn1D1 = ShiftLayerDb6VnegZ('ShiftVZn1D1');
Add1aD1 = additionLayer(2,'Name','Add1aD1');
Add1bD1 = additionLayer(2,'Name','Add1bD1');
Add1cD1 = additionLayer(2,'Name','Add1cD1');
MultNA1D1 = MultDb6NormAprox('MultNA1D1');
Sub1aD1 = additionLayer(2,'Name','Sub1aD1');
Sub1bD1 = additionLayer(2,'Name','Sub1bD1');
Sub1cD1 = additionLayer(2,'Name','Sub1cD1');
MultND1D1 = MultDb6NormDetail('MultND1D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH2D1 = ShiftLayerH('ShiftH2D1'); 
Odd2D1 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd2D1'); %Odd
Even2D1 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even2D1'); %Even
MultPA2D1 = MultLayerDb6P11('MultPA2D1');
MultPB2D1 = MultLayerDb6P12('MultPB2D1');
MultPC2D1 = MultLayerDb6P2('MultPC2D1');
MultUA2D1 = MultLayerDb6U1('MultUA2D1');
MultUB2D1 = MultLayerDb6U21('MultUB2D1');
MultUC2D1 = MultLayerDb6U22('MultUC2D1');
ShiftHZp2D1 = ShiftLayerDb6HposZ('ShiftHZp2D1');
ShiftHZn2D1 = ShiftLayerDb6HnegZ('ShiftHZn2D1');
Add2aD1 = additionLayer(2,'Name','Add2aD1');
Add2bD1 = additionLayer(2,'Name','Add2bD1');
Add2cD1 = additionLayer(2,'Name','Add2cD1');
MultNA2D1 = MultDb6NormAprox('MultNA2D1');
Sub2aD1 = additionLayer(2,'Name','Sub2aD1');
Sub2bD1 = additionLayer(2,'Name','Sub2bD1');
Sub2cD1 = additionLayer(2,'Name','Sub2cD1');
MultND2D1 = MultDb6NormDetail('MultND2D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH3D1 = ShiftLayerH('ShiftH3D1'); 
Odd3D1 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd3D1'); %Odd
Even3D1 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even3D1'); %Even
MultPA3D1 = MultLayerDb6P11('MultPA3D1');
MultPB3D1 = MultLayerDb6P12('MultPB3D1');
MultPC3D1 = MultLayerDb6P2('MultPC3D1');
MultUA3D1 = MultLayerDb6U1('MultUA3D1');
MultUB3D1 = MultLayerDb6U21('MultUB3D1');
MultUC3D1 = MultLayerDb6U22('MultUC3D1');
ShiftHZp3D1 = ShiftLayerDb6HposZ('ShiftHZp3D1');
ShiftHZn3D1 = ShiftLayerDb6HnegZ('ShiftHZn3D1');
Add3aD1 = additionLayer(2,'Name','Add3aD1');
Add3bD1 = additionLayer(2,'Name','Add3bD1');
Add3cD1 = additionLayer(2,'Name','Add3cD1');
MultNA3D1 = MultDb6NormAprox('MultNA3D1');
Sub3aD1 = additionLayer(2,'Name','Sub3aD1');
Sub3bD1 = additionLayer(2,'Name','Sub3bD1');
Sub3cD1 = additionLayer(2,'Name','Sub3cD1');
MultND3D1 = MultDb6NormDetail('MultND3D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Cat1 = concatenationLayer(3,4,'Name','Cat1');
Relu11D1 = leakyReluLayer('Name','Relu11D1');
Relu12D1 = leakyReluLayer('Name','Relu12D1');
Relu21D1 = leakyReluLayer('Name','Relu21D1');
Relu22D1 = leakyReluLayer('Name','Relu22D1');
Relu31D1 = leakyReluLayer('Name','Relu31D1');
Relu32D1 = leakyReluLayer('Name','Relu32D1');
Conv2 = convolution2dLayer([5,5],50,'Name','Conv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftV1D2 = ShiftLayerV('ShiftV1D2'); 
Odd1D2 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Odd1D2'); %Odd
Even1D2 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Even1D2'); %Even
MultPA1D2 = MultLayerDb6P11('MultPA1D2');
MultPB1D2 = MultLayerDb6P12('MultPB1D2');
MultPC1D2 = MultLayerDb6P2('MultPC1D2');
MultUA1D2 = MultLayerDb6U1('MultUA1D2');
MultUB1D2 = MultLayerDb6U21('MultUB1D2');
MultUC1D2 = MultLayerDb6U22('MultUC1D2');
ShiftVZp1D2 = ShiftLayerDb6VposZ('ShiftVZp1D2');
ShiftVZn1D2 = ShiftLayerDb6VnegZ('ShiftVZn1D2');
Add1aD2 = additionLayer(2,'Name','Add1aD2');
Add1bD2 = additionLayer(2,'Name','Add1bD2');
Add1cD2 = additionLayer(2,'Name','Add1cD2');
MultNA1D2 = MultDb6NormAprox('MultNA1D2');
Sub1aD2 = additionLayer(2,'Name','Sub1aD2');
Sub1bD2 = additionLayer(2,'Name','Sub1bD2');
Sub1cD2 = additionLayer(2,'Name','Sub1cD2');
MultND1D2 = MultDb6NormDetail('MultND1D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH2D2 = ShiftLayerH('ShiftH2D2'); 
Odd2D2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd2D2'); %Odd
Even2D2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even2D2'); %Even
MultPA2D2 = MultLayerDb6P11('MultPA2D2');
MultPB2D2 = MultLayerDb6P12('MultPB2D2');
MultPC2D2 = MultLayerDb6P2('MultPC2D2');
MultUA2D2 = MultLayerDb6U1('MultUA2D2');
MultUB2D2 = MultLayerDb6U21('MultUB2D2');
MultUC2D2 = MultLayerDb6U22('MultUC2D2');
ShiftHZp2D2 = ShiftLayerDb6HposZ('ShiftHZp2D2');
ShiftHZn2D2 = ShiftLayerDb6HnegZ('ShiftHZn2D2');
Add2aD2 = additionLayer(2,'Name','Add2aD2');
Add2bD2 = additionLayer(2,'Name','Add2bD2');
Add2cD2 = additionLayer(2,'Name','Add2cD2');
MultNA2D2 = MultDb6NormAprox('MultNA2D2');
Sub2aD2 = additionLayer(2,'Name','Sub2aD2');
Sub2bD2 = additionLayer(2,'Name','Sub2bD2');
Sub2cD2 = additionLayer(2,'Name','Sub2cD2');
MultND2D2 = MultDb6NormDetail('MultND2D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH3D2 = ShiftLayerH('ShiftH3D2'); 
Odd3D2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd3D2'); %Odd
Even3D2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even3D2'); %Even
MultPA3D2 = MultLayerDb6P11('MultPA3D2');
MultPB3D2 = MultLayerDb6P12('MultPB3D2');
MultPC3D2 = MultLayerDb6P2('MultPC3D2');
MultUA3D2 = MultLayerDb6U1('MultUA3D2');
MultUB3D2 = MultLayerDb6U21('MultUB3D2');
MultUC3D2 = MultLayerDb6U22('MultUC3D2');
ShiftHZp3D2 = ShiftLayerDb6HposZ('ShiftHZp3D2');
ShiftHZn3D2 = ShiftLayerDb6HnegZ('ShiftHZn3D2');
Add3aD2 = additionLayer(2,'Name','Add3aD2');
Add3bD2 = additionLayer(2,'Name','Add3bD2');
Add3cD2 = additionLayer(2,'Name','Add3cD2');
MultNA3D2 = MultDb6NormAprox('MultNA3D2');
Sub3aD2 = additionLayer(2,'Name','Sub3aD2');
Sub3bD2 = additionLayer(2,'Name','Sub3bD2');
Sub3cD2 = additionLayer(2,'Name','Sub3cD2');
MultND3D2 = MultDb6NormDetail('MultND3D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Cat2 = concatenationLayer(3,4,'Name','Cat2');
Relu11D2 = leakyReluLayer('Name','Relu11D2');
Relu12D2 = leakyReluLayer('Name','Relu12D2');
Relu21D2 = leakyReluLayer('Name','Relu21D2');
Relu22D2 = leakyReluLayer('Name','Relu22D2');
Relu31D2 = leakyReluLayer('Name','Relu31D2');
Relu32D2 = leakyReluLayer('Name','Relu32D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Batch2 = batchNormalizationLayer('Name','Batch2');
Conv3 = convolution2dLayer([4,4],500,'Name','Conv3');
Relu1 = reluLayer('Name','Relu1');
Conv4 = convolution2dLayer([1,1],10,'Name','Conv4');
Batch3 = batchNormalizationLayer('Name','Batch3');
Soft = softmaxLayer('Name','Soft');
Out = classificationLayer('Name','Out');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = layerGraph;
netM0 = addLayers(netM0,In);
netM0 = addLayers(netM0,Conv1);
netM0 = addLayers(netM0,Conv2);
netM0 = addLayers(netM0,Conv3);
netM0 = addLayers(netM0,Conv4);
netM0 = addLayers(netM0,Batch1);
netM0 = addLayers(netM0,Batch2);
netM0 = addLayers(netM0,Batch3);
netM0 = addLayers(netM0,Relu11D1);
netM0 = addLayers(netM0,Relu12D1);
netM0 = addLayers(netM0,Relu21D1);
netM0 = addLayers(netM0,Relu22D1);
netM0 = addLayers(netM0,Relu31D1);
netM0 = addLayers(netM0,Relu32D1);
netM0 = addLayers(netM0,Relu11D2);
netM0 = addLayers(netM0,Relu12D2);
netM0 = addLayers(netM0,Relu21D2);
netM0 = addLayers(netM0,Relu22D2);
netM0 = addLayers(netM0,Relu31D2);
netM0 = addLayers(netM0,Relu32D2);
netM0 = addLayers(netM0,Cat1);
netM0 = addLayers(netM0,Cat2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftV1D1);
netM0 = addLayers(netM0,Odd1D1);
netM0 = addLayers(netM0,Even1D1);
netM0 = addLayers(netM0,MultPA1D1);
netM0 = addLayers(netM0,MultPB1D1);
netM0 = addLayers(netM0,MultPC1D1);
netM0 = addLayers(netM0,MultUA1D1);
netM0 = addLayers(netM0,MultUB1D1);
netM0 = addLayers(netM0,MultUC1D1);
netM0 = addLayers(netM0,ShiftVZp1D1);
netM0 = addLayers(netM0,ShiftVZn1D1);
netM0 = addLayers(netM0,Add1aD1);
netM0 = addLayers(netM0,Add1bD1);
netM0 = addLayers(netM0,Add1cD1);
netM0 = addLayers(netM0,MultNA1D1);
netM0 = addLayers(netM0,Sub1aD1);
netM0 = addLayers(netM0,Sub1bD1);
netM0 = addLayers(netM0,Sub1cD1);
netM0 = addLayers(netM0,MultND1D1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftH2D1);
netM0 = addLayers(netM0,Odd2D1);
netM0 = addLayers(netM0,Even2D1);
netM0 = addLayers(netM0,MultPA2D1);
netM0 = addLayers(netM0,MultPB2D1);
netM0 = addLayers(netM0,MultPC2D1);
netM0 = addLayers(netM0,MultUA2D1);
netM0 = addLayers(netM0,MultUB2D1);
netM0 = addLayers(netM0,MultUC2D1);
netM0 = addLayers(netM0,ShiftHZp2D1);
netM0 = addLayers(netM0,ShiftHZn2D1);
netM0 = addLayers(netM0,Add2aD1);
netM0 = addLayers(netM0,Add2bD1);
netM0 = addLayers(netM0,Add2cD1);
netM0 = addLayers(netM0,MultNA2D1);
netM0 = addLayers(netM0,Sub2aD1);
netM0 = addLayers(netM0,Sub2bD1);
netM0 = addLayers(netM0,Sub2cD1);
netM0 = addLayers(netM0,MultND2D1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftH3D1);
netM0 = addLayers(netM0,Odd3D1);
netM0 = addLayers(netM0,Even3D1);
netM0 = addLayers(netM0,MultPA3D1);
netM0 = addLayers(netM0,MultPB3D1);
netM0 = addLayers(netM0,MultPC3D1);
netM0 = addLayers(netM0,MultUA3D1);
netM0 = addLayers(netM0,MultUB3D1);
netM0 = addLayers(netM0,MultUC3D1);
netM0 = addLayers(netM0,ShiftHZp3D1);
netM0 = addLayers(netM0,ShiftHZn3D1);
netM0 = addLayers(netM0,Add3aD1);
netM0 = addLayers(netM0,Add3bD1);
netM0 = addLayers(netM0,Add3cD1);
netM0 = addLayers(netM0,MultNA3D1);
netM0 = addLayers(netM0,Sub3aD1);
netM0 = addLayers(netM0,Sub3bD1);
netM0 = addLayers(netM0,Sub3cD1);
netM0 = addLayers(netM0,MultND3D1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftV1D2);
netM0 = addLayers(netM0,Odd1D2);
netM0 = addLayers(netM0,Even1D2);
netM0 = addLayers(netM0,MultPA1D2);
netM0 = addLayers(netM0,MultPB1D2);
netM0 = addLayers(netM0,MultPC1D2);
netM0 = addLayers(netM0,MultUA1D2);
netM0 = addLayers(netM0,MultUB1D2);
netM0 = addLayers(netM0,MultUC1D2);
netM0 = addLayers(netM0,ShiftVZn1D2);
netM0 = addLayers(netM0,ShiftVZp1D2);
netM0 = addLayers(netM0,Add1aD2);
netM0 = addLayers(netM0,Add1bD2);
netM0 = addLayers(netM0,Add1cD2);
netM0 = addLayers(netM0,MultNA1D2);
netM0 = addLayers(netM0,Sub1aD2);
netM0 = addLayers(netM0,Sub1bD2);
netM0 = addLayers(netM0,Sub1cD2);
netM0 = addLayers(netM0,MultND1D2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftH2D2);
netM0 = addLayers(netM0,Odd2D2);
netM0 = addLayers(netM0,Even2D2);
netM0 = addLayers(netM0,MultPA2D2);
netM0 = addLayers(netM0,MultPB2D2);
netM0 = addLayers(netM0,MultPC2D2);
netM0 = addLayers(netM0,MultUA2D2);
netM0 = addLayers(netM0,MultUB2D2);
netM0 = addLayers(netM0,MultUC2D2);
netM0 = addLayers(netM0,ShiftHZp2D2);
netM0 = addLayers(netM0,ShiftHZn2D2);
netM0 = addLayers(netM0,Add2aD2);
netM0 = addLayers(netM0,Add2bD2);
netM0 = addLayers(netM0,Add2cD2);
netM0 = addLayers(netM0,MultNA2D2);
netM0 = addLayers(netM0,Sub2aD2);
netM0 = addLayers(netM0,Sub2bD2);
netM0 = addLayers(netM0,Sub2cD2);
netM0 = addLayers(netM0,MultND2D2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftH3D2);
netM0 = addLayers(netM0,Odd3D2);
netM0 = addLayers(netM0,Even3D2);
netM0 = addLayers(netM0,MultPA3D2);
netM0 = addLayers(netM0,MultPB3D2);
netM0 = addLayers(netM0,MultPC3D2);
netM0 = addLayers(netM0,MultUA3D2);
netM0 = addLayers(netM0,MultUB3D2);
netM0 = addLayers(netM0,MultUC3D2);
netM0 = addLayers(netM0,ShiftHZp3D2);
netM0 = addLayers(netM0,ShiftHZn3D2);
netM0 = addLayers(netM0,Add3aD2);
netM0 = addLayers(netM0,Add3bD2);
netM0 = addLayers(netM0,Add3cD2);
netM0 = addLayers(netM0,MultNA3D2);
netM0 = addLayers(netM0,Sub3aD2);
netM0 = addLayers(netM0,Sub3bD2);
netM0 = addLayers(netM0,Sub3cD2);
netM0 = addLayers(netM0,MultND3D2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Relu1);
netM0 = addLayers(netM0,Soft);
netM0 = addLayers(netM0,Out);
%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'In','Conv1');
netM0 = connectLayers(netM0,'Conv1','Batch1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Batch1','Odd1D1');           % Odd11
netM0 = connectLayers(netM0,'Batch1','ShiftV1D1');
netM0 = connectLayers(netM0,'ShiftV1D1','Even1D1');       % Even11
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Odd1D1','MultUA1D1');        % U1*O11
netM0 = connectLayers(netM0,'Even1D1','Add1aD1/in1');     % s1 = E11 + _
netM0 = connectLayers(netM0,'MultUA1D1','Add1aD1/in2');   % s1 = E11 + U11*O11
netM0 = connectLayers(netM0,'Add1aD1','MultPA1D1');       % P11*s1
netM0 = connectLayers(netM0,'Odd1D1','Sub1aD1/in1');      % d1 = O11 + _
netM0 = connectLayers(netM0,'MultPA1D1','Sub1aD1/in2');   % d1 = O11 + P11*s1
netM0 = connectLayers(netM0,'Add1aD1','ShiftVZp1D1');     % s11 = s1*Z^(+1)
netM0 = connectLayers(netM0,'ShiftVZp1D1','MultPB1D1');   % P12*s11
netM0 = connectLayers(netM0,'MultPB1D1','Sub1bD1/in1');   % d2 = _ + P12*s11
netM0 = connectLayers(netM0,'Sub1aD1','Sub1bD1/in2');     % d2 = d1 + P12*s11
netM0 = connectLayers(netM0,'Sub1bD1','MultUC1D1');       % U22*d2
netM0 = connectLayers(netM0,'MultUC1D1','Add1bD1/in1');   % s2 = _ + U22*d2
netM0 = connectLayers(netM0,'Add1aD1','Add1bD1/in2');     % s2 = s1 + U22*d2
netM0 = connectLayers(netM0,'Sub1bD1','ShiftVZn1D1');     % d22 = d2*Z^(-1)
netM0 = connectLayers(netM0,'ShiftVZn1D1','MultUB1D1');   % U21*d22
netM0 = connectLayers(netM0,'MultUB1D1','Add1cD1/in1');   % s3 = _ + U21*d22
netM0 = connectLayers(netM0,'Add1bD1','Add1cD1/in2');     % s3 = s2 + U21*d22
netM0 = connectLayers(netM0,'Add1cD1','MultPC1D1');       % P2*s3
netM0 = connectLayers(netM0,'MultPC1D1','Sub1cD1/in1');   % d3 = _ + P2*s3
netM0 = connectLayers(netM0,'Sub1bD1','Sub1cD1/in2');     % d3 = d2 + P2*s3
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Add1cD1','MultNA1D1');       % NormAprox 11 * s3
netM0 = connectLayers(netM0,'Sub1cD1','MultND1D1');       % NormDetail 11 * d3
netM0 = connectLayers(netM0,'MultNA1D1','Relu11D1');       % NormAprox 11 * s3
netM0 = connectLayers(netM0,'MultND1D1','Relu12D1');       % NormDetail 11 * d3
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Relu11D1','Odd2D1');        % Odd21
netM0 = connectLayers(netM0,'Relu11D1','ShiftH2D1');
netM0 = connectLayers(netM0,'ShiftH2D1','Even2D1');       % Even21
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Odd2D1','MultUA2D1');        % U1*O21
netM0 = connectLayers(netM0,'Even2D1','Add2aD1/in1');     % s1 = E21 + _
netM0 = connectLayers(netM0,'MultUA2D1','Add2aD1/in2');   % s1 = E21 + U11*O21
netM0 = connectLayers(netM0,'Add2aD1','MultPA2D1');       % P11*s1
netM0 = connectLayers(netM0,'Odd2D1','Sub2aD1/in1');      % d1 = O21 + _
netM0 = connectLayers(netM0,'MultPA2D1','Sub2aD1/in2');   % d1 = O21 + P11*s1
netM0 = connectLayers(netM0,'Add2aD1','ShiftHZp2D1');     % s11 = s1*Z^(+1)
netM0 = connectLayers(netM0,'ShiftHZp2D1','MultPB2D1');   % P12*s11
netM0 = connectLayers(netM0,'MultPB2D1','Sub2bD1/in1');   % d2 = _ + P12*s11
netM0 = connectLayers(netM0,'Sub2aD1','Sub2bD1/in2');     % d2 = d1 + P12*s11
netM0 = connectLayers(netM0,'Sub2bD1','MultUC2D1');       % U22*d2
netM0 = connectLayers(netM0,'MultUC2D1','Add2bD1/in1');   % s2 = _ + U22*d2
netM0 = connectLayers(netM0,'Add2aD1','Add2bD1/in2');     % s2 = s1 + U22*d2
netM0 = connectLayers(netM0,'Sub2bD1','ShiftHZn2D1');     % d22 = d2*Z^(-1)
netM0 = connectLayers(netM0,'ShiftHZn2D1','MultUB2D1');   % U21*d22
netM0 = connectLayers(netM0,'MultUB2D1','Add2cD1/in1');   % s3 = _ + U21*d22
netM0 = connectLayers(netM0,'Add2bD1','Add2cD1/in2');     % s3 = s2 + U21*d22
netM0 = connectLayers(netM0,'Add2cD1','MultPC2D1');       % P2*s3
netM0 = connectLayers(netM0,'MultPC2D1','Sub2cD1/in1');   % d3 = _ + P2*s3
netM0 = connectLayers(netM0,'Sub2bD1','Sub2cD1/in2');     % d3 = d2 + P2*s3
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Add2cD1','MultNA2D1');       % NormAprox 21 * s3
netM0 = connectLayers(netM0,'Sub2cD1','MultND2D1');       % NormDetail 21 * d3
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Relu12D1','Odd3D1');        % Odd31
netM0 = connectLayers(netM0,'Relu12D1','ShiftH3D1');
netM0 = connectLayers(netM0,'ShiftH3D1','Even3D1');       % Even31
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Odd3D1','MultUA3D1');        % U1*O31
netM0 = connectLayers(netM0,'Even3D1','Add3aD1/in1');     % s1 = E31 + _
netM0 = connectLayers(netM0,'MultUA3D1','Add3aD1/in2');   % s1 = E31 + U11*O31
netM0 = connectLayers(netM0,'Add3aD1','MultPA3D1');       % P11*s1
netM0 = connectLayers(netM0,'Odd3D1','Sub3aD1/in1');      % d1 = O31 + _
netM0 = connectLayers(netM0,'MultPA3D1','Sub3aD1/in2');   % d1 = O31 + P11*s1
netM0 = connectLayers(netM0,'Add3aD1','ShiftHZp3D1');     % s11 = s1*Z^(+1)
netM0 = connectLayers(netM0,'ShiftHZp3D1','MultPB3D1');   % P12*s11
netM0 = connectLayers(netM0,'MultPB3D1','Sub3bD1/in1');   % d2 = _ + P12*s11
netM0 = connectLayers(netM0,'Sub3aD1','Sub3bD1/in2');     % d2 = d1 + P12*s11
netM0 = connectLayers(netM0,'Sub3bD1','MultUC3D1');       % U22*d2
netM0 = connectLayers(netM0,'MultUC3D1','Add3bD1/in1');   % s2 = _ + U22*d2
netM0 = connectLayers(netM0,'Add3aD1','Add3bD1/in2');     % s2 = s1 + U22*d2
netM0 = connectLayers(netM0,'Sub3bD1','ShiftHZn3D1');     % d22 = d2*Z^(-1)
netM0 = connectLayers(netM0,'ShiftHZn3D1','MultUB3D1');   % U21*d22
netM0 = connectLayers(netM0,'MultUB3D1','Add3cD1/in1');   % s3 = _ + U21*d22
netM0 = connectLayers(netM0,'Add3bD1','Add3cD1/in2');     % s3 = s2 + U21*d22
netM0 = connectLayers(netM0,'Add3cD1','MultPC3D1');       % P2*s3
netM0 = connectLayers(netM0,'MultPC3D1','Sub3cD1/in1');   % d3 = _ + P2*s3
netM0 = connectLayers(netM0,'Sub3bD1','Sub3cD1/in2');     % d3 = d2 + P2*s3
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Add3cD1','MultNA3D1');       % NormAprox 31 * s3
netM0 = connectLayers(netM0,'Sub3cD1','MultND3D1');       % NormDetail 31 * d3
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'MultNA2D1','Relu21D1'); % LL
netM0 = connectLayers(netM0,'MultND2D1','Relu22D1'); % LH
netM0 = connectLayers(netM0,'MultNA3D1','Relu31D1'); % HL
netM0 = connectLayers(netM0,'MultND3D1','Relu32D1'); % LL
netM0 = connectLayers(netM0,'Relu21D1','Cat1/in1');
netM0 = connectLayers(netM0,'Relu22D1','Cat1/in2');
netM0 = connectLayers(netM0,'Relu31D1','Cat1/in3');
netM0 = connectLayers(netM0,'Relu32D1','Cat1/in4');
netM0 = connectLayers(netM0,'Cat1','Conv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Conv2','Odd1D2');            % Odd12
netM0 = connectLayers(netM0,'Conv2','ShiftV1D2');
netM0 = connectLayers(netM0,'ShiftV1D2','Even1D2');       % Even12
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Odd1D2','MultUA1D2');        % U1*O12
netM0 = connectLayers(netM0,'Even1D2','Add1aD2/in1');     % s1 = E12 + _
netM0 = connectLayers(netM0,'MultUA1D2','Add1aD2/in2');   % s1 = E12 + U11*O12
netM0 = connectLayers(netM0,'Add1aD2','MultPA1D2');       % P11*s1
netM0 = connectLayers(netM0,'Odd1D2','Sub1aD2/in1');      % d1 = O12 + _
netM0 = connectLayers(netM0,'MultPA1D2','Sub1aD2/in2');   % d1 = O12 + P11*s1
netM0 = connectLayers(netM0,'Add1aD2','ShiftVZp1D2');     % s11 = s1*Z^(+1)
netM0 = connectLayers(netM0,'ShiftVZp1D2','MultPB1D2');   % P12*s11
netM0 = connectLayers(netM0,'MultPB1D2','Sub1bD2/in1');   % d2 = _ + P12*s11
netM0 = connectLayers(netM0,'Sub1aD2','Sub1bD2/in2');     % d2 = d1 + P12*s11
netM0 = connectLayers(netM0,'Sub1bD2','MultUC1D2');       % U22*d2
netM0 = connectLayers(netM0,'MultUC1D2','Add1bD2/in1');   % s2 = _ + U22*d2
netM0 = connectLayers(netM0,'Add1aD2','Add1bD2/in2');     % s2 = s1 + U22*d2
netM0 = connectLayers(netM0,'Sub1bD2','ShiftVZn1D2');     % d22 = d2*Z^(-1)
netM0 = connectLayers(netM0,'ShiftVZn1D2','MultUB1D2');   % U21*d22
netM0 = connectLayers(netM0,'MultUB1D2','Add1cD2/in1');   % s3 = _ + U21*d22
netM0 = connectLayers(netM0,'Add1bD2','Add1cD2/in2');     % s3 = s2 + U21*d22
netM0 = connectLayers(netM0,'Add1cD2','MultPC1D2');       % P2*s3
netM0 = connectLayers(netM0,'MultPC1D2','Sub1cD2/in1');   % d3 = _ + P2*s3
netM0 = connectLayers(netM0,'Sub1bD2','Sub1cD2/in2');     % d3 = d2 + P2*s3
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Add1cD2','MultNA1D2');       % NormAprox 12 * s3
netM0 = connectLayers(netM0,'Sub1cD2','MultND1D2');       % NormDetail 12 * d3
netM0 = connectLayers(netM0,'MultNA1D2','Relu11D2');
netM0 = connectLayers(netM0,'MultND1D2','Relu12D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Relu11D2','Odd2D2');        % Odd22
netM0 = connectLayers(netM0,'Relu11D2','ShiftH2D2');
netM0 = connectLayers(netM0,'ShiftH2D2','Even2D2');       % Even22
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Odd2D2','MultUA2D2');        % U1*O22
netM0 = connectLayers(netM0,'Even2D2','Add2aD2/in1');     % s1 = E22 + _
netM0 = connectLayers(netM0,'MultUA2D2','Add2aD2/in2');   % s1 = E22 + U11*O22
netM0 = connectLayers(netM0,'Add2aD2','MultPA2D2');       % P11*s1
netM0 = connectLayers(netM0,'Odd2D2','Sub2aD2/in1');      % d1 = O22 + _
netM0 = connectLayers(netM0,'MultPA2D2','Sub2aD2/in2');   % d1 = O22 + P11*s1
netM0 = connectLayers(netM0,'Add2aD2','ShiftHZp2D2');     % s11 = s1*Z^(+1)
netM0 = connectLayers(netM0,'ShiftHZp2D2','MultPB2D2');   % P12*s11
netM0 = connectLayers(netM0,'MultPB2D2','Sub2bD2/in1');   % d2 = _ + P12*s11
netM0 = connectLayers(netM0,'Sub2aD2','Sub2bD2/in2');     % d2 = d1 + P12*s11
netM0 = connectLayers(netM0,'Sub2bD2','MultUC2D2');       % U22*d2
netM0 = connectLayers(netM0,'MultUC2D2','Add2bD2/in1');   % s2 = _ + U22*d2
netM0 = connectLayers(netM0,'Add2aD2','Add2bD2/in2');     % s2 = s1 + U22*d2
netM0 = connectLayers(netM0,'Sub2bD2','ShiftHZn2D2');     % d22 = d2*Z^(-1)
netM0 = connectLayers(netM0,'ShiftHZn2D2','MultUB2D2');   % U21*d22
netM0 = connectLayers(netM0,'MultUB2D2','Add2cD2/in1');   % s3 = _ + U21*d22
netM0 = connectLayers(netM0,'Add2bD2','Add2cD2/in2');     % s3 = s2 + U21*d22
netM0 = connectLayers(netM0,'Add2cD2','MultPC2D2');       % P2*s3
netM0 = connectLayers(netM0,'MultPC2D2','Sub2cD2/in1');   % d3 = _ + P2*s3
netM0 = connectLayers(netM0,'Sub2bD2','Sub2cD2/in2');     % d3 = d2 + P2*s3
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Add2cD2','MultNA2D2');       % NormAprox 22 * s3
netM0 = connectLayers(netM0,'Sub2cD2','MultND2D2');       % NormDetail 22 * d3
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Relu12D2','Odd3D2');        % Odd32
netM0 = connectLayers(netM0,'Relu12D2','ShiftH3D2');
netM0 = connectLayers(netM0,'ShiftH3D2','Even3D2');       % Even32
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Odd3D2','MultUA3D2');        % U1*O32
netM0 = connectLayers(netM0,'Even3D2','Add3aD2/in1');     % s1 = E32 + _
netM0 = connectLayers(netM0,'MultUA3D2','Add3aD2/in2');   % s1 = E32 + U11*O32
netM0 = connectLayers(netM0,'Add3aD2','MultPA3D2');       % P11*s1
netM0 = connectLayers(netM0,'Odd3D2','Sub3aD2/in1');      % d1 = O32 + _
netM0 = connectLayers(netM0,'MultPA3D2','Sub3aD2/in2');   % d1 = O32 + P11*s1
netM0 = connectLayers(netM0,'Add3aD2','ShiftHZp3D2');     % s11 = s1*Z^(+1)
netM0 = connectLayers(netM0,'ShiftHZp3D2','MultPB3D2');   % P12*s11
netM0 = connectLayers(netM0,'MultPB3D2','Sub3bD2/in1');   % d2 = _ + P12*s11
netM0 = connectLayers(netM0,'Sub3aD2','Sub3bD2/in2');     % d2 = d1 + P12*s11
netM0 = connectLayers(netM0,'Sub3bD2','MultUC3D2');       % U22*d2
netM0 = connectLayers(netM0,'MultUC3D2','Add3bD2/in1');   % s2 = _ + U22*d2
netM0 = connectLayers(netM0,'Add3aD2','Add3bD2/in2');     % s2 = s1 + U22*d2
netM0 = connectLayers(netM0,'Sub3bD2','ShiftHZn3D2');     % d22 = d2*Z^(-1)
netM0 = connectLayers(netM0,'ShiftHZn3D2','MultUB3D2');   % U21*d22
netM0 = connectLayers(netM0,'MultUB3D2','Add3cD2/in1');   % s3 = _ + U21*d22
netM0 = connectLayers(netM0,'Add3bD2','Add3cD2/in2');     % s3 = s2 + U21*d22
netM0 = connectLayers(netM0,'Add3cD2','MultPC3D2');       % P2*s3
netM0 = connectLayers(netM0,'MultPC3D2','Sub3cD2/in1');   % d3 = _ + P2*s3
netM0 = connectLayers(netM0,'Sub3bD2','Sub3cD2/in2');     % d3 = d2 + P2*s3
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Add3cD2','MultNA3D2');       % NormAprox 32 * s3
netM0 = connectLayers(netM0,'Sub3cD2','MultND3D2');       % NormDetail 32 * d3
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'MultNA2D2','Relu21D2');
netM0 = connectLayers(netM0,'MultND2D2','Relu22D2');
netM0 = connectLayers(netM0,'MultNA3D2','Relu31D2');
netM0 = connectLayers(netM0,'MultND3D2','Relu32D2');
netM0 = connectLayers(netM0,'Relu21D2','Cat2/in1');
netM0 = connectLayers(netM0,'Relu22D2','Cat2/in2');
netM0 = connectLayers(netM0,'Relu31D2','Cat2/in3');
netM0 = connectLayers(netM0,'Relu32D2','Cat2/in4');
netM0 = connectLayers(netM0,'Cat2','Batch2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Batch2','Conv3');
netM0 = connectLayers(netM0,'Conv3','Relu1');
netM0 = connectLayers(netM0,'Relu1','Conv4');
netM0 = connectLayers(netM0,'Conv4','Batch3');
netM0 = connectLayers(netM0,'Batch3','Soft');
netM0 = connectLayers(netM0,'Soft','Out');
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0.Layers
figure(1)
plot(netM0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%
