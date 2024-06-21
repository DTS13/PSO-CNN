%%%%%%%%%%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%%%
%%%  CNN MNIST Pool Haar All Coefficients Net E %%%%

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
MultP1D1 = MultLayerP('MultP1D1');
MultU1D1 = MultLayerU('MultU1D1');
Add1D1 = additionLayer(2,'Name','Add1D1');
Sub1D1 = additionLayer(2,'Name','Sub1D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH2D1 = ShiftLayerH2('ShiftH2D1'); 
Odd2D1 = averagePooling2dLayer([1 1],'Stride',[2 1],'Name','Odd2D1'); %Odd
Even2D1 = averagePooling2dLayer([1 1],'Stride',[2 1],'Name','Even2D1'); %Even
MultP2D1 = MultLayerP('MultP2D1');
MultU2D1 = MultLayerU('MultU2D1');
Add2D1 = additionLayer(2,'Name','Add2D1');
Sub2D1 = additionLayer(2,'Name','Sub2D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH3D1 = ShiftLayerH2('ShiftH3D1'); 
Odd3D1 = averagePooling2dLayer([1 1],'Stride',[2 1],'Name','Odd3D1'); %Odd
Even3D1 = averagePooling2dLayer([1 1],'Stride',[2 1],'Name','Even3D1'); %Even
MultP3D1 = MultLayerP('MultP3D1');
MultU3D1 = MultLayerU('MultU3D1');
Add3D1 = additionLayer(2,'Name','Add3D1');
Sub3D1 = additionLayer(2,'Name','Sub3D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Cat1 = concatenationLayer(3,4,'Name','Cat1');
Conv2 = convolution2dLayer([5,5],50,'Name','Conv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftV1D2 = ShiftLayerV('ShiftV1D2'); 
Odd1D2 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Odd1D2'); %Odd
Even1D2 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Even1D2'); %Even
MultP1D2 = MultLayerP('MultP1D2');
MultU1D2 = MultLayerU('MultU1D2');
Add1D2 = additionLayer(2,'Name','Add1D2');
Sub1D2 = additionLayer(2,'Name','Sub1D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH2D2 = ShiftLayerH2('ShiftH2D2'); 
Odd2D2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd2D2'); %Odd
Even2D2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even2D2'); %Even
MultP2D2 = MultLayerP('MultP2D2');
MultU2D2 = MultLayerU('MultU2D2');
Add2D2 = additionLayer(2,'Name','Add2D2');
Sub2D2 = additionLayer(2,'Name','Sub2D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH3D2 = ShiftLayerH2('ShiftH3D2'); 
Odd3D2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd3D2'); %Odd
Even3D2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even3D2'); %Even
MultP3D2 = MultLayerP('MultP3D2');
MultU3D2 = MultLayerU('MultU3D2');
Add3D2 = additionLayer(2,'Name','Add3D2');
Sub3D2 = additionLayer(2,'Name','Sub3D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
MultNA1D1 = MultHaarNormAprox('MultNA1D1');
MultND1D1 = MultHaarNormDetall('MultND1D1');
MultNA2D1 = MultHaarNormAprox('MultNA2D1');
MultND2D1 = MultHaarNormDetall('MultND2D1');
MultNA3D1 = MultHaarNormAprox('MultNA3D1');
MultND3D1 = MultHaarNormDetall('MultND3D1');
MultNA1D2 = MultHaarNormAprox('MultNA1D2');
MultND1D2 = MultHaarNormDetall('MultND1D2');
MultNA2D2 = MultHaarNormAprox('MultNA2D2');
MultND2D2 = MultHaarNormDetall('MultND2D2');
MultNA3D2 = MultHaarNormAprox('MultNA3D2');
MultND3D2 = MultHaarNormDetall('MultND3D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Cat2 = concatenationLayer(3,4,'Name','Cat2');
Batch2 = batchNormalizationLayer('Name','Batch2');
Conv3 = convolution2dLayer([4,4],500,'Name','Conv3');
Relu1 = reluLayer('Name','Relu1');
Conv4 = convolution2dLayer([1,1],10,'Name','Conv4');
Batch3 = batchNormalizationLayer('Name','Batch3');
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
netM0 = addLayers(netM0,Cat1);
netM0 = addLayers(netM0,ShiftH3D1);
netM0 = addLayers(netM0,Odd3D1);
netM0 = addLayers(netM0,Even3D1);
netM0 = addLayers(netM0,MultP3D1);
netM0 = addLayers(netM0,MultU3D1);
netM0 = addLayers(netM0,Add3D1);
netM0 = addLayers(netM0,Sub3D1);
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
netM0 = addLayers(netM0,MultNA1D1);
netM0 = addLayers(netM0,MultND1D1);
netM0 = addLayers(netM0,MultNA2D1);
netM0 = addLayers(netM0,MultND2D1);
netM0 = addLayers(netM0,MultNA3D1);
netM0 = addLayers(netM0,MultND3D1);
netM0 = addLayers(netM0,MultNA1D2);
netM0 = addLayers(netM0,MultND1D2);
netM0 = addLayers(netM0,MultNA2D2);
netM0 = addLayers(netM0,MultND2D2);
netM0 = addLayers(netM0,MultNA3D2);
netM0 = addLayers(netM0,MultND3D2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Cat2);
netM0 = addLayers(netM0,Relu1);
netM0 = addLayers(netM0,Soft);
netM0 = addLayers(netM0,Out);
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'In','Conv1');
netM0 = connectLayers(netM0,'Conv1','Batch1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Batch1','Odd1D1');
netM0 = connectLayers(netM0,'Batch1','ShiftV1D1');
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
netM0 = connectLayers(netM0,'MultNA1D1','ShiftH2D1');
netM0 = connectLayers(netM0,'ShiftH2D1','Odd2D1');
netM0 = connectLayers(netM0,'ShiftH2D1','Even2D1');

netM0 = connectLayers(netM0,'MultND1D1','ShiftH3D1');
netM0 = connectLayers(netM0,'ShiftH3D1','Odd3D1');
netM0 = connectLayers(netM0,'ShiftH3D1','Even3D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%

netM0 = connectLayers(netM0,'Even2D1','MultP2D1');
netM0 = connectLayers(netM0,'Even2D1','Add2D1/in1');
netM0 = connectLayers(netM0,'Odd2D1','Sub2D1/in1');
netM0 = connectLayers(netM0,'Sub2D1','MultU2D1');
netM0 = connectLayers(netM0,'MultU2D1','Add2D1/in2');
netM0 = connectLayers(netM0,'MultP2D1','Sub2D1/in2');
netM0 = connectLayers(netM0,'Add2D1','MultNA2D1');
netM0 = connectLayers(netM0,'Sub2D1','MultND2D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Even3D1','MultP3D1');
netM0 = connectLayers(netM0,'Even3D1','Add3D1/in1');
netM0 = connectLayers(netM0,'Odd3D1','Sub3D1/in1');
netM0 = connectLayers(netM0,'Sub3D1','MultU3D1');
netM0 = connectLayers(netM0,'MultU3D1','Add3D1/in2');
netM0 = connectLayers(netM0,'MultP3D1','Sub3D1/in2');
netM0 = connectLayers(netM0,'Add3D1','MultNA3D1');
netM0 = connectLayers(netM0,'Sub3D1','MultND3D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'MultNA2D1','Cat1/in1');
netM0 = connectLayers(netM0,'MultND2D1','Cat1/in2');
netM0 = connectLayers(netM0,'MultNA3D1','Cat1/in3');
netM0 = connectLayers(netM0,'MultND3D1','Cat1/in4');
netM0 = connectLayers(netM0,'Cat1','Conv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Conv2','Odd1D2');
netM0 = connectLayers(netM0,'Conv2','ShiftV1D2');
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
netM0 = connectLayers(netM0,'MultNA1D2','ShiftH2D2');
netM0 = connectLayers(netM0,'MultND1D2','ShiftH3D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'ShiftH2D2','Even2D2');
netM0 = connectLayers(netM0,'ShiftH2D2','Odd2D2');

netM0 = connectLayers(netM0,'Even2D2','MultP2D2');
netM0 = connectLayers(netM0,'Even2D2','Add2D2/in1');
netM0 = connectLayers(netM0,'Odd2D2','Sub2D2/in1');
netM0 = connectLayers(netM0,'Sub2D2','MultU2D2');
netM0 = connectLayers(netM0,'MultU2D2','Add2D2/in2');
netM0 = connectLayers(netM0,'MultP2D2','Sub2D2/in2');
netM0 = connectLayers(netM0,'Add2D2','MultNA2D2');
netM0 = connectLayers(netM0,'Sub2D2','MultND2D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'ShiftH3D2','Even3D2');
netM0 = connectLayers(netM0,'ShiftH3D2','Odd3D2');
netM0 = connectLayers(netM0,'Even3D2','MultP3D2');
netM0 = connectLayers(netM0,'Even3D2','Add3D2/in1');
netM0 = connectLayers(netM0,'Odd3D2','Sub3D2/in1');
netM0 = connectLayers(netM0,'Sub3D2','MultU3D2');
netM0 = connectLayers(netM0,'MultU3D2','Add3D2/in2');
netM0 = connectLayers(netM0,'MultP3D2','Sub3D2/in2');
netM0 = connectLayers(netM0,'Add3D2','MultNA3D2');
netM0 = connectLayers(netM0,'Sub3D2','MultND3D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'MultNA2D2','Cat2/in1');
netM0 = connectLayers(netM0,'MultND2D2','Cat2/in2');
netM0 = connectLayers(netM0,'MultNA3D2','Cat2/in3');
netM0 = connectLayers(netM0,'MultND3D2','Cat2/in4');
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
figure()
plot(netM0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%
