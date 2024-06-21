%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%
%%%%%%%%%%%%  CNN MNIST Net 2k  %%%%%%%%%%%%

clear all;
close all;
clc;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
In = imageInputLayer([28 28 1],'Name','In');
Batch1 = batchNormalizationLayer('Name','Batch1');
Conv1 = convolution2dLayer([5,5],20,'Name','Conv1');
Batch2 = batchNormalizationLayer('Name','Batch2');
Pool1 = averagePooling2dLayer([2,2],'Stride',[2 2],'Name','Pool1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftV1 = ShiftLayerV('ShiftV1'); 
Odd1 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Odd1'); %Odd
Even1 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Even1'); %Even
MultP1 = MultLayerP('MultP1');
MultU1 = MultLayerU('MultU1');
Add1 = additionLayer(2,'Name','Add1');
Sub1 = additionLayer(2,'Name','Sub1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH2 = ShiftLayerH('ShiftH2'); 
Odd2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd2'); %Odd
Even2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even2'); %Even
MultP2 = MultLayerP('MultP2');
MultU2 = MultLayerU('MultU2');
Add2 = additionLayer(2,'Name','Add2');
Sub2 = additionLayer(2,'Name','Sub2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ReluD11 = leakyReluLayer('Name','ReluD11');
ReluD12 = leakyReluLayer('Name','ReluD12');
AddD1 = additionLayer(2,'Name','AddD1');
Conv2 = convolution2dLayer([5,5],50,'Name','Conv2');
Pool2 = averagePooling2dLayer([2,2],'Stride',[2 2],'Name','Pool2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftV3 = ShiftLayerV('ShiftV3'); 
Odd3 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Odd3'); %Odd
Even3 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Even3'); %Even
MultP3 = MultLayerP('MultP3');
MultU3 = MultLayerU('MultU3');
Add3 = additionLayer(2,'Name','Add3');
Sub3 = additionLayer(2,'Name','Sub3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
ShiftH4 = ShiftLayerH('ShiftH4'); 
Odd4 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd4'); %Odd
Even4 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even4'); %Even
MultP4 = MultLayerP('MultP4');
MultU4 = MultLayerU('MultU4');
Add4 = additionLayer(2,'Name','Add4');
Sub4 = additionLayer(2,'Name','Sub4');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ReluD21 = leakyReluLayer('Name','ReluD21');
ReluD22 = leakyReluLayer('Name','ReluD22');
AddD2 = additionLayer(2,'Name','AddD2');
Batch3 = batchNormalizationLayer('Name','Batch3');
Conv3 = convolution2dLayer([4,4],500,'Name','Conv3');
Relu1 = reluLayer('Name','Relu1');
Conv4 = convolution2dLayer([1,1],10,'Name','Conv4');
Batch4 = batchNormalizationLayer('Name','Batch4');
FC1 = fullyConnectedLayer(10,'Name','FC1');
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
netM0 = addLayers(netM0,ReluD11);
netM0 = addLayers(netM0,ReluD12);
netM0 = addLayers(netM0,ReluD21);
netM0 = addLayers(netM0,ReluD22);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Pool1);
netM0 = addLayers(netM0,ShiftV1);
netM0 = addLayers(netM0,Odd1);
netM0 = addLayers(netM0,Even1);
netM0 = addLayers(netM0,MultP1);
netM0 = addLayers(netM0,MultU1);
netM0 = addLayers(netM0,Add1);
netM0 = addLayers(netM0,Sub1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftH2);
netM0 = addLayers(netM0,Odd2);
netM0 = addLayers(netM0,Even2);
netM0 = addLayers(netM0,MultP2);
netM0 = addLayers(netM0,MultU2);
netM0 = addLayers(netM0,Add2);
netM0 = addLayers(netM0,Sub2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,AddD1);
netM0 = addLayers(netM0,Pool2);
netM0 = addLayers(netM0,ShiftV3);
netM0 = addLayers(netM0,Odd3);
netM0 = addLayers(netM0,Even3);
netM0 = addLayers(netM0,MultP3);
netM0 = addLayers(netM0,MultU3);
netM0 = addLayers(netM0,Add3);
netM0 = addLayers(netM0,Sub3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,ShiftH4);
netM0 = addLayers(netM0,Odd4);
netM0 = addLayers(netM0,Even4);
netM0 = addLayers(netM0,MultP4);
netM0 = addLayers(netM0,MultU4);
netM0 = addLayers(netM0,Add4);
netM0 = addLayers(netM0,Sub4);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,AddD2);
netM0 = addLayers(netM0,Relu1);
netM0 = addLayers(netM0,Soft);
netM0 = addLayers(netM0,FC1);
netM0 = addLayers(netM0,Out);
%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'In','Batch1');
netM0 = connectLayers(netM0,'Batch1','Conv1');
netM0 = connectLayers(netM0,'Conv1','Batch2');
netM0 = connectLayers(netM0,'Batch2','Pool1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Batch2','Odd1');
netM0 = connectLayers(netM0,'Batch2','ShiftV1');
netM0 = connectLayers(netM0,'ShiftV1','Even1');
netM0 = connectLayers(netM0,'Even1','MultP1');
netM0 = connectLayers(netM0,'Even1','Add1/in1');
netM0 = connectLayers(netM0,'Odd1','Sub1/in1');
netM0 = connectLayers(netM0,'Sub1','MultU1');
netM0 = connectLayers(netM0,'MultU1','Add1/in2');
netM0 = connectLayers(netM0,'MultP1','Sub1/in2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Add1','Odd2');
netM0 = connectLayers(netM0,'Add1','ShiftH2');
netM0 = connectLayers(netM0,'ShiftH2','Even2');
netM0 = connectLayers(netM0,'Even2','MultP2');
netM0 = connectLayers(netM0,'Even2','Add2/in1');
netM0 = connectLayers(netM0,'Odd2','Sub2/in1');
netM0 = connectLayers(netM0,'Sub2','MultU2');
netM0 = connectLayers(netM0,'MultU2','Add2/in2');
netM0 = connectLayers(netM0,'MultP2','Sub2/in2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Pool1','ReluD11');
netM0 = connectLayers(netM0,'Add2','ReluD12');
netM0 = connectLayers(netM0,'ReluD11','AddD1/in1');
netM0 = connectLayers(netM0,'ReluD12','AddD1/in2');
netM0 = connectLayers(netM0,'AddD1','Conv2');
netM0 = connectLayers(netM0,'Conv2','Pool2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Conv2','Odd3');
netM0 = connectLayers(netM0,'Conv2','ShiftV3');
netM0 = connectLayers(netM0,'ShiftV3','Even3');
netM0 = connectLayers(netM0,'Even3','MultP3');
netM0 = connectLayers(netM0,'Even3','Add3/in1');
netM0 = connectLayers(netM0,'Odd3','Sub3/in1');
netM0 = connectLayers(netM0,'Sub3','MultU3');
netM0 = connectLayers(netM0,'MultU3','Add3/in2');
netM0 = connectLayers(netM0,'MultP3','Sub3/in2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Add3','Odd4');
netM0 = connectLayers(netM0,'Add3','ShiftH4');
netM0 = connectLayers(netM0,'ShiftH4','Even4');
netM0 = connectLayers(netM0,'Even4','MultP4');
netM0 = connectLayers(netM0,'Even4','Add4/in1');
netM0 = connectLayers(netM0,'Odd4','Sub4/in1');
netM0 = connectLayers(netM0,'Sub4','MultU4');
netM0 = connectLayers(netM0,'MultU4','Add4/in2');
netM0 = connectLayers(netM0,'MultP4','Sub4/in2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Pool2','ReluD21');
netM0 = connectLayers(netM0,'Add4','ReluD22');
netM0 = connectLayers(netM0,'ReluD21','AddD2/in1');
netM0 = connectLayers(netM0,'ReluD22','AddD2/in2');
netM0 = connectLayers(netM0,'AddD2','Batch3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Batch3','Conv3');
netM0 = connectLayers(netM0,'Conv3','Relu1');
netM0 = connectLayers(netM0,'Relu1','Conv4');
netM0 = connectLayers(netM0,'Conv4','Batch4');
netM0 = connectLayers(netM0,'Batch4','FC1');
netM0 = connectLayers(netM0,'FC1','Soft');
netM0 = connectLayers(netM0,'Soft','Out');
%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0.Layers
figure(1)
plot(netM0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%
