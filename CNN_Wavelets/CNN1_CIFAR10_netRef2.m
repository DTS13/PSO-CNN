%%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%%%%%
%%%%%%%%%%%  CNN CIFAR10 Net Ref2  %%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
In = imageInputLayer([32 32 3],'Name','In');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Batch1 = batchNormalizationLayer('Name','Batch1');
Conv1 = convolution2dLayer([5,5],32,'Padding',[2 2 2 2],'Name','Conv1');
Batch2 = batchNormalizationLayer('Name','Batch2');
Relu1 = reluLayer('Name','Relu1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PoolD1 = maxPooling2dLayer([2 2],'Stride',[2 2],'Name','PoolD1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Conv2 = convolution2dLayer([5,5],32,'Padding',[2 2 2 2],'Name','Conv2');
Batch3 = batchNormalizationLayer('Name','Batch3');
Relu2 = reluLayer('Name','Relu2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PoolD2 = maxPooling2dLayer([2 2],'Stride',[2 2],'Name','PoolD2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Conv3 = convolution2dLayer([5,5],32,'Padding',[2 2 2 2],'Name','Conv3');
Batch4 = batchNormalizationLayer('Name','Batch4');
Relu3 = reluLayer('Name','Relu3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PoolD3 = maxPooling2dLayer([2 2],'Stride',[2 2],'Name','PoolD3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Drop1 = dropoutLayer(0.1,'Name','Drop1');
Conv4 = convolution2dLayer([4,4],64,'Name','Conv4');
Relu4 = reluLayer('Name','Relu4');
Drop2 = dropoutLayer(0.2,'Name','Drop2');
Conv5 = convolution2dLayer([1,1],10,'Name','Conv5');
FC1 = fullyConnectedLayer(10,'Name','FC1');
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
netM0 = addLayers(netM0,PoolD1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Conv2);
netM0 = addLayers(netM0,Batch3);
netM0 = addLayers(netM0,Relu2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,PoolD2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Conv3);
netM0 = addLayers(netM0,Batch4);
netM0 = addLayers(netM0,Relu3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,PoolD3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Drop1);
netM0 = addLayers(netM0,Conv4);
netM0 = addLayers(netM0,Relu4);
netM0 = addLayers(netM0,Drop2);
netM0 = addLayers(netM0,Conv5);
netM0 = addLayers(netM0,FC1);
netM0 = addLayers(netM0,Soft);
netM0 = addLayers(netM0,Out);
%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'In','Batch1');
netM0 = connectLayers(netM0,'Batch1','Conv1');
netM0 = connectLayers(netM0,'Conv1','Batch2');
netM0 = connectLayers(netM0,'Batch2','Relu1');
netM0 = connectLayers(netM0,'Relu1','PoolD1');
netM0 = connectLayers(netM0,'PoolD1','Conv2');
netM0 = connectLayers(netM0,'Conv2','Batch3');
netM0 = connectLayers(netM0,'Batch3','Relu2');
netM0 = connectLayers(netM0,'Relu2','PoolD2');
netM0 = connectLayers(netM0,'PoolD2','Conv3');
netM0 = connectLayers(netM0,'Conv3','Batch4');
netM0 = connectLayers(netM0,'Batch4','Relu3');
netM0 = connectLayers(netM0,'Relu3','PoolD3');
netM0 = connectLayers(netM0,'PoolD3','Drop1');
netM0 = connectLayers(netM0,'Drop1','Conv4');
netM0 = connectLayers(netM0,'Conv4','Relu4');
netM0 = connectLayers(netM0,'Relu4','Drop2');
netM0 = connectLayers(netM0,'Drop2','Conv5');
netM0 = connectLayers(netM0,'Conv5','FC1');
netM0 = connectLayers(netM0,'FC1','Soft');
netM0 = connectLayers(netM0,'Soft','Out');
%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Cat1 = concatenationLayer(3,3,'Name','Cat1');
%netM0 = addLayers(netM0,Cat1);
%netM0 = connectLayers(netM0,'Cat1','Conv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%AddCD1 = additionLayer(3,'Name','AddCD1')
%netM0 = addLayers(netM0,AddCD1);
%netM0 = connectLayers(netM0,'AddCD1','Cat1/in2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PoolD1 = maxPooling2dLayer([2 2],'Stride',[2 2],'Name','PoolD1');
%netM0 = addLayers(netM0,PoolD1);
%netM0 = connectLayers(netM0,'Batch2','PoolD1');
%netM0 = connectLayers(netM0,'PoolD1','Cat1/in1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Cat2 = concatenationLayer(3,3,'Name','Cat2');
%netM0 = addLayers(netM0,Cat2);
%netM0 = connectLayers(netM0,'Cat2','Batch3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%AddCD2 = additionLayer(3,'Name','AddCD2')
%netM0 = addLayers(netM0,AddCD2);
%netM0 = connectLayers(netM0,'AddCD2','Cat2/in2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PoolD2 = maxPooling2dLayer([2 2],'Stride',[2 2],'Name','PoolD2');
%netM0 = addLayers(netM0,PoolD2);
%netM0 = connectLayers(netM0,'Conv2','PoolD2');
%netM0 = connectLayers(netM0,'PoolD2','Cat2/in1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ShiftV1D1 = ShiftLayerV('ShiftV1D1'); 
%Odd1D1 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd1D1'); %Odd
%Even1D1 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even1D1'); %Even
%MultP1D1 = MultLayerP('MultP1D1');
%MultU1D1 = MultLayerU('MultU1D1');
%Add1D1 = additionLayer(2,'Name','Add1D1');
%Sub1D1 = additionLayer(2,'Name','Sub1D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ShiftH2D1 = ShiftLayerH('ShiftH2D1'); 
%Odd2D1 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Odd2D1'); %Odd
%Even2D1 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Even2D1'); %Even
%MultP2D1 = MultLayerP('MultP2D1');
%MultU2D1 = MultLayerU('MultU2D1');
%Add2D1 = additionLayer(2,'Name','Add2D1');
%Sub2D1 = additionLayer(2,'Name','Sub2D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ShiftH3D1 = ShiftLayerH('ShiftH3D1'); 
%Odd3D1 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Odd3D1'); %Odd
%Even3D1 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Even3D1'); %Even
%MultP3D1 = MultLayerP('MultP3D1');
%MultU3D1 = MultLayerU('MultU3D1');
%Add3D1 = additionLayer(2,'Name','Add3D1');
%Sub3D1 = additionLayer(2,'Name','Sub3D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%LLD1 = batchNormalizationLayer('Name','LLD1');
%HLD1 = batchNormalizationLayer('Name','HLD1');
%LHD1 = batchNormalizationLayer('Name','LHD1');
%HHD1 = batchNormalizationLayer('Name','HHD1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = addLayers(netM0,ShiftV1D1);
%netM0 = addLayers(netM0,Odd1D1);
%netM0 = addLayers(netM0,Even1D1);
%netM0 = addLayers(netM0,MultP1D1);
%netM0 = addLayers(netM0,MultU1D1);
%netM0 = addLayers(netM0,Add1D1);
%netM0 = addLayers(netM0,Sub1D1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = addLayers(netM0,ShiftH2D1);
%netM0 = addLayers(netM0,Odd2D1);
%netM0 = addLayers(netM0,Even2D1);
%netM0 = addLayers(netM0,MultP2D1);
%netM0 = addLayers(netM0,MultU2D1);
%netM0 = addLayers(netM0,Add2D1);
%netM0 = addLayers(netM0,Sub2D1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = addLayers(netM0,ShiftH3D1);
%netM0 = addLayers(netM0,Odd3D1);
%netM0 = addLayers(netM0,Even3D1);
%netM0 = addLayers(netM0,MultP3D1);
%netM0 = addLayers(netM0,MultU3D1);
%netM0 = addLayers(netM0,Add3D1);
%netM0 = addLayers(netM0,Sub3D1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = addLayers(netM0,LLD1);
%netM0 = addLayers(netM0,HLD1);
%netM0 = addLayers(netM0,LHD1);
%netM0 = addLayers(netM0,HHD1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = connectLayers(netM0,'Batch2','Odd1D1');             % Input
%netM0 = connectLayers(netM0,'Batch2','ShiftV1D1');          % Input
%netM0 = connectLayers(netM0,'ShiftV1D1','Even1D1');
%netM0 = connectLayers(netM0,'Even1D1','MultP1D1');
%netM0 = connectLayers(netM0,'Even1D1','Add1D1/in1');
%netM0 = connectLayers(netM0,'Odd1D1','Sub1D1/in1');
%netM0 = connectLayers(netM0,'Sub1D1','MultU1D1');
%netM0 = connectLayers(netM0,'MultU1D1','Add1D1/in2');
%netM0 = connectLayers(netM0,'MultP1D1','Sub1D1/in2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = connectLayers(netM0,'Add1D1','Odd2D1');
%netM0 = connectLayers(netM0,'Add1D1','ShiftH2D1');
%netM0 = connectLayers(netM0,'ShiftH2D1','Even2D1');
%netM0 = connectLayers(netM0,'Even2D1','MultP2D1');
%netM0 = connectLayers(netM0,'Even2D1','Add2D1/in1');
%netM0 = connectLayers(netM0,'Odd2D1','Sub2D1/in1');
%netM0 = connectLayers(netM0,'Sub2D1','MultU2D1');
%netM0 = connectLayers(netM0,'MultU2D1','Add2D1/in2');
%netM0 = connectLayers(netM0,'MultP2D1','Sub2D1/in2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = connectLayers(netM0,'Sub1D1','Odd3D1');
%netM0 = connectLayers(netM0,'Sub1D1','ShiftH3D1');
%netM0 = connectLayers(netM0,'ShiftH3D1','Even3D1');
%netM0 = connectLayers(netM0,'Even3D1','MultP3D1');
%netM0 = connectLayers(netM0,'Even3D1','Add3D1/in1');
%netM0 = connectLayers(netM0,'Odd3D1','Sub3D1/in1');
%netM0 = connectLayers(netM0,'Sub3D1','MultU3D1');
%netM0 = connectLayers(netM0,'MultU3D1','Add3D1/in2');
%netM0 = connectLayers(netM0,'MultP3D1','Sub3D1/in2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = connectLayers(netM0,'Add2D1','LLD1');
%netM0 = connectLayers(netM0,'Add3D1','HLD1');
%netM0 = connectLayers(netM0,'Sub2D1','LHD1');
%netM0 = connectLayers(netM0,'Sub3D1','HHD1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = connectLayers(netM0,'LLD1','Cat1/in3'); % LL Aprox.
%netM0 = connectLayers(netM0,'HLD1','AddCD1/in1'); % HL Horiz.
%netM0 = connectLayers(netM0,'LHD1','AddCD1/in2'); % LH Vert.
%netM0 = connectLayers(netM0,'HHD1','AddCD1/in3'); % HH Diag.
%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ShiftV1D2 = ShiftLayerV('ShiftV1D2'); 
%Odd1D2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Odd1D2'); %Odd
%Even1D2 = maxPooling2dLayer([1 1],'Stride',[2 1],'Name','Even1D2'); %Even
%MultP1D2 = MultLayerP('MultP1D2');
%MultU1D2 = MultLayerU('MultU1D2');
%Add1D2 = additionLayer(2,'Name','Add1D2');
%Sub1D2 = additionLayer(2,'Name','Sub1D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ShiftH2D2 = ShiftLayerH('ShiftH2D2'); 
%Odd2D2 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Odd2D2'); %Odd
%Even2D2 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Even2D2'); %Even
%MultP2D2 = MultLayerP('MultP2D2');
%MultU2D2 = MultLayerU('MultU2D2');
%Add2D2 = additionLayer(2,'Name','Add2D2');
%Sub2D2 = additionLayer(2,'Name','Sub2D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ShiftH3D2 = ShiftLayerH('ShiftH3D2'); 
%Odd3D2 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Odd3D2'); %Odd
%Even3D2 = maxPooling2dLayer([1 1],'Stride',[1 2],'Name','Even3D2'); %Even
%MultP3D2 = MultLayerP('MultP3D2');
%MultU3D2 = MultLayerU('MultU3D2');
%Add3D2 = additionLayer(2,'Name','Add3D2');
%Sub3D2 = additionLayer(2,'Name','Sub3D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%LLD2 = batchNormalizationLayer('Name','LLD2');
%HLD2 = batchNormalizationLayer('Name','HLD2');
%LHD2 = batchNormalizationLayer('Name','LHD2');
%HHD2 = batchNormalizationLayer('Name','HHD2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = addLayers(netM0,ShiftV1D2);
%netM0 = addLayers(netM0,Odd1D2);
%netM0 = addLayers(netM0,Even1D2);
%netM0 = addLayers(netM0,MultP1D2);
%netM0 = addLayers(netM0,MultU1D2);
%netM0 = addLayers(netM0,Add1D2);
%netM0 = addLayers(netM0,Sub1D2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = addLayers(netM0,ShiftH2D2);
%netM0 = addLayers(netM0,Odd2D2);
%netM0 = addLayers(netM0,Even2D2);
%netM0 = addLayers(netM0,MultP2D2);
%netM0 = addLayers(netM0,MultU2D2);
%netM0 = addLayers(netM0,Add2D2);
%netM0 = addLayers(netM0,Sub2D2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = addLayers(netM0,ShiftH3D2);
%netM0 = addLayers(netM0,Odd3D2);
%netM0 = addLayers(netM0,Even3D2);
%netM0 = addLayers(netM0,MultP3D2);
%netM0 = addLayers(netM0,MultU3D2);
%netM0 = addLayers(netM0,Add3D2);
%netM0 = addLayers(netM0,Sub3D2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = addLayers(netM0,LLD2);
%netM0 = addLayers(netM0,HLD2);
%netM0 = addLayers(netM0,LHD2);
%netM0 = addLayers(netM0,HHD2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = connectLayers(netM0,'Conv2','Odd1D2');             % Input
%netM0 = connectLayers(netM0,'Conv2','ShiftV1D2');          % Input
%netM0 = connectLayers(netM0,'ShiftV1D2','Even1D2');
%netM0 = connectLayers(netM0,'Even1D2','MultP1D2');
%netM0 = connectLayers(netM0,'Even1D2','Add1D2/in1');
%netM0 = connectLayers(netM0,'Odd1D2','Sub1D2/in1');
%netM0 = connectLayers(netM0,'Sub1D2','MultU1D2');
%netM0 = connectLayers(netM0,'MultU1D2','Add1D2/in2');
%netM0 = connectLayers(netM0,'MultP1D2','Sub1D2/in2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = connectLayers(netM0,'Add1D2','Odd2D2');
%netM0 = connectLayers(netM0,'Add1D2','ShiftH2D2');
%netM0 = connectLayers(netM0,'ShiftH2D2','Even2D2');
%netM0 = connectLayers(netM0,'Even2D2','MultP2D2');
%netM0 = connectLayers(netM0,'Even2D2','Add2D2/in1');
%netM0 = connectLayers(netM0,'Odd2D2','Sub2D2/in1');
%netM0 = connectLayers(netM0,'Sub2D2','MultU2D2');
%netM0 = connectLayers(netM0,'MultU2D2','Add2D2/in2');
%netM0 = connectLayers(netM0,'MultP2D2','Sub2D2/in2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = connectLayers(netM0,'Sub1D2','Odd3D2');
%netM0 = connectLayers(netM0,'Sub1D2','ShiftH3D2');
%netM0 = connectLayers(netM0,'ShiftH3D2','Even3D2');
%netM0 = connectLayers(netM0,'Even3D2','MultP3D2');
%netM0 = connectLayers(netM0,'Even3D2','Add3D2/in1');
%netM0 = connectLayers(netM0,'Odd3D2','Sub3D2/in1');
%netM0 = connectLayers(netM0,'Sub3D2','MultU3D2');
%netM0 = connectLayers(netM0,'MultU3D2','Add3D2/in2');
%netM0 = connectLayers(netM0,'MultP3D2','Sub3D2/in2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = connectLayers(netM0,'Add2D2','LLD2');
%netM0 = connectLayers(netM0,'Add3D2','HLD2');
%netM0 = connectLayers(netM0,'Sub2D2','LHD2');
%netM0 = connectLayers(netM0,'Sub3D2','HHD2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%netM0 = connectLayers(netM0,'LLD2','Cat2/in3'); % LL Aprox.
%netM0 = connectLayers(netM0,'HLD2','AddCD2/in1'); % HL Horiz.
%netM0 = connectLayers(netM0,'LHD2','AddCD2/in2'); % LH Vert.
%netM0 = connectLayers(netM0,'HHD2','AddCD2/in3'); % HH Diag.
%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0.Layers
figure()
plot(netM0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%