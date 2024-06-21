%%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%%
%%%%%%%%%%%%  CNN MNIST 5b  %%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
In = imageInputLayer([28 28 1],'Name','In');
Batch1 = batchNormalizationLayer('Name','Batch1');
Conv1 = convolution2dLayer([5,5],20,'Name','Conv1');
Batch2 = batchNormalizationLayer('Name','Batch2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Pool1a = averagePooling2dLayer([2,2],'stride',[2,2],'Name','Pool1a');
Pool1m = maxPooling2dLayer([2,2],'stride',[2,2],'Name','Pool1m');
Add1D = additionLayer(2,'Name','Add1D');
Relu1D = leakyReluLayer('Name','Relu1D');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Conv2 = convolution2dLayer([5,5],50,'Name','Conv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Pool2a = averagePooling2dLayer([2,2],'stride',[2,2],'Name','Pool2a');
Pool2m = maxPooling2dLayer([2,2],'stride',[2,2],'Name','Pool2m');
Add2D = additionLayer(2,'Name','Add2D');
Relu2D = leakyReluLayer('Name','Relu2D');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Pool1a);
netM0 = addLayers(netM0,Pool1m);
netM0 = addLayers(netM0,Add1D);
netM0 = addLayers(netM0,Relu1D);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Pool2a);
netM0 = addLayers(netM0,Pool2m);
netM0 = addLayers(netM0,Add2D);
netM0 = addLayers(netM0,Relu2D);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Relu1);
netM0 = addLayers(netM0,Soft);
netM0 = addLayers(netM0,FC1);
netM0 = addLayers(netM0,Out);
%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'In','Batch1');
netM0 = connectLayers(netM0,'Batch1','Conv1');
netM0 = connectLayers(netM0,'Conv1','Batch2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Batch2','Pool1a');
netM0 = connectLayers(netM0,'Batch2','Pool1m');
netM0 = connectLayers(netM0,'Pool1a','Add1D/in1');
netM0 = connectLayers(netM0,'Pool1m','Add1D/in2');
netM0 = connectLayers(netM0,'Add1D','Relu1D');
netM0 = connectLayers(netM0,'Relu1D','Conv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Conv2','Pool2a');
netM0 = connectLayers(netM0,'Conv2','Pool2m');
netM0 = connectLayers(netM0,'Pool2a','Add2D/in1');
netM0 = connectLayers(netM0,'Pool2m','Add2D/in2');
netM0 = connectLayers(netM0,'Add2D','Relu2D');
netM0 = connectLayers(netM0,'Relu2D','Batch3');
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
