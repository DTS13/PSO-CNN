%%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%%%%%
%%%%%%%%%%%%  CNN MNIST Ref 3  %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Mix Pooling %%%%%%%%%%%%%%%%%%

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
Pool1x = mixpoolLayer('Pool1x');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Conv2 = convolution2dLayer([5,5],50,'Name','Conv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Pool2a = averagePooling2dLayer([2,2],'stride',[2,2],'Name','Pool2a');
Pool2m = maxPooling2dLayer([2,2],'stride',[2,2],'Name','Pool2m');
Pool2x = mixpoolLayer('Pool2x');
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
netM0 = addLayers(netM0,Pool1x);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Pool2a);
netM0 = addLayers(netM0,Pool2m);
netM0 = addLayers(netM0,Pool2x);
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
netM0 = connectLayers(netM0,'Pool1a','Pool1x/in1');
netM0 = connectLayers(netM0,'Pool1m','Pool1x/in2');
netM0 = connectLayers(netM0,'Pool1x','Conv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Conv2','Pool2a');
netM0 = connectLayers(netM0,'Conv2','Pool2m');
netM0 = connectLayers(netM0,'Pool2a','Pool2x/in1');
netM0 = connectLayers(netM0,'Pool2m','Pool2x/in2');
netM0 = connectLayers(netM0,'Pool2x','Batch3');
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
