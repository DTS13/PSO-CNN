%%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%%%%%
%%%%%%%%%%%  IR by CNN KDEF %%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
In = imageInputLayer([128 128 1],'Name','In');
Batch1 = batchNormalizationLayer('Name','Batch1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Conv1 = convolution2dLayer([5,5],32,'Padding',[2 2 2 2],'Name','Conv1');
Batch2 = batchNormalizationLayer('Name','Batch2');
Relu1 = reluLayer('Name','Relu1');
PoolD1 = maxPooling2dLayer([2 2],'Stride',[2 2],'Name','PoolD1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Conv2 = convolution2dLayer([5,5],32,'Padding',[0 0 0 0],'Name','Conv2');
Batch3 = batchNormalizationLayer('Name','Batch3');
Relu2 = reluLayer('Name','Relu2');
PoolD2 = maxPooling2dLayer([2 2],'Stride',[2 2],'Name','PoolD2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Conv3 = convolution2dLayer([5,5],32,'Padding',[0 0 0 0],'Name','Conv3');
Batch4 = batchNormalizationLayer('Name','Batch4');
Relu3 = reluLayer('Name','Relu3');
PoolD3 = maxPooling2dLayer([2 2],'Stride',[2 2],'Name','PoolD3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Drop1 = dropoutLayer(0.4,'Name','Drop1');
Conv4 = convolution2dLayer([4,4],64,'Padding',[0 0 0 0],'Name','Conv4');
Batch5 = batchNormalizationLayer('Name','Batch5');
Relu4 = reluLayer('Name','Relu4');
PoolD4 = maxPooling2dLayer([2 2],'Stride',[2 2],'Name','PoolD4');
Drop2 = dropoutLayer(0.2,'Name','Drop2');
Conv5 = convolution2dLayer([5,5],4,'Padding',[0 0 0 0],'Name','Conv5'); % Number of Classes: 4
%FC1 = fullyConnectedLayer(4,'Name','FC1');
Soft = softmaxLayer('Name','Soft');
Out = classificationLayer('Name','Out');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%}


%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = layerGraph;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,In);
netM0 = addLayers(netM0,Batch1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Conv1);
netM0 = addLayers(netM0,Batch2);
netM0 = addLayers(netM0,Relu1);
netM0 = addLayers(netM0,PoolD1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Conv2);
netM0 = addLayers(netM0,Batch3);
netM0 = addLayers(netM0,Relu2);
netM0 = addLayers(netM0,PoolD2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Conv3);
netM0 = addLayers(netM0,Batch4);
netM0 = addLayers(netM0,Relu3);
netM0 = addLayers(netM0,PoolD3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Drop1);
netM0 = addLayers(netM0,Conv4);
netM0 = addLayers(netM0,Batch5);
netM0 = addLayers(netM0,Relu4);
netM0 = addLayers(netM0,PoolD4);
netM0 = addLayers(netM0,Drop2);
netM0 = addLayers(netM0,Conv5);
%netM0 = addLayers(netM0,FC1);
netM0 = addLayers(netM0,Soft);
netM0 = addLayers(netM0,Out);
%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'In','Batch1');
netM0 = connectLayers(netM0,'Batch1','Conv1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Conv1','Batch2');
netM0 = connectLayers(netM0,'Batch2','Relu1');
netM0 = connectLayers(netM0,'Relu1','PoolD1');
netM0 = connectLayers(netM0,'PoolD1','Conv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Conv2','Batch3');
netM0 = connectLayers(netM0,'Batch3','Relu2');
netM0 = connectLayers(netM0,'Relu2','PoolD2');
netM0 = connectLayers(netM0,'PoolD2','Conv3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Conv3','Batch4');
netM0 = connectLayers(netM0,'Batch4','Relu3');
netM0 = connectLayers(netM0,'Relu3','PoolD3');
netM0 = connectLayers(netM0,'PoolD3','Drop1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Drop1','Conv4');
netM0 = connectLayers(netM0,'Conv4','Batch5');
netM0 = connectLayers(netM0,'Batch5','Relu4');
netM0 = connectLayers(netM0,'Relu4','PoolD4');
netM0 = connectLayers(netM0,'PoolD4','Drop2');
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