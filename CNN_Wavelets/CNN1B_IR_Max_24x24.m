%%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%
%%%%%%%%%%  CNN IR Maxpool %%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lgraphM0(1) = imageInputLayer([24 24 1],'Name','ImageInputLayer');
lgraphM0(2) = batchNormalizationLayer('Name','Batch1');
lgraphM0(3) = convolution2dLayer([2,2],36,'Name','Conv1','Padding','same');
lgraphM0(4) = convolution2dLayer([2,2],36,'Name','Conv2','Padding','same');
lgraphM0(5) = convolution2dLayer([2,2],106,'Name','Conv3','Padding','same');
lgraphM0(6) = convolution2dLayer([5,5],106,'Name','Conv4','Padding','same');
lgraphM0(7) = reluLayer('Name','Relu1');
lgraphM0(8) = fullyConnectedLayer(120,'Name','FC1');
lgraphM0(9) = fullyConnectedLayer(80,'Name','FC2'); 
lgraphM0(10) = fullyConnectedLayer(4,'Name','FC3'); %Number Classes
lgraphM0(11) = softmaxLayer('Name','Soft');
lgraphM0(12) = classificationLayer('Name','output');
netM0 = layerGraph(lgraphM0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

netM0.Layers
figure(1)
plot(netM0)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

