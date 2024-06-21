%%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%
%%%%%%%%%%  CNN MNIST Net Ref 6 %%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lgraphM0(1) = imageInputLayer([28 28 1],'Name','ImageInputLayer');
lgraphM0(2) = batchNormalizationLayer('Name','Batch1');
lgraphM0(3) = convolution2dLayer([5,5],20,'Name','Conv1');
lgraphM0(4) = batchNormalizationLayer('Name','Batch2');
lgraphM0(5) = maxPooling2dLayer([2,2],'Stride',[2,2],'Name','Pool1');
lgraphM0(6) = leakyReluLayer('Name','Relu1D');
lgraphM0(7) = convolution2dLayer([5,5],50,'Name','Conv2');
lgraphM0(8) = maxPooling2dLayer([2,2],'Stride',[2,2],'Name','Pool2');
lgraphM0(9) = leakyReluLayer('Name','Relu2D');
lgraphM0(10) = batchNormalizationLayer('Name','Batch3');
lgraphM0(11) = convolution2dLayer([4,4],500,'Name','Conv3');
lgraphM0(12) = reluLayer('Name','Relu1');
lgraphM0(13) = convolution2dLayer([1,1],10,'Name','Conv4');
lgraphM0(14) = batchNormalizationLayer('Name','Batch4');
lgraphM0(15) = fullyConnectedLayer(10,'Name','FC1');
lgraphM0(16) = softmaxLayer('Name','Soft');
lgraphM0(17) = classificationLayer('Name','output');
netM0 = layerGraph(lgraphM0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

netM0.Layers
figure(1)
plot(netM0)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

