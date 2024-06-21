%%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%%%%%
%%%%%%%%%%%  CNN MNIST MAX & AVG POOL  %%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
In = imageInputLayer([28 28 1],'Name','In');
max1 = maxPooling2dLayer([2 2],'Stride',[2 2],'Name','max1');
avg1 = averagePooling2dLayer([2 2],'Stride',[2 2],'Name','avg1');
cat1 = concatenationLayer(3,2,'Name','cat1');
%%%%%%%%%%%%%%%%%%%%%
Conv2 = convolution2dLayer([5,5],50,'Name','Conv2');
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
netM0 = addLayers(netM0,Conv2);
netM0 = addLayers(netM0,Conv3);
netM0 = addLayers(netM0,Conv4);
netM0 = addLayers(netM0,Batch3);
netM0 = addLayers(netM0,Batch4);
netM0 = addLayers(netM0,max1);
netM0 = addLayers(netM0,avg1);
netM0 = addLayers(netM0,cat1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Relu1);
netM0 = addLayers(netM0,Soft);
netM0 = addLayers(netM0,FC1);
netM0 = addLayers(netM0,Out);
%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'In','max1');
netM0 = connectLayers(netM0,'In','avg1');
netM0 = connectLayers(netM0,'max1','cat1/in1');
netM0 = connectLayers(netM0,'avg1','cat1/in2');
netM0 = connectLayers(netM0,'cat1','Conv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Conv2','Batch3');
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
