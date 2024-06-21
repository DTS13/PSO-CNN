%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%%%%
%%%%%%%%%%%%  CNN MNIST Pool Max Net D %%%%%

%clear all;
%close all;
%clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
In = imageInputLayer([28 28 1],'Name','In');
Conv1 = convolution2dLayer([5,5],20,'Name','Conv1');
Batch2 = batchNormalizationLayer('Name','Batch2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Pool1 = maxPooling2dLayer([2,2],'Stride',[2 2],'Name','Pool1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Conv2 = convolution2dLayer([5,5],50,'Name','Conv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Pool2 = maxPooling2dLayer([2,2],'Stride',[2 2],'Name','Pool2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Batch3 = batchNormalizationLayer('Name','Batch3');
Conv3 = convolution2dLayer([4,4],500,'Name','Conv3');
Relu = reluLayer('Name','Relu');
Conv4 = convolution2dLayer([1,1],10,'Name','Conv4');
Batch4 = batchNormalizationLayer('Name','Batch4');
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
netM0 = addLayers(netM0,Batch2);
netM0 = addLayers(netM0,Batch3);
netM0 = addLayers(netM0,Batch4);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Pool1);
netM0 = addLayers(netM0,Pool2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Relu);
netM0 = addLayers(netM0,Soft);
netM0 = addLayers(netM0,Out);
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'In','Conv1');
netM0 = connectLayers(netM0,'Conv1','Batch2');
netM0 = connectLayers(netM0,'Batch2','Pool1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Pool1','Conv2');
netM0 = connectLayers(netM0,'Conv2','Pool2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Pool2','Batch3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Batch3','Conv3');
netM0 = connectLayers(netM0,'Conv3','Relu');
netM0 = connectLayers(netM0,'Relu','Conv4');
netM0 = connectLayers(netM0,'Conv4','Batch4');
netM0 = connectLayers(netM0,'Batch4','Soft');
netM0 = connectLayers(netM0,'Soft','Out');
%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0.Layers
figure()
plot(netM0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%
