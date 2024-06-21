%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%  CNN MNIST Pool Channel Mix Net C %%%%%%%%

%clear all;
%close all;
%clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
In = imageInputLayer([28 28 1],'Name','In');
Conv1 = convolution2dLayer([5,5],20,'Name','Conv1');
Batch2 = batchNormalizationLayer('Name','Batch2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Pool11 = maxPooling2dLayer([2,2],'Stride',[2 2],'Name','Pool11');
Pool12 = averagePooling2dLayer([2,2],'Stride',[2 2],'Name','Pool12');
Pool13 = mixpoolChLayer('Pool13');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Conv2 = convolution2dLayer([5,5],50,'Name','Conv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Pool21 = maxPooling2dLayer([2,2],'Stride',[2 2],'Name','Pool21');
Pool22 = averagePooling2dLayer([2,2],'Stride',[2 2],'Name','Pool22');
Pool23 = mixpoolChLayer('Pool23');
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
netM0 = addLayers(netM0,Pool11);
netM0 = addLayers(netM0,Pool12);
netM0 = addLayers(netM0,Pool13);
netM0 = addLayers(netM0,Pool21);
netM0 = addLayers(netM0,Pool22);
netM0 = addLayers(netM0,Pool23);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Relu);
netM0 = addLayers(netM0,Soft);
netM0 = addLayers(netM0,Out);
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'In','Conv1');
netM0 = connectLayers(netM0,'Conv1','Batch2');
netM0 = connectLayers(netM0,'Batch2','Pool11');
netM0 = connectLayers(netM0,'Batch2','Pool12');
netM0 = connectLayers(netM0,'Pool11','Pool13/in1');
netM0 = connectLayers(netM0,'Pool12','Pool13/in2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Pool13','Conv2');
netM0 = connectLayers(netM0,'Conv2','Pool21');
netM0 = connectLayers(netM0,'Conv2','Pool22');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Pool21','Pool23/in1');
netM0 = connectLayers(netM0,'Pool22','Pool23/in2');
netM0 = connectLayers(netM0,'Pool23','Batch3');
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
