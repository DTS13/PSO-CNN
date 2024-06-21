%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%%%%%%%%%
%%%%%%%  CNN MNIST Pool Stochastic Net H %%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
In = imageInputLayer([28 28 1],'Name','In');
Conv1 = convolution2dLayer([5,5],20,'Name','Conv1');
Batch2 = batchNormalizationLayer('Name','Batch2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Separate11D1 = separate11Layer('Separate11D1');
Separate12D1 = separate12Layer('Separate12D1');
Separate21D1 = separate21Layer('Separate21D1');
Separate22D1 = separate22Layer('Separate22D1');
Pool11D1 = maxPooling2dLayer([2,2],'Stride',[2 2],'Name','Pool11D1');
Pool12D1 = maxPooling2dLayer([2,2],'Stride',[2 2],'Name','Pool12D1');
Pool21D1 = maxPooling2dLayer([2,2],'Stride',[2 2],'Name','Pool21D1');
Pool22D1 = maxPooling2dLayer([2,2],'Stride',[2 2],'Name','Pool22D1');
Pool1 = stochpoolLayer('Pool1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Conv2 = convolution2dLayer([5,5],50,'Name','Conv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Separate11D2 = separate11Layer('Separate11D2');
Separate12D2 = separate12Layer('Separate12D2');
Separate21D2 = separate21Layer('Separate21D2');
Separate22D2 = separate22Layer('Separate22D2');
Pool11D2 = maxPooling2dLayer([2,2],'Stride',[2 2],'Name','Pool11D2');
Pool12D2 = maxPooling2dLayer([2,2],'Stride',[2 2],'Name','Pool12D2');
Pool21D2 = maxPooling2dLayer([2,2],'Stride',[2 2],'Name','Pool21D2');
Pool22D2 = maxPooling2dLayer([2,2],'Stride',[2 2],'Name','Pool22D2');
Pool2 = stochpoolLayer('Pool2');
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
netM0 = addLayers(netM0,Separate11D1);
netM0 = addLayers(netM0,Separate12D1);
netM0 = addLayers(netM0,Separate21D1);
netM0 = addLayers(netM0,Separate22D1);
netM0 = addLayers(netM0,Pool11D1);
netM0 = addLayers(netM0,Pool12D1);
netM0 = addLayers(netM0,Pool21D1);
netM0 = addLayers(netM0,Pool22D1);
netM0 = addLayers(netM0,Pool1);
netM0 = addLayers(netM0,Separate11D2);
netM0 = addLayers(netM0,Separate12D2);
netM0 = addLayers(netM0,Separate21D2);
netM0 = addLayers(netM0,Separate22D2);
netM0 = addLayers(netM0,Pool11D2);
netM0 = addLayers(netM0,Pool12D2);
netM0 = addLayers(netM0,Pool21D2);
netM0 = addLayers(netM0,Pool22D2);
netM0 = addLayers(netM0,Pool2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Relu);
netM0 = addLayers(netM0,Soft);
netM0 = addLayers(netM0,Out);
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'In','Conv1');
netM0 = connectLayers(netM0,'Conv1','Batch2');
netM0 = connectLayers(netM0,'Batch2','Separate11D1');
netM0 = connectLayers(netM0,'Batch2','Separate12D1');
netM0 = connectLayers(netM0,'Batch2','Separate21D1');
netM0 = connectLayers(netM0,'Batch2','Separate22D1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Separate11D1','Pool11D1');
netM0 = connectLayers(netM0,'Separate12D1','Pool12D1');
netM0 = connectLayers(netM0,'Separate21D1','Pool21D1');
netM0 = connectLayers(netM0,'Separate22D1','Pool22D1');
netM0 = connectLayers(netM0,'Pool11D1','Pool1/in1');
netM0 = connectLayers(netM0,'Pool12D1','Pool1/in2');
netM0 = connectLayers(netM0,'Pool21D1','Pool1/in3');
netM0 = connectLayers(netM0,'Pool22D1','Pool1/in4');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Pool1','Conv2');
netM0 = connectLayers(netM0,'Conv2','Separate11D2');
netM0 = connectLayers(netM0,'Conv2','Separate12D2');
netM0 = connectLayers(netM0,'Conv2','Separate21D2');
netM0 = connectLayers(netM0,'Conv2','Separate22D2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Separate11D2','Pool11D2');
netM0 = connectLayers(netM0,'Separate12D2','Pool12D2');
netM0 = connectLayers(netM0,'Separate21D2','Pool21D2');
netM0 = connectLayers(netM0,'Separate22D2','Pool22D2');
netM0 = connectLayers(netM0,'Pool11D2','Pool2/in1');
netM0 = connectLayers(netM0,'Pool12D2','Pool2/in2');
netM0 = connectLayers(netM0,'Pool21D2','Pool2/in3');
netM0 = connectLayers(netM0,'Pool22D2','Pool2/in4');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Pool2','Batch3');
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
