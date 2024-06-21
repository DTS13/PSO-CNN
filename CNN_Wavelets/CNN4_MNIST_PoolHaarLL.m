%%%%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%  CNN MNIST Pool Harr LL Net A %%%%%%%%%%%%%
% con ReluD1 y ReluD2 => leakyReluLayers  99.1 %
% sin ReluD1 y ReluD2 => leakyReluLayers  98.4 %

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
In = imageInputLayer([28 28 1],'Name','In');
Conv1 = convolution2dLayer([5,5],20,'Name','Conv1');
Batch1 = batchNormalizationLayer('Name','Batch1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LayerOddH1 = LayerOddH('LayerOddH1');
LayerEvenH1 = LayerEvenH('LayerEvenH1');
OddH1 = maxPooling2dLayer([2 1],'Stride',[2 1],'Name','OddH1'); %Odd
EvenH1 = maxPooling2dLayer([2 1],'Stride',[2 1],'Name','EvenH1'); %Even
LayerPH1 = LayerP('LayerPH1');
LayerUH1 = LayerU('LayerUH1');
AddH1 = additionLayer(2,'Name','AddH1');
SubH1 = additionLayer(2,'Name','SubH1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
LayerOddV1 = LayerOddV('LayerOddV1');
LayerEvenV1 = LayerEvenV('LayerEvenV1');
OddV1 = maxPooling2dLayer([1 2],'Stride',[1 2],'Name','OddV1'); %Odd
EvenV1 = maxPooling2dLayer([1 2],'Stride',[1 2],'Name','EvenV1'); %Even
LayerPV1 = LayerP('LayerPV1');
LayerUV1 = LayerU('LayerUV1');
AddV1 = additionLayer(2,'Name','AddV1');
SubV1 = additionLayer(2,'Name','SubV1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ReluD1 = leakyReluLayer('Name','ReluD1');
Conv2 = convolution2dLayer([5,5],50,'Name','Conv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MultNAH1 = MultHaarNormAprox('MultNAH1');
MultNAV1 = MultHaarNormAprox('MultNAV1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ReluD2 = leakyReluLayer('Name','ReluD2');
Batch2 = batchNormalizationLayer('Name','Batch2');
Conv3 = convolution2dLayer([4,4],500,'Name','Conv3');
Relu1 = reluLayer('Name','Relu1');
Conv4 = convolution2dLayer([1,1],10,'Name','Conv4');
Batch3 = batchNormalizationLayer('Name','Batch3');
Batch4 = batchNormalizationLayer('Name','Batch4');
Batch8 = batchNormalizationLayer('Name','Batch8');
FC1 = fullyConnectedLayer(10,'Name','FC1')
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
netM0 = addLayers(netM0,Batch8);
netM0 = addLayers(netM0,ReluD1);
netM0 = addLayers(netM0,ReluD2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,LayerOddH1);
netM0 = addLayers(netM0,LayerEvenH1);
netM0 = addLayers(netM0,OddH1);
netM0 = addLayers(netM0,EvenH1);
netM0 = addLayers(netM0,LayerPH1);
netM0 = addLayers(netM0,LayerUH1);
netM0 = addLayers(netM0,AddH1);
netM0 = addLayers(netM0,SubH1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,LayerOddV1);
netM0 = addLayers(netM0,LayerEvenV1);
netM0 = addLayers(netM0,OddV1);
netM0 = addLayers(netM0,EvenV1);
netM0 = addLayers(netM0,LayerPV1);
netM0 = addLayers(netM0,LayerUV1);
netM0 = addLayers(netM0,AddV1);
netM0 = addLayers(netM0,SubV1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,MultNAH1);
netM0 = addLayers(netM0,MultNAV1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = addLayers(netM0,Relu1);
netM0 = addLayers(netM0,FC1);
netM0 = addLayers(netM0,Soft);
netM0 = addLayers(netM0,Out);
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'In','Conv1');
netM0 = connectLayers(netM0,'Conv1','Batch1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Batch1','LayerOddH1');
netM0 = connectLayers(netM0,'Batch1','LayerEvenH1');
netM0 = connectLayers(netM0,'LayerOddH1','OddH1');
netM0 = connectLayers(netM0,'LayerEvenH1','EvenH1');
netM0 = connectLayers(netM0,'EvenH1','LayerPH1');
netM0 = connectLayers(netM0,'EvenH1','AddH1/in1');
netM0 = connectLayers(netM0,'OddH1','SubH1/in1');
netM0 = connectLayers(netM0,'SubH1','LayerUH1');
netM0 = connectLayers(netM0,'LayerUH1','AddH1/in2');
netM0 = connectLayers(netM0,'LayerPH1','SubH1/in2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'AddH1','MultNAH1');
netM0 = connectLayers(netM0,'MultNAH1','Batch8');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Batch8','LayerOddV1');
netM0 = connectLayers(netM0,'Batch8','LayerEvenV1');
netM0 = connectLayers(netM0,'LayerOddV1','OddV1');
netM0 = connectLayers(netM0,'LayerEvenV1','EvenV1');
netM0 = connectLayers(netM0,'EvenV1','LayerPV1');
netM0 = connectLayers(netM0,'EvenV1','AddV1/in1');
netM0 = connectLayers(netM0,'OddV1','SubV1/in1');
netM0 = connectLayers(netM0,'SubV1','LayerUV1');
netM0 = connectLayers(netM0,'LayerUV1','AddV1/in2');
netM0 = connectLayers(netM0,'LayerPV1','SubV1/in2');
netM0 = connectLayers(netM0,'AddV1','MultNAV1');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'MultNAV1','Batch4');
netM0 = connectLayers(netM0,'Batch4','ReluD1');
netM0 = connectLayers(netM0,'ReluD1','Conv2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0 = connectLayers(netM0,'Conv2','ReluD2');
netM0 = connectLayers(netM0,'ReluD2','Batch2');
netM0 = connectLayers(netM0,'Batch2','Conv3');
netM0 = connectLayers(netM0,'Conv3','Relu1');
netM0 = connectLayers(netM0,'Relu1','Conv4');
netM0 = connectLayers(netM0,'Conv4','Batch3');
netM0 = connectLayers(netM0,'Batch3','FC1');
netM0 = connectLayers(netM0,'FC1','Soft');
netM0 = connectLayers(netM0,'Soft','Out');
%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%
netM0.Layers
figure()
plot(netM0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%
