%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%
%%%%%%%%%%%%  CNN MNIST Ref  %%%%%%%%%%%%


Ntrain = 60000; % MAX: 60,000
Ntest =  10000; % MAX: 10,000
% Ntrain=NTest --> 10,000 -> 98%, 2,000 -> 93%, 1,000 -> 91.6%.
% NTrain --> 60,000, NTest --> 99.0%.

trainImagesFile = 'D:\Documents\MATLAB\THESIS\Datasets\MNIST\train-images.idx3-ubyte';
trainLabelsFile = 'D:\Documents\MATLAB\THESIS\Datasets\MNIST\train-labels.idx1-ubyte';
testImagesFile = 'D:\Documents\MATLAB\THESIS\Datasets\MNIST\t10k-images.idx3-ubyte';
testLabelsFile = 'D:\Documents\MATLAB\THESIS\Datasets\MNIST\t10k-labels.idx1-ubyte';

XTrain = processImagesMNIST(trainImagesFile);
YTrain = processLabelsMNIST(trainLabelsFile);

XTest = processImagesMNIST(testImagesFile);
YTest = processLabelsMNIST(testLabelsFile);

numTrainImages = size(XTrain,4);

XTrain1 = extractdata(XTrain(:,:,1,1:Ntrain));
YTrain1 = YTrain(1:Ntrain);
XTest1 = XTest(:,:,1,1:Ntest);
YTest1 = YTest(1:Ntest);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M0options = trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',20,'MiniBatchSize',64,'Plots','training-progress');
[M0mnist,infoM0] = trainNetwork(XTrain1,YTrain1,netM0,M0options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

