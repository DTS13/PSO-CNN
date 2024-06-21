%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%
%%%%%%%%%%%  CNN CIFAR10 Train  %%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%  I. CONFIGURE DATA  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creating new data storage for the neural network.
imds = imageDatastore('CIFAR10','IncludeSubfolders',true,'LabelSource','foldernames');
%[train,test] = splitEachLabel(imds,0.8571,'randomized');
[train,val,test] = splitEachLabel(imds,0.8333,0.0833,0.0833,'randomized');

% CREATE LABELS for the Objects in the DATABASE.
names = imds.Labels;
% SET NUMBER OF CLASSES in the DATABASE.
numClasses = numel(categories(names));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M0options = trainingOptions('sgdm',...
'InitialLearnRate',0.001,...
'MaxEpochs',50,...
'MiniBatchSize',64,...
'ValidationFrequency',50,...
'ValidationPatience',21,...
'ValidationData',{val,val.Labels},...
'Plots','training-progress');
% Val.Pat.:15,17,21

[M0mnist,infoM0] = trainNetwork(train,netM0,M0options);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    