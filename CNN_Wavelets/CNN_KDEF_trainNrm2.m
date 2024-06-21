%%%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%%%
%%%%%%%%%%  IR by CNN KDEF Train  %%%%%%%%%%

%%%%%%%%%%%%  I. CONFIGURE DATA  %%%%%%%%%%%%%%%%%
% Creating new data storage for the neural network.
imds = imageDatastore('w4IR','IncludeSubfolders',true,'LabelSource','foldernames');
[train,test] = splitEachLabel(imds,0.85,'randomized');

% CREATE LABELS for the Objects in the DATABASE.
names = imds.Labels;
% SET NUMBER OF CLASSES in the DATABASE.
numClasses = numel(categories(names));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M0options = trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',40,'MiniBatchSize',4,'Plots','training-progress');
[M0IR,infoM0] = trainNetwork(train,netM0,M0options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%