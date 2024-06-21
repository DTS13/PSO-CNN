%%%%%%%%%% Resize IR Dataset for CNN CIFAR10 %%%%%%%%

% EXECUTION ORDER:
% 1-CNN_CIFAR10_Data.m          Resize all dataset
% 2-CNN_CIFAR10_Maxpool.m
% 3-CNN_CIFAR10_trainNrm.m
% 4-CNN_CIFAR10_test.m

clc;
Size=32;   % CNN CIFAR10 

%%%%%%%%%%%%  I. IMAGE DATASET PREPROCESSED AND READY  %%%%%%%%%%%%%%%%%%%
% Creating image data storage to EXTRACT filenames and addresses.
imds = imageDatastore('w2IR','IncludeSubfolders',true,'LabelSource','foldernames');
% CREATE LABELS for the Objects in the DATABASE.
w2IRnames = imds.Labels;
% SET NUMBER OF CLASSES in the DATABASE.
numClasses = numel(categories(w2IRnames));
% Extract filenames into an array.
fileNames = imds.Files;
save('fileNames.mat','fileNames');


for k = 1 : length(fileNames)
	thisFileName = fileNames{k};
    img=imresize(imread(thisFileName),[Size,Size]); %>--Resize Image to: Size x Size x 3
    imwrite(img,thisFileName);
    %delete(thisFileName);
end
