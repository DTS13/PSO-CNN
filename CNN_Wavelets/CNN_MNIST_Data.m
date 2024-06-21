%%%%%%%%%% Resize IR Dataset for CNN MNIST %%%%%%%%

% EXECUTION ORDER:
% 1-CNN_MNIST_Data.m          Resize all dataset
% 2-CNN_MNIST_Maxpool.m
% 3-CNN_MNIST_trainNrm.m
% 4-CNN_MNIST_test.m

clc;
Size=28;   % CNN MNIST 

%%%%%%%%%%%%  I. IMAGE DATASET PREPROCESSED AND READY  %%%%%%%%%%%%%%%%%%%
% Creating image data storage to EXTRACT filenames and addresses.
imds = imageDatastore('w1IR','IncludeSubfolders',true,'LabelSource','foldernames');
% CREATE LABELS for the Objects in the DATABASE.
w1IRnames = imds.Labels;
% SET NUMBER OF CLASSES in the DATABASE.
numClasses = numel(categories(w1IRnames));
% Extract filenames into an array.
fileNames = imds.Files;
save('fileNames.mat','fileNames');


for k = 1 : length(fileNames)
	thisFileName = fileNames{k};
    img=imresize(imread(thisFileName),[Size,Size]); %>--Resize Image to: Size x Size x 3
    imwrite(img,thisFileName);
    %delete(thisFileName);
end
