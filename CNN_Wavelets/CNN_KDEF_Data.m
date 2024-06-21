%%%%%%%%%% Resize IR Dataset for CNN KDEF %%%%%%%%

% EXECUTION ORDER:
% 1-CNN_KDEF_Data.m          Resize all dataset
% 2-CNN_KDEF_Maxpool.m
% 3-CNN_KDEF_trainNrm.m
% 4-CNN_KDEF_test.m

clc;
Size=128;   % CNN SVHN

%%%%%%%%%%%%  I. IMAGE DATASET PREPROCESSED AND READY  %%%%%%%%%%%%%%%%%%%
% Creating image data storage to EXTRACT filenames and addresses.
imds = imageDatastore('w4IR','IncludeSubfolders',true,'LabelSource','foldernames');
% CREATE LABELS for the Objects in the DATABASE.
w4IRnames = imds.Labels;
% SET NUMBER OF CLASSES in the DATABASE.
numClasses = numel(categories(w4IRnames));
% Extract filenames into an array.
fileNames = imds.Files;
save('fileNames.mat','fileNames');


for k = 1 : length(fileNames)
	thisFileName = fileNames{k};
    img=imresize(imread(thisFileName),[Size,Size]); %>--Resize Image to: Size x Size x 3
    imwrite(img,thisFileName);
    %delete(thisFileName);
end
