%car
%bicycle
%person
%dog

srcFiles = dir('D:\Documents\MATLAB\THESIS\ExperIR_April22\w3IR\car\*.jpeg');  % the folder in which ur images exists
for i = 1 : length(srcFiles)
filename = strcat('D:\Documents\MATLAB\THESIS\ExperIR_April22\w3IR\car\',srcFiles(i).name);
im = imread(filename);
k=imresize(im,[64,64]);
newfilename=strcat('D:\Documents\MATLAB\THESIS\ExperIR_April22\w3IR\car\',srcFiles(i).name);
imwrite(k,newfilename,'jpeg');
end