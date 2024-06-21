%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%
%%%%%%%%%%%%  IR by CNN SVHN TEST  %%%%%%%%%%%%

testpreds = classify(M0IR,test);
plotconfusion(testpreds,test.Labels)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%im = imread('D:\Documents\MATLAB\THESIS\Experim9jun21\MNIST\0\0_te11.jpg');

%actvn1 = activations(M0mnist,im,'Batch1c');
%actvn2 = activations(M0mnist,im,'Batch2c');
%actvn3 = activations(M0mnist,im,'Batch3c');

%figure(); montage(rescale(actvn1))
%figure(); montage(rescale(actvn2))
%figure(); montage(rescale(actvn3))
