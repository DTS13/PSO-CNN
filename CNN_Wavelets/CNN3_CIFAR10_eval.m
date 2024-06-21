%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%
%%%%%%%%%%%%  CNN CIFAR10  %%%%%%%%%%%%


testpreds = classify(M0mnist,test);
figure()
plotconfusion(testpreds,test.Labels)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%im = imread('D:\Documents\MATLAB\THESIS\ExperCIFAR10_5Jul21\CIFAR10\0\image1_30.jpg');

%actvn1 = activations(M0mnist,im,'Batch1c');
%actvn2 = activations(M0mnist,im,'Batch2c');
%actvn3 = activations(M0mnist,im,'Batch3c');
%actvn4 = activations(M0mnist,im,'Batch4c');
%actvn5 = activations(M0mnist,im,'Batch3D1');
%actvn6 = activations(M0mnist,im,'PoolD1');
%actvn7 = activations(M0mnist,im,'Batch1D1');
%actvn8 = activations(M0mnist,im,'Batch2D1');
%actvn9 = activations(M0mnist,im,'Batch3D1');
%actvn10 = activations(M0mnist,im,'Batch4D1');

%figure(3); montage(rescale(actvn1))
%figure(4); montage(rescale(actvn2))
%figure(5); montage(rescale(actvn3))
%figure(6); montage(rescale(actvn4))
%figure(7); montage(rescale(actvn5))
%figure(8); montage(rescale(actvn6))
%figure(9); montage(rescale(actvn7))
%figure(10); montage(rescale(actvn8))
%figure(11); montage(rescale(actvn9))
%figure(12); montage(rescale(actvn10))

