%%%%%%%%%% Proposed FrameWork %%%%%%%%%%%
%%%%%%%%%%%%  CNN MNIST Ref  %%%%%%%%%%%%


testpreds = classify(M0mnist,XTest1);
figure(2)
plotconfusion(testpreds,YTest1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

