%clc;

% STEP 1: Load Initial IR Image
Io=imread('D:\Documents\MATLAB\THESIS\ExperIR_April22\w5IR\car\FLIR_00001.jpeg');
S1=Io;
figure(1); imshow(Io)
title('STEP 1: Load Initial IR Image')
Iin=double(Io);

% STEP 2: Image Contrast Enhanced with Wavelets
Imc=ContrFunc(Iin,'haar');
S2=uint8(Imc);
figure(2); imshow(uint8(Imc))           
title('STEP 2: Image Contrast Enhanced with Wavelets')

% STEP 3: Image Filtered with Wavelets and Soft Threshold
Im2=NoiseWaveFilter(Imc,'db2'); 
S3=uint8(Im2);
figure(3); imshow(uint8(Im2))         
title('STEP 3: Image Filtered with Wavelets and Soft Threshold')

% STEP 4: Image Binarized with High and Low Thresholds
Im=BinaryThreshold(Im2,135,80); % 120,107   112,107  90,80   113,107sinContr  120,80
S4=uint8(Im);
figure(4); imshow(uint8(Im))
title('STEP 4: Image Binarized with High and Low Thresholds')

% STEP 5: Median Filter 3x3 Applied to the Image
Im1=MedianFilter(Im); %(3x3)
S5=uint8(Im1);
figure(5); imshow(uint8(Im1))
title('STEP 5: Median Filter 3x3 Applied to the Image')

% STEP 6: Edges Extrated from Image with Wavelets
wave='db2';  % 'sym4','fk6'
[A,H,V,D]=dwt2(Im1,wave); A=A*0;
Ied=idwt2(A,H,V,D,wave);
S6=uint8(5*Ied);
figure(6); imshow(uint8(5*Ied))
title('STEP 6: Edges Extrated from Image with Wavelets')

% STEP 7: Analog AND using STEP-3 and STEP-5 Images
S7=uint8(Im2).*(Im1/255);
figure(7); imshow(S7)
title('STEP 7: Analog AND using STEP-3 and STEP-5 Images')
