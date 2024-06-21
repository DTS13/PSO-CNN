clear all;
clc;

ir1=imread('D:\Documents\MATLAB\THESIS\ExperIR_April22\IR\car\FLIR_00070.jpeg');
IR1=imread('D:\Documents\MATLAB\THESIS\Datasets\IR Database\FLIR_ADAS_1_3\train\thermal_8_bit\FLIR_00018.jpeg');
ir1=double(ir1);
IR1=double(IR1);

figure(1)
imshow(uint8(ir1))

figure(2)
imshow(uint8(IR1))