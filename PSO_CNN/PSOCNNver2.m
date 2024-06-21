%clc;
%clear;
%close all;

%load('D:\Documents\MATLAB\THESIS\Datasets\IR Database\CVL Database Itir 8_16bit\ConceptNET2.mat');

%% Parameters of PSO

%IR1=imread('D:\Documents\MATLAB\THESIS\Datasets\IR Database\FLIR_ADAS_1_3\train\thermal_8_bit\FLIR_00018.jpeg'); %ConceptNET2.mat
%IR1=imread('D:\Documents\MATLAB\THESIS\Datasets\IR Database\FLIR_ADAS_1_3\train\thermal_8_bit\FLIR_00006.jpeg'); %ConceptNET.mat
%IR1=imread('D:\Documents\MATLAB\THESIS\Datasets\IR Database\CVL Database Itir 8_16bit\4IRimgBW\car\00001420.png'); %ConceptNET.mat
%IR1=imread('D:\Desktop\ObjetosBase\ImagePre\FLIR_00018.jpeg'); %ConceptNET4.mat

[X,Y]=size(IR1);

MaxIt = 3; %3   %100;          % Maximum Number of Iterations
w = 0.1;                       % Inertia Coefficient
wdamp = 0.99; %1.3; %1.3; %0.99; 1.1;      % Damping Ratio of Inetia Coefficient
c1 = 0.1; %0.2    %0.4; 0.8;       % Personal Acceleration Coeffient
c2 = 0.1; %0.2    %0.4; 0.8;       % Social Acceleration Coeffient
Width = 24;

%% Show Image

figure(1)
imshow(IR1);
hold on
%plot(247,271,'b*')

%% Initialization

% Start Creating Particles, Distributed Location Solution

k=1;
for i1=1:(1)*Width:floor(X/Width)*Width %floor((X-Width)/(2*Width))+1
    for j1=1:(1)*Width:floor(Y/Width)*Width %floor((Y-Width)/(2*Width))+1
        particle(k,1) = i1; %i*2*Width-2*Width+1;
        particle(k,2) = j1; %j*2*Width-2*Width+1;
        particle(k,8) = inf;
        %plot(particle(k,2),particle(k,1),'ro')
        rectangle('Position',[particle(k,2),particle(k,1),Width,Width],'EdgeColor','r','LineWidth',2);
        k=k+1;
    end
end
hold off

nPop = k-1;

% Initialize Global Best
GlobalBestCost = inf;
GlobalBestPosition = [1,1];

% Initialize Population Members
for i=1:nPop

    % Initialize Velocity
    particle(i,5) = 0;
    particle(i,6) = 0;

    % Evaluation & Cost Function
    particleMatrix(1:Width,1:Width) = IR1(floor(particle(i,1)):floor(particle(i,1))+Width-1,floor(particle(i,2)):floor(particle(i,2))+Width-1);
        
    [Label,Score]=classify(M0mnist,particleMatrix); %uint8()
    particle(i,7) = 1 - Score(1,2);
    
    %particle(i,7) = 146880 - sum(sum(uint8(particleMatrix)));
        
     
    % Update the Personal Best
   if particle(i,7) < particle(i,8)
    particle(i,3) = particle(i,1);
    particle(i,4) = particle(i,2);
    particle(i,8) = particle(i,7);
   end
   
   % Update Global Best
   if particle(i,8) < GlobalBestCost
    GlobalBestCost = particle(i,8);
    GlobalBestPosition(1) = round(particle(i,3));
    GlobalBestPosition(2) = round(particle(i,4));
   end
   
end

% Array to Hold Best Cost Value on Each Iteration
BestCosts = zeros(MaxIt, 1);

%% Main Loop of PSO

for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        AA = w*particle(i,5:6)...
            + c1*(rand)*(particle(i,3:4) - particle(i,1:2))...
            + c2*(rand)*(GlobalBestPosition - particle(i,1:2));
        particle(i,5) = AA(1);
        particle(i,6) = AA(2);
        
        % Update Position
        BB(i,1:2) = particle(i,1:2) + particle(i,5:6);
        particle(i,1) = abs(BB(i,1));
        particle(i,2) = abs(BB(i,2));
        %particle(i,1) = min(BB(i,1),1);
        %particle(i,2) = min(BB(i,2),1);
        %particle(i,1) = max(BB(i,1),X-50);
        %particle(i,2) = max(BB(i,2),Y-50);
        
%        if particle(i,1) > (X-Width)
%            particle(i,1) = 10; %(512-Width);
%        end
%        if particle(i,2) > (Y-Width)
%            particle(i,2) = 10; %(640-Width);
%        end

        
        % Evaluation & Cost Function
        particleMatrix(1:Width,1:Width) = IR1(floor(particle(i,1)):floor(particle(i,1))+Width-1,floor(particle(i,2)):floor(particle(i,2))+Width-1);
        
        [Label,Score]=classify(M0mnist,particleMatrix); %uint8()
        particle(i,7) = 1-Score(1,2);
        
        %particle(i,7) = 146880 - sum(sum(uint8(particleMatrix)));  
        
        % Update Personal Best
        if particle(i,7) < particle(i,8)            
            particle(i,3) = particle(i,1);
            particle(i,4) = particle(i,2);
            particle(i,8) = particle(i,7);
        end
           
        % Update Global Best
        if particle(i,8) < GlobalBestCost
         GlobalBestCost = particle(i,8);
         GlobalBestPosition(1) = round(particle(i,3));
         GlobalBestPosition(2) = round(particle(i,4));
        end
        
    end
    
    % Store the Best Cost Value
    BestCosts(it) = GlobalBestCost;
    
    % Display Iteration Information
    disp(['Iteration' num2str(it) ': Best Cost = ' num2str(BestCosts(it))]);
    
    % Damping Inertia Coefficient
    w = w * wdamp;
    
end

%% Results

%figure(2);
%plot(BestCosts, 'LineWidth', 2);
%xlabel('Iterations');
%ylabel('Best Cost');

%figure(3);
%semilogy(BestCosts, 'LineWidth', 2);
%xlabel('Iterations');
%ylabel('Best Cost');
%grid on;

figure(4)
imshow((IR1));
hold on
mm=0;
for i=1:nPop
    if particle(i,8) < 0.01 %== 0 %.16
        mm=mm+1;
        rectangle('Position',[particle(i,2),particle(i,1),Width,Width],'EdgeColor','y','LineWidth',2);
        %plot(particle(i,2),particle(i,1),'ro')
        sample(mm,1:2)=round(particle(i,1:2));
    end
end

%plot(247,256,'go')
%plot(247,280,'b+')
%plot(271,256,'b+')
%plot(271,280,'b+')

%GlobalBestPosition
mm;
rectangle('Position',[floor(GlobalBestPosition(2)),floor(GlobalBestPosition(1)),Width,Width],'EdgeColor','r','LineWidth',2);
hold off

figure(); imshow(IR1); hold on
idx=dbscan(sample,25,3);
gscatter(sample(:,2),sample(:,1),idx,'rgbm','.',20)
hold off

figure(); imshow(IR1); hold on
DATA=[sample idx];
Class0 = DATA(DATA(:,3)==-1,:);
Class1 = DATA(DATA(:,3)==1,:);
Class2 = DATA(DATA(:,3)==2,:);
Class3 = DATA(DATA(:,3)==3,:);

[C1Val1 C1ind1]=min(Class1(:,1)); [C1Val2 C1ind2]=min(Class1(:,2));
[C2Val1 C2ind1]=min(Class2(:,1)); [C2Val2 C2ind2]=min(Class2(:,2));
[C3Val1 C3ind1]=min(Class3(:,1)); [C3Val2 C3ind2]=min(Class3(:,2));

C1W1=(max(Class1(:,1))+23) - C1Val1;
C1W2=(max(Class1(:,2))+23) - C1Val2;

C2W1=(max(Class2(:,1))+23) - C2Val1;
C2W2=(max(Class2(:,2))+23) - C2Val2;

C3W1=(max(Class3(:,1))+23) - C3Val1;
C3W2=(max(Class3(:,2))+23) - C3Val2;

if C1W1<24
    C1W1=24;
end
if C1W2<24
    C1W2=24;
end
if C2W1<24
    C2W1=24;
end
if C2W2<24
    C2W2=24;
end
if C3W1<24
    C3W1=24;
end
if C3W2<24
    C3W2=24;
end
if size(Class0,1)>0
    for i=1:size(Class0,1)
        rectangle('Position',[Class0(i,2),Class0(i,1),24,24],'EdgeColor','r','LineWidth',2);
    end
end
if size(Class1,1)>0
    rectangle('Position',[Class1(C1ind2,2),Class1(C1ind1,1),C1W2,C1W1],'EdgeColor','g','LineWidth',2);
end
if size(Class2,1)>0
    rectangle('Position',[Class2(C2ind2,2),Class2(C2ind1,1),C2W2,C2W1],'EdgeColor','b','LineWidth',2);
end
if size(Class3,1)>0
    rectangle('Position',[Class3(C3ind2,2),Class3(C3ind1,1),C3W2,C3W1],'EdgeColor','y','LineWidth',2);
end

hold off