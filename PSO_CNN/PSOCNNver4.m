%%clc;
%%clear;
%%close all;

%load('D:\Documents\MATLAB\THESIS\Datasets\IR Database\CVL Database Itir 8_16bit\ConceptNET2.mat');
%%load('D:\Documents\MATLAB\THESIS\Datasets\IR Database\CVL Database Itir 8_16bit\ConceptNET4.mat');

%% Parameters of PSO

%IR1=imread('D:\Documents\MATLAB\THESIS\Datasets\IR Database\FLIR_ADAS_1_3\train\thermal_8_bit\FLIR_00018.jpeg'); %ConceptNET2.mat
%IR1=imread('D:\Documents\MATLAB\THESIS\Datasets\IR Database\FLIR_ADAS_1_3\train\thermal_8_bit\FLIR_00006.jpeg'); %ConceptNET.mat
%IR1=imread('D:\Documents\MATLAB\THESIS\Datasets\IR Database\CVL Database Itir 8_16bit\4IRimgBW\car\00001420.png'); %ConceptNET.mat

%%IR1=imread('D:\Desktop\ObjetosBase\ImagePre\FLIR_00018.jpeg'); %ConceptNET4.mat
%%[X,Y]=size(IR1);


tic


MaxIt = 1; %3   %100;          % Maximum Number of Iterations
w = 0.3;                       % Inertia Coefficient
wdamp = 1.3; %1.3; %1.3; %0.99; 1.1;      % Damping Ratio of Inertia Coefficient
c1 = 0.1; %0.2    %0.4; 0.8;       % Personal Acceleration Coeffient
c2 = 0.1; %0.2    %0.4; 0.8;       % Social Acceleration Coeffient
Width = 24;


%% Show Image

%figure(1)
%imshow(IR1);
%hold on
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
        %rectangle('Position',[particle(k,2),particle(k,1),Width,Width],'EdgeColor','r','LineWidth',2);
        k=k+1;
    end
end
%particle(:,8) = inf;
%hold off

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
    particleMatrix(1:Width,1:Width,1:1) = IR1(floor(particle(i,1)):floor(particle(i,1))+Width-1,floor(particle(i,2)):floor(particle(i,2))+Width-1);
        
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
    fprintf('Iteration: %d Best Cost: %1.2f \n',it,(1-BestCosts(it)));
    
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

GlobalBestPosition

rectangle('Position',[floor(GlobalBestPosition(2)),floor(GlobalBestPosition(1)),Width,Width],'EdgeColor','r','LineWidth',2);
hold off

figure();
imshow(IR1);
hold on
idx=dbscan(sample,25,3);
gscatter(sample(:,2),sample(:,1),idx,'rgbm','.',20)
hold off

figure();
imshow(IR1);
hold on
%DATA=[sample idx];
uv=unique(idx);

for j=1:size(uv,1)
    if uv(1,1) == 1
        eval(['ROI',num2str(j),'=sample(idx(:)==',num2str(j),',:);']); %ROI1 = DATA(DATA(:,3)==1,:);
        eval(['[R',num2str(j),'Val1',' R',num2str(j),'ind1]=min(ROI',num2str(j),'(:,1));']);
        eval(['[R',num2str(j),'Val2',' R',num2str(j),'ind2]=min(ROI',num2str(j),'(:,2));']);
        eval(['R',num2str(j),'W1=(max(ROI',num2str(j),'(:,1))+23)-R',num2str(j),'Val1;']);
        eval(['R',num2str(j),'W2=(max(ROI',num2str(j),'(:,2))+23)-R',num2str(j),'Val2;']);            
        if eval(['R',num2str(j),'W1<24'])
            eval(['R',num2str(j),'W1=24;']);
        end
        if eval(['R',num2str(j),'W2<24'])
            eval(['R',num2str(j),'W2=24;']);
        end    
    else
        %if uv(1,1) == -1
            eval(['ROI0=sample(idx(:)==-1,:);']); %ROI0 = DATA(DATA(:,3)==-1,:);
        %else
            %ROI0=[];
            eval(['ROI',num2str(j-1),'=sample(idx(:)==',num2str(j-1),',:);']);            
            eval(['[R',num2str(j-1),'Val1',' R',num2str(j-1),'ind1]=min(ROI',num2str(j-1),'(:,1));']);
            eval(['[R',num2str(j-1),'Val2',' R',num2str(j-1),'ind2]=min(ROI',num2str(j-1),'(:,2));']);
            eval(['R',num2str(j-1),'W1=(max(ROI',num2str(j-1),'(:,1))+23)-R',num2str(j-1),'Val1;']);
            eval(['R',num2str(j-1),'W2=(max(ROI',num2str(j-1),'(:,2))+23)-R',num2str(j-1),'Val2;']);         
            if eval(['R',num2str(j-1),'W1<24'])
                eval(['R',num2str(j-1),'W1=24;']);
            end
            if eval(['R',num2str(j-1),'W2<24'])
                eval(['R',num2str(j-1),'W2=24;']);
            end
        %end
    end
    
end


if size(ROI0,1)>0
    for i=1:size(ROI0,1)
        rectangle('Position',[ROI0(i,2),ROI0(i,1),24,24],'EdgeColor','r','LineWidth',2);
    end
end

if uv(j,1) == j
    N=size(uv,1);
else
    N=size(uv,1)-1;
end    


for j=1:N    
    
    switch j
    case -1
        C='k';
    case 0
        C='r';
    case 1
        C='g';
    case 2
        C='b';
    case 3
        C='m';        
    otherwise
        C='y';
    end

    D1=eval(['ROI',num2str(j),'(R',num2str(j),'ind2,2)']);
    D2=eval(['ROI',num2str(j),'(R',num2str(j),'ind1,1)']);
    D3=eval(['R',num2str(j),'W2']);
    D4=eval(['R',num2str(j),'W1']);
    
    rectangle('Position',[D1,D2,D3,D4],'EdgeColor',C,'LineWidth',2);
end
toc
hold off

