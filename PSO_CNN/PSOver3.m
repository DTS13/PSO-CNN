clc;
clear;
close all;

%% Parameters of PSO

IR1=imread('D:\Documents\MATLAB\THESIS\Datasets\IR Database\FLIR_ADAS_1_3\train\thermal_8_bit\FLIR_00018.jpeg');
IR1i=int8(IR1);
[X,Y]=size(IR1i);

MaxIt = 2500;                  % Maximum Number of Iterations

w = 0.1;                       % Inertia Coefficient
wdamp = 0.99;                   % Damping Ratio of Inetia Coefficient
c1 = 0.1;                        % Personal Acceleration Coeffient
c2 = 0.1;                        % Social Acceleration Coeffient

Width = 25;

nPop = (floor((X-Width)/(2*Width))+1)*(floor((Y-Width)/(2*Width))+1); %50; % Population Size (Swarm Size)

%% Initialization

% Create Particles
particle = zeros(nPop,8); %Position, BestPosition, Velocity, Cost, BestCost
particleMatrix = zeros(nPop,Width,Width);

% Initialize Global Best
GlobalBestCost = inf;
GlobalBestPosition = [0,0];

% Start Distributed Particles Solution

particle(1,:) = [247,271,0,0,0,0,0,0];
figure(1)
imshow(IR1);
hold on
plot(particle(1,2),particle(1,1),'ro')


k=1;
for i=1:floor((X-Width)/(2*Width))+1
    for j=1:floor((Y-Width)/(2*Width))+1
        particle(k,1) = i*2*Width-2*Width+1;
        particle(k,2) = j*2*Width-2*Width+1;
        particle(k,8) = inf;
        plot(particle(k,2),particle(k,1),'ro')
        k=k+1;
    end
end
hold off

% Initialize Population Members
for i=1:nPop

    % Initialize Velocity
    particle(i,5:6) = [0,0];

    % Evaluation & Cost Function
        particleMatrix(i,1:Width,1:Width) = IR1(floor(particle(i,1)):floor(particle(i,1))+Width-1,floor(particle(i,2)):floor(particle(i,2))+Width-1);
        particleMatrixB(1:25,1:25) = int8(particleMatrix(i,1:Width,1:Width));
        particle(i,7) = sqrt(sum(sum((IR1i(247:271,256:280)-particleMatrixB(1:25,1:25)).^2)));
        %particle(i,7) = -IR1(floor(particle(i,1)),floor(particle(i,2)));
     
    % Update the Personal Best
    particle(i,3:4) = particle(i,1:2);
    particle(i,8) = particle(i,7);
   
   % Update Global Best
   if particle(i,8) < GlobalBestCost
       GlobalBestCost = particle(i,8);
       GlobalBestPosition = particle(i,3:4);
   end
   
end

% Array to Hold Best Cost Value on Each Iteration
BestCosts = zeros(MaxIt, 1);

%% Main Loop of PSO

for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        particle(i,5:6) = w*particle(i,5:6)...
            + c1*(rand)*abs(particle(i,3:4) - particle(i,1:2))...
            + c2*(rand)*abs(GlobalBestPosition - particle(i,1:2));

        %if particle(i,8) == 0
        %    particle(i,5:6) = [0,0];
        %end   
        
        % Update Position
        particle(i,1:2) = particle(i,1:2) + particle(i,5:6);

        
        if particle(i,1) > (512-Width)
            particle(i,1) = 10; %(512-Width);
        end
        if particle(i,2) > (640-Width)
            particle(i,2) = 10; %(640-Width);
        end

        
        % Evaluation & Cost Function
        particleMatrix(i,1:Width,1:Width) = IR1(floor(particle(i,1)):floor(particle(i,1))+Width-1,floor(particle(i,2)):floor(particle(i,2))+Width-1);
        particleMatrixB(1:25,1:25) = int8(particleMatrix(i,1:Width,1:Width));
        particle(i,7) = sqrt(sum(sum((IR1i(247:271,256:280)-particleMatrixB(1:25,1:25)).^2)));
        %particle(i,7) = -IR1(floor(particle(i,1)),floor(particle(i,2)));
   
        % Update Personal Best
        if particle(i,7) < particle(i,8)            
            particle(i,3:4) = particle(i,1:2);
            particle(i,8) = particle(i,7);
           
            % Update Global Best
            if particle(i,8) < GlobalBestCost
                GlobalBestCost = particle(i,8);
                GlobalBestPosition = particle(i,3:4);
            end
            
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

figure;
plot(BestCosts, 'LineWidth', 2);
xlabel('Iterations');
ylabel('Best Cost');

figure;
semilogy(BestCosts, 'LineWidth', 2);
xlabel('Iterations');
ylabel('Best Cost');
grid on;


figure(1)
imshow((IR1));
hold on
plot(particle(1,1),particle(1,2),'ro')

for i=1:nPop
    if particle(i,8) < 10
        plot(particle(i,1),particle(i,2),'ro')
    end
end

plot(247,256,'go')
plot(247,280,'b+')
plot(271,256,'b+')
plot(271,280,'b+')
hold off

GlobalBestPosition

rectangle('Position',[floor(GlobalBestPosition(1)),floor(GlobalBestPosition(2)),Width,Width],'EdgeColor','b','LineWidth',3);
