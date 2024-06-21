function out = PSO(problem, params)

%% Problem Definition

CostFunction = problem.CostFunction; % Cost Function

nVar = problem.nVar;                 % Number of Unknown (Decision Variables)

VarSize = [1 nVar];                  % Matrix Size of Decision Variables

VarMin = problem.VarMin;             % Lower Bound of Decision Variables
VarMax = problem.VarMax;             % Upper Bound of Decision Variables


%% Parameters of PSO

MaxIt = 1000;                   % Maximum Number of Iterations

nPop = 50;                     % Population Size (Swarm Size)

w = 1;                         % Inertia Coefficient
wdamp = 0.99;                  % Damping Ratio of Inetia Coefficient
c1 = 2;                        % Personal Acceleration Coeffient
c2 = 2;                        % Social Acceleration Coeffient


%% Initialization

% The Particle Template
empty_particle.Position = [];
empty_particle.Velocity = [];
empty_particle.Cost = [];
empty_particle.Best.Position = [];
empty_particle.Best.Cost = [];

% Create Population Array
particle = repmat(empty_particle, nPop, 1);

% Initialize Global Best
GlobalBest.Cost = inf;

% Initialize Population Members
for i=1:nPop
   
   % Generate Random Solution
   particle(i).Position = unifrnd(VarMin, VarMax, VarSize);
   
   % Initialize Velocity
   particle(i).Velocity = zeros(VarSize);
   
   % Evaluation
   particle(i).Cost = CostFunction(particle(i).Position);
    
   % Update the Personal Best
   particle(i).Best.Position = particle(i).Position;
   particle(i).Best.Cost = particle(i).Cost;
   
   % Update Global Best
   if particle(i).Best.Cost < GlobalBest.Cost
       GlobalBest = particle(i).Best;
   end
   
end

% Array to Hold Best Cost Value on Each Iteration
BestCosts = zeros(MaxIt, 1);

%% Main Loop of PSO

for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity...
            + c1*rand(VarSize).*(particle(i).Best.Position - particle(i).Position)...
            + c2*rand(VarSize).*(GlobalBest.Position - particle(i).Position);
        
        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Evaluation
        particle(i).Cost = CostFunction(particle(i).Position);
        
        % Update Personal Best
        if particle(i).Cost < particle(i).Best.Cost
            
            particle(i).Best.Position = particle(i).Position;
            particle(i).Best.Cost = particle(i).Cost;
           
            % Update Global Best
            if particle(i).Best.Cost < GlobalBest.Cost
                GlobalBest = particle(i).Best;
            end
            
        end
        
    end
    
    % Store the Best Cost Value
    BestCosts(it) = GlobalBest.Cost;
    
    % Display Iteration Information
    disp(['Iteration' num2str(it) ': Best Cost = ' num2str(BestCosts(it))]);
    
    % Damping Inertia Coefficient
    w = w * wdamp;
    
end

end