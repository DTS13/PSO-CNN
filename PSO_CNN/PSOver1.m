clc;
I=2;
N=500;

[X Y] = size(IR1);
[x y] = size(ir1);
P(:,1)=randi([1 X-x],N,1);
P(:,2)=randi([1 Y-y],N,1);
Vx(:,1)=rand(N,1);
Vy(:,2)=rand(N,1);
LBest(1:N,3)=100000;  %Cost
Cn(1:N,3)=100000; %Cost


figure(1)
imshow(uint8(IR1))
%hold on

for j=1:I
    
    for i=1:N
        Cn(i,3)=(sqrt(sum(sum(IR1(floor(P(i,1)):floor(P(i,1)+(x-1)),floor(P(i,2)):floor(P(i,2)+(y-1)) - ir1))))); %Cost
        if Cn(i,3) < LBest(i,3)
            LBest(i,3) = Cn(i,3)
            LBest(i,2) = P(i,2)
            LBest(i,1) = P(i,1)
        end    
    
    [GBest(j,3) H] = min(LBest(:,3))
    GBest(j,1) = LBest(H,1)
    GBest(j,2) = LBest(H,2)
    
    Vx(i,1)=2*rand*Vx(i,1) + 2*rand*(LBest(i,1)-P(i,1)) + 2*rand*(GBest(j,1)-P(i,1));
    Vy(i,2)=2*rand*Vy(i,1) + 2*rand*(LBest(i,2)-P(i,2)) + 2*rand*(GBest(j,2)-P(i,2));
    
    P(i,1)=P(i,1)+V(i,1);
    P(i,2)=P(i,2)+V(i,2);
    end

    
end

figure(2)
plot(GBest(1,2),GBest(1,2),'ro')
hold on
for i=1:N
    plot(GBest(1,2),GBest(1,2),'ro')
end


