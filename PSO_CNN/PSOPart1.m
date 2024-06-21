clc;
N=500;

[X,Y]=size(IR1);
Vent=20;
X=X-Vent;
Y=Y-Vent;

V1=110;
V2=180;

C11=V1;   C12=V1;   C13=V1;   C14=V1;
C21=V1;   C22=V1;   C23=V1;   C24=V1;
C31=V1;   C32=V2;   C33=V2;   C34=V2;
C41=V1;   C42=V2;   C43=V2;   C44=V2;

V=20*ones(N,2);
Vn=20*ones(N,2);

P=[randi([1 X],N,1),randi([1 Y],N,1)];
Pn(N,1) = P(N,1) + Vn(N,1);
Pn(N,2) = P(N,2) + Vn(N,2);

Cost=ones(N,1);
Costn=zeros(N,1);
Cost1=1;
Cost2=1;
Cost3=1;
Cost4=1;

Cost=double(Cost);
Cost1=double(Cost1);
Cost2=double(Cost2);
Cost3=double(Cost3);
Cost4=double(Cost4);

for j=1:50
j
figure(2)
imshow(uint8(IR1))

    
hold on
for i=1:N

%Cost1=double((abs(IR1(B(i,1),B(i,2))-C11))+(abs(IR1(B(i,1),B(i,2)+1)-C12))+(abs(IR1(B(i,1),B(i,2)+2)-C13))+(abs(IR1(B(i,1),B(i,2)+3)-C14)));
%Cost2=double((abs(IR1(B(i,1)+1,B(i,2))-C21))+(abs(IR1(B(i,1)+1,B(i,2))-C22)+1)+(abs(IR1(B(i,1)+1,B(i,2)+2)-C23))+(abs(IR1(B(i,1)+1,B(i,2)+3)-C24)));
%Cost3=double((abs(IR1(B(i,1)+2,B(i,2))-C31))+(abs(IR1(B(i,1)+2,B(i,2)+1)-C32))+(abs(IR1(B(i,1)+2,B(i,2)+2)-C33))+(abs(IR1(B(i,1)+2,B(i,2)+3)-C34))+(abs(IR1(B(i,1)+3,B(i,2))-C41))+(abs(IR1(B(i,1)+3,B(i,2)+1)-C42))+(abs(IR1(B(i,1)+3,B(i,2)+2)-C43))+(abs(IR1(B(i,1)+3,B(i,2)+3)-C44)));

Pn(i,1) = abs( P(i,1) + Vn(i,1) );
Pn(i,2) = abs( P(i,2) + Vn(i,2) );

if Pn(i,1) > X
    Pn(i,1) = 246;
end
if Pn(i,2) > Y
    Pn(i,2) = 310;
end

Costn(i,1)=sum(sum(IR1(P(i,1):P(i,1)+Vent,P(i,2):P(i,2)+Vent)));

%Cost1=double(IR1(P(i,1),P(i,2)))+double(IR1(P(i,1)+1,P(i,2)))+double(IR1(P(i,1)+2,P(i,2)))+double(IR1(P(i,1)+3,P(i,2)));
%Cost2=double(IR1(P(i,1),P(i,2)+1))+double(IR1(P(i,1)+1,P(i,2)+1))+double(IR1(P(i,1)+2,P(i,2)+1))+double(IR1(P(i,1)+3,P(i,2)+1));
%Cost3=double(IR1(P(i,1),P(i,2)+2))+double(IR1(P(i,1)+1,P(i,2)+2))+double(IR1(P(i,1)+2,P(i,2)+2))+double(IR1(P(i,1)+3,P(i,2)+2));
%Cost4=double(IR1(P(i,1),P(i,2)+3))+double(IR1(P(i,1)+1,P(i,2)+3))+double(IR1(P(i,1)+2,P(i,2)+3))+double(IR1(P(i,1)+3,P(i,2)+3));

%Costn(i,1)=double(Cost1+Cost2+Cost3+Cost4);
    
if Cost(i,1) < Costn(i,1)  %Costn(i,1) < Cost(i,1)
    Cost(i,1) = Costn(i,1);
    P(i,1) = Pn(i,1);
    P(i,2) = Pn(i,2);
    %plot(P(i,2),P(i,1),'b.')
else
    Cost(i,1) = Cost(i,1);
    P(i,1) = P(i,1);
    P(i,2) = P(i,2);
    %plot(P(i,2),P(i,1),'ro')
end

%Vn=randi([-10 10],N,2);

plot(P(i,2),P(i,1),'b.')

[A B]=max(Cost);
plot(P(B,2),P(B,1),'ro')
plot(P(B,2),P(B,1),'r+')

%if Pn(i,1) > X
%    Pn(i,1) = 1;
%end
%if Pn(i,2) > Y
%    Pn(i,2) = 1;
%end

end
end

