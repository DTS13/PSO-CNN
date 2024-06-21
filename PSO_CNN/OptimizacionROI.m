clear all
clc

tic

% Parametros del algoritmo
n=6;          % Número de angulos o variables
N=1000;      % Número de muestras o puntos totales  10000 (1000)
%Vi=15;       % Tamaño de la vecindad inicial        7     (3)
FR=0.7;      % Factor de Reducción de la Vecindad   0.99       (1)
Error=0.00009;    % Error mínio aceptable: 0.009
J=0;          % Contador de paro
Jm=1;        % Número máximo de iteraciones

% Momentos de objetos a buscar.
Car1=imread('Car1.jpeg');
Car1=double(Car1);
MObjeto1(1,1:7) = InvMom2(Car1);
%MObjeto2(1,1:7) = [0.95,,,,,,];
%MObjeto3(1,1:7) = [0.95,,,,,,];

%Genera la posición inicial de manera aleatoria.
Imagen=imread('FLIR_00001.jpeg');
Imagen=double(Imagen);
[X,Y,Z]=size(Imagen);

Xf(1,1) = floor(65 + rand*(X-128));
Yf(1,1) = floor(65 + rand*(Y-128));

XYZf(1,1:3)=[Xf(1,1),Yf(1,1),1];
Xc = Xf(1,1);
Yc = Yf(1,1);
Zc = 1;
FrgImg = Imagen(Xf(1,1)-64:Xf(1,1)+63,Yf(1,1)-64:Yf(1,1)+63,1);
% size(FrgImg)

MpROI = InvMom2(FrgImg);
dist(1,1) = sqrt(abs(sum(MObjeto1.^2 - MpROI.^2)));
dbest = dist(1,1);

figure(1)
imshow(uint8(Imagen))
hold on
plot(Yc,Xc,'go')


while (dbest>Error)
V=5*dbest;

J=J+1;
if J > Jm
    break
end

% T=[rand(N,n)*360-180];
%Genera los puntos aleatorios totales de 6 angulos cada uno, todos dentro de la vecindad V.

for i=2:N

Xf(i,1) = floor(65 + rand*(X-128));
Yf(i,1) = floor(65 + rand*(Y-128));

%while (Xf(i,1)+128) > X | Xf(i,1) <= 0
%    Xf(i,1) = floor(rand*(X-128));
%end

%while (Yf(i,1)+128) > Y | Yf(i,1) <= 0
%    Yf(i,1) = floor(rand*(Y-128));
%end

% Zf(i,1) = ceil(rand*(Z-0));
XYZf(i,1:3)=[Xf(i,1),Yf(i,1),1];
FrgImg = Imagen(Xf(i,1)-64:Xf(i,1)+63,Yf(i,1)-64:Yf(i,1)+63,1);
MpROI = InvMom2(FrgImg);
dist(i,1) = sqrt(abs(sum(MObjeto1.^2 - MpROI.^2)));

%    XYZ(i,:) = FitnessUR5(T(i,1),T(i,2),T(i,3),T(i,4),T(i,5),T(i,6));
%    dist(i,1)=sqrt((XYZ(i,1)-X)^2 + (XYZ(i,2)-Y)^2 + (XYZ(i,3)-Z)^2);
plot(Yf(i,1),Xf(i,1),'b+')

end

[d,a]=min(dist);

if d < dbest
    dbest=d
    Xc=XYZf(a,1);
    Yc=XYZf(a,2);
    Zc=XYZf(a,3);
else
    dbest=dbest;
    Xc=Xc;
    Yc=Yc;
    Zc=Zc;
end


%%%%% Grafica la posición actual %%%%%
%figure(1)
%plot3(Xc,Yc,Zc,'b+')
%hold on

%Xa=X1:(Xc-X1)/10:Xc;
%Ya=Y1:(Yc-Y1)/10:Yc;
%Za=Z1:(Zc-Z1)/10:Zc;
%plot3(Xa,Ya,Za,'k')

%V=V-(Error/dbest); % *FR

end

%%%%% Grafica la posición inicial %%%%%
%imshow(uint8(Imagen))
%hold on
%plot(XYZf(1,1),XYZf(1,2),'bo')
%%%%% Grafica la posición deseada %%%%%
% plot3(X,Y,Z,'go')
%%%%% Grafica el Origen %%%%%
%plot3(0,0,0,'g+')
%%%%% Grafica la posición final %%%%%
%plot(Xc,Yc,'ro')

%[X,Xc;Y,Yc;Z,Zc;Jm,J;V,V;d,dbest]
plot(Yc,Xc,'rx')
plot(Yc,Xc,'ro')

toc