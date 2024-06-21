function Out=ContrFunc(In,wave)

In=double(In);
%figure(20); histogram(In)
%title('Initial Histogram')

%%% Contrast, METHOD 1: Exponential %%%
MAX=max(max(In)); k=1.01; % k just above 1.
c=255/((k^MAX)-1);
Ie=c*((k.^In)-1);

%figure(21); histogram(Ie)
%title('Contrast, Method 1: Exponential')
%figure(31); imshow(uint8(Ie))
%title('Contrast, Method 1: Exponential')


%%% Contrast METHOD 2: Wavelets %%%
[A,H,V,D]=dwt2(Ie,wave); k=5;
Out=idwt2(A,H*k,V*k,D*k,wave);

%figure(22); histogram(Out)
%title('Contrast, Method 2: Wavelets')
%figure(32); imshow(uint8(Out))
%title('Contrast, Method 2: Wavelets')


%%% Contrast, METHOD 3: Max-min %%%
%MAX=max(max(In));
%MIN=min(min(In));
%Ix=(255/(MAX-MIN))*(In-MIN);

%figure(23); histogram(Ix)
%title('Contrast, Method 3: Max-min')
%figure(33); imshow(uint8(Ix))
%title('Contrast, Method 3: Max-min')


%%% Contrast, METHOD 4: Gamma %%%
%Ig=In.^2.22; % 1, 2.22, 0.45
%Ig=real(Ig/1000);
%MAX=max(max(Ig));
%MIN=min(min(Ig));
%Ig=(255/(MAX-MIN))*(Ig-MIN);

%figure(24); histogram(Ig)
%title('Contrast, Method 4: Gamma')
%figure(34); imshow(uint8(Ig))
%title('Contrast, Method 4: Gamma')


%%% Contrast, METHOD 5: Logarithmic %%%
%MAX=max(max(In));   % STRETCH HISTOGRAM
%c=255/log(1+MAX);
%Il=c*log(1+In);

%figure(25); histogram(real(Il))
%title('Contrast, Method 5: Logarithmic')
%figure(35); imshow(uint8(real(Il)))
%title('Contrast, Method 5: Logarithmic')

end