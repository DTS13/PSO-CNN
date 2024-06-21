function Out=NoiseWaveFilter(In,wave)

T=4.5*std(std(In));        % Define Threshold
[A,H,V,D]=dwt2(In,wave);   % Wavelet Decomposition

[x,y]=size(A);
for i=1:x
    for j=1:y
        if abs(A(i,j)) > T
            An(i,j)=sign(A(i,j))*(abs(A(i,j))-T);
        else
            An(i,j)=0;
        end
        if abs(H(i,j)) > T
            Hn(i,j)=sign(H(i,j))*(abs(H(i,j))-T);
        else
            Hn(i,j)=0;
        end
        if abs(V(i,j)) > T
            Vn(i,j)=sign(V(i,j))*(abs(V(i,j))-T);
        else
            Vn(i,j)=0;
        end
        if abs(D(i,j)) > T
            Dn(i,j)=sign(D(i,j))*(abs(D(i,j))-T);
        else
            Dn(i,j)=0;
        end
    end
end
Out=idwt2(An,Hn,Vn,Dn,wave);

end