function Out=BinaryThreshold(In,TH,TL)

% TL=107; TH=120;
[X,Y]=size(In);
Im=In*0;
for i=1:X
    for j=1:Y
        if In(i,j) > TL && In(i,j) < TH
            Im(i,j)=255;
        else
            Im(i,j)=0;
        end
    end
end
Out=uint8(Im);

end