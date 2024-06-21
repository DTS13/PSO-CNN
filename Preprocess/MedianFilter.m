function [med]=MedianFilter(in)

imagecolor=in;
[x,y,z]=size(imagecolor);
v=3; % Ventana 3x3, siempre deben ser ventanas con numeros impares.

for k=1:z
    for i=1:x
        output1(i,1,k)=median([imagecolor(i,1,k),imagecolor(i,2,k),imagecolor(i,3,k)]);
        for j=(v-1):1:y-(v-2)
            output1(i,j,k)=median([imagecolor(i,j-1,k),imagecolor(i,j,k),imagecolor(i,j+1,k)]);
        end
        output1(i,y,k)=median([imagecolor(i,y-2,k),imagecolor(i,y-1,k),imagecolor(i,y,k)]);
    end
    [x,y,z]=size(output1);
    for j=1:y
        output(1,j,k)=median([output1(1,j,k),output1(2,j,k),output1(3,j,k)]);
        for i=(v-1):1:x-(v-2)
            output(i,j,k)=median([output1(i-1,j,k),output1(i,j,k),output1(i+1,j,k)]);
        end
        output(x,j,k)=median([output1(x-2,j,k),output1(x-1,j,k),output1(x,j,k)]);
    end
end

med=uint8(output);