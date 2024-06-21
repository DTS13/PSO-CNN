clc;
Imm=imread('D:\Desktop\ObjetosBase\Pre\obj4.png');
%Imm=imread('D:\Desktop\ObjetosBase\Pre\obj5.png');
[X Y]=size(Imm);

formatSpec2 = '%05.f';
 
for i=1:X-23
    for j=1:Y-23
        n = num2str((i*j)+45369,formatSpec2);
        P=Imm(i:i+23,j:j+23);
        filename1=['D:\Desktop\Prueba Concepto2\Objetos\OB2',n,'.png'];
        imwrite(P,filename1);
    end
end
