clc;
%Imm=imread('D:\Desktop\Prueba Concepto\Objetos\Non\obj1.png');
%Imm=imread('D:\Desktop\Prueba Concepto\Objetos\Non\obj2.png');
Imm=imread('D:\Desktop\Prueba Concepto\Objetos\Non\obj3.png');
[X Y]=size(Imm);

formatSpec2 = '%05.f';
 
for i=1:X-23
    for j=1:Y-23
        n = num2str((i*j)+45369,formatSpec2);
        P=Imm(i:i+23,j:j+23);
        filename1=['D:\Desktop\Prueba Concepto\Objetos\OBJ',n,'.png'];
        imwrite(P,filename1);
    end
end
