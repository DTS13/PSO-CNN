clc;
Imm=imread('D:\Desktop\ObjetosBase\Non\bg1.png');
[X Y]=size(Imm);

formatSpec2 = '%05.f';
 
for i=1:20000
        m=randi([1 (X-23)],1);
        k=randi([1 (Y-23)],1);
        n = num2str(i,formatSpec2);
        P=Imm(m:m+23,k:k+23);
        filename1=['D:\Desktop\Prueba Concepto\Background\BG',n,'.png'];
        imwrite(P,filename1);
end
