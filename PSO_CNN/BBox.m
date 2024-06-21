x1 = [0 1 2];
y1 = [0 1 0];
x2 = [2 3 4];
y2 = [1 2 1];
polyin = polyshape({x1,x2},{y1,y2});
[xlim,ylim] = boundingbox(polyin);
plot(polyin)
hold on
plot(xlim,ylim,'r*',xlim,fliplr(ylim),'r*')
plot(points2(:,1),points2(:,2), '*-');
bbox = [0,0,4,2];
%bbox = [10,20,50,60];
points = bbox2points(bbox);
rectangle('Position',[0,0,4,2],'EdgeColor','b','LineWidth',3);