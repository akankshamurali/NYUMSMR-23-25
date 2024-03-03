% clear;
% clc;
%No of joints
%am14013
n=6;
dh=["0","1","0","-pi/2";
    "0","0","2","0";
    "-pi/2","0","0","-pi/2";
    "0","4","0","pi/2";
    "0","0","2","-pi/2";
    "0","0","3","0"];
T={};
O={};
x=[];
y=[];
z=[];
for i=1:n        
    th0=str2num(dh(i,1));
    th=input(strcat("Enter Desired Angle on Joint ",num2str(i), " -->  "));
    d=str2num(dh(i,2));
    a=str2num(dh(i,3));
    al=str2num(dh(i,4));

    T{i}=[cos(th0+th),-cos(al)*sin(th0+th),sin(al)*sin(th0+th),a*cos(th0+th);
          sin(th0+th),cos(al)*cos(th0+th),-sin(al)*cos(th0+th),a*sin(th0+th);
          0,sin(al),cos(al),d;
          0,0,0,1];

    if i==1
        O{i}=round(T{i},3);
    else
        O{i}=round(O{i-1}*T{i},3);
    end
x(i)=O{i}(1,4);
y(i)=O{i}(2,4);
z(i)=O{i}(3,4);
end



x=round([0,x],4);
y=round([0,y],4);
z=round([0,z],4);
plot3(x,y,z,"*-","MarkerSize",10,"LineWidth",1.5);
Origins=round([x;y;z]',3);
