% Exercise1
% vector of length 3 with random entries
a = rand(1,3)
%3x3 matrix with random entries
b = rand(3,3)
%the inbuilt funch rand generates numbers from (0,1)
%b(randperm(9,3)) = 1

%Exercise2
m = reshape(1:16,4,4);
M = [zeros(1,5);zeros(4,1),m]

%Exercise3
det(M)

%Exercise4
I = eye(5);
C=I*M

%Exercise5
sumFifthRow = sum(M(5,:))

%Exercise6
V=M(:,end)

%Exercise7
D=M*V

%Exercise8
A = [1,2,3,4,5;2,4,6,8,10;0.5,1,1.5,2,2.5];
[Q, R] = qr(A);
A_result = Q*R
G =   [0,0,0; 1,2,0.5;2,4,1;3,6,1.5];
[Q1, R1] = qr(G);
G_result = Q1*R1

%Exercise9
t=0:0.01:50;
y1 = exp(2*t);
y2=t.*t;
figure();
hold on;
plot(t,y1,"-r");
plot(t,y2,"-b");
title('Question 9')
hold off;

%Exercise10
figure();
hold on;
plot(t,y1,"oy","MarkerSize",7);
plot(t,y2,"+g","MarkerSize",9);
title('Question 10')
hold off;

%Excercise11
r=input("Enter the Radius of circle --> ");
d=input("Enter the center of circle in form of [x y] --> ");
circle(r, d);

function [perimeter, area] = circle(radius, center)
    perimeter = round((2* pi * radius), 4);
    area = round((pi * radius * radius), 4);
    theta = linspace(0, 2*pi, 500);
    x = (radius * cos(theta)) + center(1);
    y = (radius * sin(theta)) + center(2);
    figure;
    plot(x,y);
    axis equal;
    title("Circle with radius " + radius);
end



