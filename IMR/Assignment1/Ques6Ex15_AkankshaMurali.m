A = [5,-3,2; -3,8,4;2,4,-9];
B = [10;20;9];
disp("Solution with matrix inversion method");
Sol = inv(A)*B; % Inverse Method
disp(Sol);
disp("Solution with inbuilt function");
Sol_1 = linsolve(A,B); %Inbuilt Fuction
disp(Sol_1);
