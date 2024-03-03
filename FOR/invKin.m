
function Theta=invKin(T06_matrix)

Ttarget=T06_matrix;

% clear;
load('Workspace_Mapping_DataBase.mat');
tic;
syms theta d a al q1 q2 q3 q4 q5 q6

if size(Ttarget)~=[4,4]
    disp("Wrong Inputs");
    Theta=NaN;
    return;
end

px=Ttarget(1,4); 
py=Ttarget(2,4);
pz=Ttarget (3,4);

if (px^2+py^2+(pz-1)^2)>5^2
    display('Out of workspace');
    Theta=NaN;
    return;
    endS
tmp=O1-[px,py,pz];
tmp=tmp<[0.1,0.1,0.1];
if (ismember([1,1,1],tmp,'rows')~=1)
    display('Out of workspace');
    Theta=NaN;
    return;
end


  i_1_T_i = matlabFunction([cos(theta), round(-cos(al), 3)*sin(theta), round(sin(al),3)*sin(theta), a*cos(theta); sin(theta), round(cos(al),3)*cos(theta), -round(sin(al),3)*cos(theta), a * sin(theta); 0, round(sin(al),3), round(cos(al),3), d; 0,0,0,1], 'Vars',[theta d a al]);

T01 = i_1_T_i( q1, 1, 0,pi/2);
T12 = i_1_T_i(((pi/2)+q2), 0, 1, 0);
T23 = i_1_T_i(q3, 0, 0, -pi/2);
T34 = i_1_T_i(q4, 2, 0, pi/2);
T45 = i_1_T_i(q5, 0, 1, -pi/2);
T56 = i_1_T_i(q6, 0, 1, 0);


T06 = T01 * T12 * T23 * T34 * T45 * T56;
T06=simplify(T06);

Ttarget = round(Ttarget, 3);
Theta = vpasolve([T06(1,4) == Ttarget(1,4), T06(2,4) == Ttarget(2,4), T06(3,4) == Ttarget(3,4), T06(1,1) == Ttarget(1,1), T06(2,2) == Ttarget(2,2), T06(3,3) == Ttarget(3,3)], [q1 q2 q3 q4 q5 q6]);

if class(Theta.q1)=='sym'
    disp("No Solutions Exist")
end

disp(num2str(toc));
end