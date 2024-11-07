%Exercise12
x=input("Enter a 3x3 matrix --> ");
[r c]=size(x);

if r~=c | r~=3 | c~=3
   error("Entered data is not 3x3 or not a square matrix"); 
else
    x_out=x.^2;
    d=round(det(x_out));
    
    if d==0
        disp("Determinant of the matrix(with the elements raised to 2) is 0");
    elseif d<0
            disp("Determinant of the matrix(with the elements raised to 2) is negative");
    else
        disp("Determinant of the matrix(with the elements raised to 2) is positive");
    end
    
    
    disp("Entered Matrix is ");
    disp(x);
    fprintf('Determinant of Entered Matrix is %d\n', det(x));
    %disp(det(x));
end