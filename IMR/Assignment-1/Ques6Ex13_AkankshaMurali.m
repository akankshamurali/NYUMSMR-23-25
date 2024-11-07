%Exercise13
function EX13
x="";
    weight=input("Enter your weight in Pounds (lb) --> ");
    height=input("Enter your height in inches (in) --> ");
    bmi=weight.*703 /(height.^2);
    if bmi <= 18.5
        x="Underweight";
    elseif bmi > 18.5 & bmi<=24.9
        x="Normal";
    elseif bmi>=29.9
        x="Overweight";
    end
if x~="" fprintf("Your BMI is %s\n",x);
end