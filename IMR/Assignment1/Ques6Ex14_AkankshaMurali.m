%Exercise14
t=0:0.1:10;
a=sin(t);
b=cos(t);

plot(t,a,"-r");
hold on
plot(t,b,'--b');
grid on;
set(gca,'GridLineStyle','--');
xlabel("time");
ylabel("sin & cos");
title("Sin and Cos");
legend('sin(t)','cos(t)');
text(5,0.7,"cos(t)");
text(6,-0.3,"sin(t)");
hold off;
