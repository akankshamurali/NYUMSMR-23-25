% Bode plots - Pade approximations
figure()
hold on
delay = 0.1; %100ms
% First order approximation
[numerator1, denominator1] = pade(delay,1); % time delay of 0.2 seconds
sys1 = tf(numerator1, denominator1)
bode(sys1)


% Second order approximation
[numerator2, denominator2] = pade(delay,2);
sys2 = tf(numerator2, denominator2)
bode(sys2)

% Third order approximation
[numerator3, denominator3] = pade(delay,3);
sys3 = tf(numerator3, denominator3)
bode(sys3)

legend('1st Order Pade approximation', '2nd Order Pade approximation', '3rd Order Pade approximation')
hold off