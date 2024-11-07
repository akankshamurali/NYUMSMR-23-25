% Define the transfer functions
numerator_1 = 1;
denominator_1 = [0.2, 1];
sys_1 = tf(numerator_1, denominator_1);
figure;
bode(sys_1);
title('Bode Plot of Transfer Function 1');

numerator_2 = [1, 0.1]
denominator_2 = [0.2, 1];
sys_2 = tf(numerator_2, denominator_2);
figure;
bode(sys_2);
title('Bode Plot of Transfer Function 2');

numerator_3 = [1, -0.1]
denominator_3 = [0.2, 1];
sys_3 = tf(numerator_3, denominator_3);
figure;
bode(sys_3);
title('Bode Plot of Transfer Function 3');






