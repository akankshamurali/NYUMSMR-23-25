cutoff_frequency = 6;
omega_1Hz = 2 * pi * 1;
omega_30Hz = 2 * pi * 30;
numerator = 1;
denominator = [[1/(2*pi*cutoff_frequency) 1]];
sys = tf(numerator, denominator);
H_1Hz = freqresp(sys, omega_1Hz);
H_30Hz = freqresp(sys, omega_30Hz);
gain_1Hz = abs(H_1Hz);
gain_30Hz = abs(H_30Hz);
disp(['Amplification gain for 1 Hz: ' num2str(gain_1Hz)]);
disp(['Amplification gain for 30 Hz: ' num2str(gain_30Hz)]);
hold on;
bode(sys);
hold off;
grid on;

