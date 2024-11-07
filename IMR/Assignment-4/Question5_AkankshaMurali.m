%Define constants
clear;
dt = 0.001; % time step
T = 120; %total time in s
N = 50; % number of dominant frequency components
F0 = 1; %starting frequency rad/s
df = 1; %frequency step
mu = 0.01; %adaptive gain parameter
%variables
omega = zeros(1, N+1); % array of frequencies
phi_mp = zeros(2, N+1); %reference input,1st row sin 2nd row cos
theta_mp = zeros(2,N+1); %weights, 1st row sin 2nd row cos
%reference signal
t = zeros(1, T/dt);
omega=F0:df:F0+N;
t=0:dt:T-dt;
s =sin(t)+2*sin(2*t)+3*sin(3*t)+0.1*sin(30*t)+0.1*sin(35*t)+cos(t)+2*cos(2*t)+3*cos(3*t)+0.1*cos(30*t)+0.1*cos(35*t);
y_mp=zeros(1,length(t));

for i=1:length(t)
phi_mp(1,:)=sin(omega*t(i));
phi_mp(2,:)=cos(omega*t(i));
    for k=1:N+1
       y_mp(i) = y_mp(i) + theta_mp(1,k)*phi_mp(1,k)+theta_mp(2,k)*phi_mp(2,k);
       err = s(i) - y_mp(i);
       theta_mp(1,k) = theta_mp(1,k) +2*mu*phi_mp(1,k)*err;
       theta_mp(2,k) = theta_mp(2,k) +2*mu*phi_mp(2,k)*err;
    end
end
Mp=s;
plot(t,Mp,'LineWidth',2,'Color','b');
hold on;
plot(t,y_mp,'LineWidth',1,'Color','r'); %estimate of total signal
legend('M\_p','Signal estimate');
N = 3; % truncated number
%variables
Mpv = sin(t)+2*sin(2*t)+3*sin(3*t)+cos(t)+2*cos(2*t)+3*cos(3*t);
s=Mpv;
omega_v=F0:df:F0+N;
phi_mpv = zeros(2, N+1); %reference input,1st row sin 2nd row cos
theta_mpv = zeros(2,N+1); %weights, 1st row sin 3nd row cos
y_mpv=zeros(1,length(t));
for i=1:length(t)
phi_mpv(1,:)=sin(omega_v*t(i));
phi_mpv(2,:)=cos(omega_v*t(i));

    for k=1:N
       y_mpv(i) = y_mpv(i) + theta_mpv(1,k)*phi_mpv(1,k)+theta_mpv(2,k)*phi_mpv(2,k);
       err = s(i) - y_mpv(i);
       theta_mpv(1,k) = theta_mpv(1,k) +2*mu*phi_mpv(1,k)*err;
       theta_mpv(2,k) = theta_mpv(2,k) +2*mu*phi_mpv(2,k)*err;
    end
end
figure;
plot(t,Mpv,'LineWidth',2,'Color','b'); %real voluntary movement signal
hold on;
plot(t,y_mpv,'LineWidth',1,'Color','r'); %voluntary movement estimate
legend('M\_pv','Voluntary signal estimate');
