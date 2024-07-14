clear clc
syms C m xdot lf lr Iz;

% Exercise 1.1

% For lateral control
A = [0 1 0 0;
0 -4*C/(m*xdot) 4*C/m -2*C*(lf-lr)/(m*xdot);
0 0 0 1;
0 -2*C*(lf-lr)/(Iz*xdot) 2*C*(lf-lr)/Iz -2*C*(lf^2+lr^2)/(Iz*xdot)];

B = [0 0;
2*C/m 0;
0 0;
2*C*lf/Iz 0];

A = subs(A,[C,m,Iz,lf,lr],[20000,1888.6,25854,1.55,1.39]);
B = subs(B,[C,m,Iz,lf,lr],[20000,1888.6,25854,1.55,1.39]);
C = eye(4);
D = 0;

P = simplify([B A*B A*A*B A*A*A*B]);
Psubs = subs(P,xdot,5);

Q = simplify([C; C*A; C*A*A; C*A*A*A]);
Qsubs = subs(Q,xdot,5);

% For longitudinal control
A2 = [0 1;0 0];
B2 = [0 0;0 1/1888.6];
C2 = eye(2);
D2 = 0;

P2 = [B2 A2*B2];
Q2 = [C2;C2*A2];

% Exercise 1.2
velocities = [1:40];
singVals = zeros(size(velocities));

for i=1:size(velocities)
    p = subs(P,xdot,i);
    S = svd(p);
    singVals(i) = log10(S(1)/S(4));
end

plot(velocities,singVals)
xlabel('Velocities')
ylabel('log10(sigma_1/sigma_n)')
title('log10(sigma_1/sigma_n) vs Velocities')

pole1 = zeros(size(velocities));
pole2 = zeros(size(velocities));
pole3 = zeros(size(velocities));
pole4 = zeros(size(velocities));

for i=1:size(velocities)
    a = double(subs(A,xdot,i));
    b = double(B);
    sys = ss(a,b,C,D);
    Pole = pole(sys);
    pole1(i) = Pole(1);
    pole2(i) = Pole(2);
    pole3(i) = Pole(3);
    pole4(i) = Pole(4);
end

figure
subplot(2,2,1)
plot(velocities,pole1)
xlabel('Velocities')
ylabel('Pole-1')
title('Pole-1 vs Velocities')

subplot(2,2,2)
plot(velocities,pole2)
xlabel('Velocities')
ylabel('Pole-2')
title('Pole-2 vs Velocities')

subplot(2,2,3)
plot(velocities,pole3)
xlabel('Velocities')
ylabel('Pole-3')
title('Pole-3 vs Velocities')

subplot(2,2,4)
plot(velocities,pole4)
xlabel('Velocities')
ylabel('Pole-4')
title('Pole-4 vs Velocities')