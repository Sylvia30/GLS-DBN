x = -2:0.01:2;

L1 = abs(x);
L2 = (abs(x)).^0.1;
cachy = -1./(0.5.*(1+(x./0.5).^2));

figure;

plot(x,L1);
hold on
plot(x,L2);
hold on
plot(x,cachy+2);