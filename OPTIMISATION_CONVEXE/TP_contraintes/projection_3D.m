function P = projection_3D(x)
P = x;
P(1) = x(1)/max(1,sqrt(x(1)^2+x(2)^2));
P(2) = x(2)/max(1,sqrt(x(1)^2+x(2)^2));