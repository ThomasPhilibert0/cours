function P = projection_3D(x)
P = x./max(1,sqrt(x(1)^2+x(2)^2));