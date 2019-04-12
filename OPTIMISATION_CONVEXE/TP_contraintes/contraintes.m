function C = contraintes(x)
C = max(x(1)+x(2)-2, 0)^2 + max(x(1)+2*x(2)-3, 0)^2;