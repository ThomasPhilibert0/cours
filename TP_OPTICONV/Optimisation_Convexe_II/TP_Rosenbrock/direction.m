function dir = direction(x,rho)
dir = x - rho*JAC(x);