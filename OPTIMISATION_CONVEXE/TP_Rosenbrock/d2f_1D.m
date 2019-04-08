function ders_1D = d2f_1D(x,d,t)
ders_1D = dot(HESS(x +d*t)*d,d) ;