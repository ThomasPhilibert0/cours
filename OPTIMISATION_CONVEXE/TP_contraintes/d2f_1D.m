function ders_1D = d2f_1D(x,d,t)
ders_1D = dot(Hess_act(x +d*t)*d,d) ;