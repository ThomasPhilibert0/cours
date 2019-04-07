function derf = df_1D(x,d,t)
derf = dot(JAC(x + d*t),d);