function derf = df_1D(x,d,t)
derf = dot(grad_act(x + d*t),d);