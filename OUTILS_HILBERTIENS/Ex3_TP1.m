z = [1,1,0,2];
w = [i,0,1,i];
tic; conv_zw = cconv(z,w,4); fft(conv_zw); toc;
tic; Fz = fft(z); Fw = fft(w); conv_zw1 = Fz.*Fw; toc; %hashtag c plu rapide

