I1 = imread('sea01.jpg'); I1 = rgb2hsv(I1);
A = fftshift(abs(fft2(I1)));
imshow(log(1+0.001*A),[])
I2 = imread('sea02.jpg'); I2 = rgb2hsv(I2);
figure(2);
B = fftshift(abs(fft2(I2)));
imshow(log(1+0.001*B),[]);
I3 = imread('sea03.jpg'); I3 = rgb2hsv(I3);
figure(3);
C = fftshift(abs(fft2(I3)));
imshow(log(1+0.001*C),[]);
I2 = I2(:,:,3);
s = 100;                                %plu s gran plu flou image
h = fspecial('gaussian', size(I2),s);
H = fft2(h); I = fft2(fftshift(I2));
O = H.*I;
o = real(ifft2(O));
imshow(o,[]);