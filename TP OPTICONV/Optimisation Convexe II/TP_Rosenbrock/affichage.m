function out = affichage(x,d,t)
  
for i = 1:length(t);
  T(i) = banane_1D(x,d,t(i));
end
     
plot(t,T)
