% 2015 Juan Gabriel Colonna, Federal University of Amazonas (UFAM - ICOMP)
% juancolonna@gmail.com

function E = FFT_entropy(x)


NFFT = length(x);

Y = fft(x,NFFT);
S = abs(Y(1:NFFT/2+1))/NFFT;
S = S./sum(S);
S=S(find(S~=0)); % elimino los valores iguales a cero para poder calcular el logaritmo
p = S/sum(S);
E = -sum(p .* log(p))/log(NFFT);
