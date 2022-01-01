%���ƹ������ܶȵĺ���������Ƶ��500Hz,���ֵ��ӦdeltaƵ��ռȫƵ�ι��ʵı�ֵ��
function [z]=pdfplot(ics)
Fs = 500;                    % Sampling frequency
L = size(ics,2);              % Length of signal
NFFT = 2^nextpow2(L);        % Next power of 2 from length of y
Y = fft(ics,NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2+1);
a=2*abs(Y(1:NFFT/2+1));
a=smooth(a,401);     %����������ƽ��ǿ��
b=round(size(a,1)/62.5);   
z=sum(a(1:b))/sum(a(1:round(size(a,1)/6.25)));   
%%
%����һάʱ���źŵĹ������ܶ�ͼ
% figure;
% plot(f,a) 
% title('Power density spectrum')
% xlabel('Frequency (Hz)')
% ylabel('Power (db)')
% axis([0 500 0 0.2])
% axis 'auto y'