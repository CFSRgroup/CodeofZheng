%���ƹ������ܶȵĺ���������Ƶ��500Hz,���ֵ��ӦdeltaƵ��ռȫƵ�ι��ʵı�ֵ��
function [z]=pdf(ics)
% Fs = 500;                    % Sampling frequency
L = size(ics,2);              % Length of signal
NFFT = L;        % Next power of 2 from length of y
Y = fft(ics,NFFT)/L;
% f = Fs/2*linspace(0,1,NFFT/2+1);
a=2*abs(Y(1:NFFT/2));
% a=smooth(a,7);     %����������ƽ��ǿ��
b=size(a,2)/250*40;   
z=a(1:b);
%%
%����һάʱ���źŵĹ������ܶ�ͼ
% figure;
% plot(f,a) 
% title('Power density spectrum')
% xlabel('Frequency (Hz)')
% ylabel('Power (db)')
% axis([0 40 0 0.2])
% axis 'auto y'