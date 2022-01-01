%绘制功率谱密度的函数，采样频率500Hz,输出值对应delta频段占全频段功率的比值。
function [z]=pdfplot(ics)
Fs = 500;                    % Sampling frequency
L = size(ics,2);              % Length of signal
NFFT = 2^nextpow2(L);        % Next power of 2 from length of y
Y = fft(ics,NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2+1);
a=2*abs(Y(1:NFFT/2+1));
a=smooth(a,401);     %功率谱曲线平滑强度
b=round(size(a,1)/62.5);   
z=sum(a(1:b))/sum(a(1:round(size(a,1)/6.25)));   
%%
%绘制一维时间信号的功率谱密度图
% figure;
% plot(f,a) 
% title('Power density spectrum')
% xlabel('Frequency (Hz)')
% ylabel('Power (db)')
% axis([0 500 0 0.2])
% axis 'auto y'