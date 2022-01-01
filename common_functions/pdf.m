%绘制功率谱密度的函数，采样频率500Hz,输出值对应delta频段占全频段功率的比值。
function [z]=pdf(ics)
% Fs = 500;                    % Sampling frequency
L = size(ics,2);              % Length of signal
NFFT = L;        % Next power of 2 from length of y
Y = fft(ics,NFFT)/L;
% f = Fs/2*linspace(0,1,NFFT/2+1);
a=2*abs(Y(1:NFFT/2));
% a=smooth(a,7);     %功率谱曲线平滑强度
b=size(a,2)/250*40;   
z=a(1:b);
%%
%绘制一维时间信号的功率谱密度图
% figure;
% plot(f,a) 
% title('Power density spectrum')
% xlabel('Frequency (Hz)')
% ylabel('Power (db)')
% axis([0 40 0 0.2])
% axis 'auto y'