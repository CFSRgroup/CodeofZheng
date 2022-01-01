%function：pearson correlation coefficient(老师)
%function：spearman correlation coefficient(老师)
%function：kendall correlation coefficient(论文)
%author：郑展鹏
%2019/04/013

%%%%(a) Single channel features from Case 1
%Session1
clc;
clear;
close all;
warning off;

for pp=1:8
eval(['load F:\matlab\trial_procedure\study_1\features\ex1\s' num2str(pp) '_1'])

Y1=y(:,1);

for j=1:11
%     X1=x(:,[1+(j-1)*4:4+(j-1)*4 61+(j-1)*10:70+(j-1)*10]);
      X1=x(:,[1+(j-1)*4:4+(j-1)*4 61+(j-1)*1 72+(j-1)*1 83+(j-1)*1 94+(j-1)*1 105+(j-1)*1 116+(j-1)*1 127+(j-1)*1 138+(j-1)*1 149+(j-1)*1 160+(j-1)*1]);
    %coeff(j+(pp-1)*11,:)=corr(X1,Y1)'; %pearson相关系数    
    %coeff(j+(pp-1)*11,:)=corr(X1,Y1,'type','Spearman')'; %spearman相关系数   
    coeff_1(j+(pp-1)*11,:)=corr(X1,Y1,'type','kendall')'; %kendall相关系数  
end
end

for i=1:11
coeff_mean_1(i,:)=(coeff_1(1+(i-1),:)+coeff_1(12+(i-1),:)+coeff_1(23+(i-1),:)+coeff_1(34+(i-1),:)+coeff_1(45+(i-1),:)+coeff_1(56+(i-1),:)+coeff_1(67+(i-1),:)+coeff_1(78+(i-1),:))/8;
end

%Session2
for qq=1:8
eval(['load F:\matlab\trial_procedure\study_1\features\ex1\s' num2str(qq) '_2'])

Y2=y(:,1);

for m=1:11
%     X2=x(:,[1+(m-1)*4:4+(m-1)*4 61+(m-1)*10:70+(m-1)*10]);
      X2=x(:,[1+(m-1)*4:4+(m-1)*4 61+(m-1)*1 72+(m-1)*1 83+(m-1)*1 94+(m-1)*1 105+(m-1)*1 116+(m-1)*1 127+(m-1)*1 138+(m-1)*1 149+(m-1)*1 160+(m-1)*1]);
%     coeff(j+(pp-1)*11,:)=corr(X,Y)'; %pearson相关系数    
    %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','Spearman')'; %spearman相关系数   
    coeff_2(m+(qq-1)*11,:)=corr(X2,Y2,'type','kendall')'; %kendall相关系数  
end
end

for n=1:11
coeff_mean_2(n,:)=(coeff_2(1+(n-1),:)+coeff_2(12+(n-1),:)+coeff_2(23+(n-1),:)+coeff_2(34+(n-1),:)+coeff_2(45+(n-1),:)+coeff_2(56+(n-1),:)+coeff_2(67+(n-1),:)+coeff_2(78+(n-1),:))/8;
end

coeff_mean_case1=(coeff_mean_1+coeff_mean_2)/2;

subplot(2,1,1);
bar(coeff_mean_case1,'DisplayName','coeff_mean_case1'); set(gca,'XTickLabel',{'F3','F4','Fz','C3','C4','Cz','P3','P4','Pz','O1','O2'})
title('(a) Single channel features from Case 1','FontWeight','bold');
xlabel('EEG channels','FontWeight','bold');
ylabel('Kendall correlation coefficient','FontWeight','bold');
grid on;
legend('Theta','Alpha','Beta','Gamma','Zcr','Shannon','Kurtosis','Skewness','Peak','Std','Rms','Sf','Cf','Pi');


%%%%(b) Single channel features from Case 2
%Session1
for ww=1:6
eval(['load F:\matlab\trial_procedure\study_1\features\ex2\s' num2str(ww) '_1'])

Y3=y(:,1);

for a=1:11
%     X3=x(:,[1+(a-1)*4:4+(a-1)*4 61+(a-1)*10:70+(a-1)*10]);
      X3=x(:,[1+(a-1)*4:4+(a-1)*4 61+(a-1)*1 72+(a-1)*1 83+(a-1)*1 94+(a-1)*1 105+(a-1)*1 116+(a-1)*1 127+(a-1)*1 138+(a-1)*1 149+(a-1)*1 160+(a-1)*1]);
    %coeff(a+(pp-1)*11,:)=corr(X,Y)'; 
    %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','Spearman')'; %spearman相关系数   
    coeff_3(a+(ww-1)*11,:)=corr(X3,Y3,'type','kendall')'; %kendall相关系数 
end
end

for b=1:11
coeff_mean_3(b,:)=(coeff_3(1+(b-1),:)+coeff_3(12+(b-1),:)+coeff_3(23+(b-1),:)+coeff_3(34+(b-1),:)+coeff_3(45+(b-1),:)+coeff_3(56+(b-1),:))/6;
end

%session2
for xx=1:6
eval(['load F:\matlab\trial_procedure\study_1\features\ex2\s' num2str(xx) '_2'])

Y4=y(:,1);

for c=1:11
%     X4=x(:,[1+(c-1)*4:4+(c-1)*4 61+(c-1)*10:70+(c-1)*10]);
      X4=x(:,[1+(c-1)*4:4+(c-1)*4 61+(c-1)*1 72+(c-1)*1 83+(c-1)*1 94+(c-1)*1 105+(c-1)*1 116+(c-1)*1 127+(c-1)*1 138+(c-1)*1 149+(c-1)*1 160+(c-1)*1]);
    %coeff_3(c+(xx-1)*11,:)=corr(X4,Y4)';    
    %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','Spearman')'; %spearman相关系数   
    coeff_4(c+(xx-1)*11,:)=corr(X4,Y4,'type','kendall')'; %kendall相关系数 
end
end

for d=1:11
coeff_mean_4(d,:)=(coeff_4(1+(d-1),:)+coeff_4(12+(d-1),:)+coeff_4(23+(d-1),:)+coeff_4(34+(d-1),:)+coeff_4(45+(d-1),:)+coeff_4(56+(d-1),:))/6;
end

coeff_mean_case2=(coeff_mean_3+coeff_mean_4)/2;

subplot(2,1,2);
bar(coeff_mean_case2,'DisplayName','coeff_mean_case2'); set(gca,'XTickLabel',{'F3','F4','Fz','C3','C4','Cz','P3','P4','Pz','O1','O2'})
title('(b) Single channel features from Case 2','FontWeight','bold');
xlabel('EEG channels','FontWeight','bold');
ylabel('Kendall correlation coefficient','FontWeight','bold');
grid on;
legend('Theta','Alpha','Beta','Gamma','Zcr','Shannon','Kurtosis','Skewness','Peak','Std','Rms','Sf','Cf','Pi');
