%function：pearson correlation coefficient(老师)
%function：spearman correlation coefficient(老师)
%function：kendall correlation coefficient(论文)
%author：郑展鹏
%2019/04/013

%%%%(c) Two channel features from Case 1
%Session1
clc;
clear;
close all;
warning off;

for pp=1:8
eval(['load F:\matlab\trial_procedure\study_1\features\ex1\s' num2str(pp) '_1'])

Y1=y(:,1);

for j=1:4
    X1=x(:,45+(j-1)*4:48+(j-1)*4);
    %coeff(j+(pp-1)*4,:)=corr(X1,Y1)';   
    %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','Spearman')'; %spearman相关系数   
    coeff_1(j+(pp-1)*4,:)=corr(X1,Y1,'type','kendall')'; %kendall相关系数 
end
end

for i=1:4
coeff_mean_1(i,:)=(coeff_1(1+(i-1),:)+coeff_1(5+(i-1),:)+coeff_1(9+(i-1),:)+coeff_1(13+(i-1),:)+coeff_1(17+(i-1),:)+coeff_1(21+(i-1),:)+coeff_1(25+(i-1),:)+coeff_1(29+(i-1),:))/8;
end


%Session2
for qq=1:8
eval(['load F:\matlab\trial_procedure\study_1\features\ex1\s' num2str(qq) '_2'])

Y2=y(:,1);

for m=1:4
    X2=x(:,45+(m-1)*4:48+(m-1)*4);
    %coeff(j+(pp-1)*4,:)=corr(X2,Y2)';  
    %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','Spearman')'; %spearman相关系数   
    coeff_2(m+(qq-1)*4,:)=corr(X2,Y2,'type','kendall')'; %kendall相关系数 
end
end

for n=1:4
coeff_mean_2(n,:)=(coeff_2(1+(n-1),:)+coeff_2(5+(n-1),:)+coeff_2(9+(n-1),:)+coeff_2(13+(n-1),:)+coeff_2(17+(n-1),:)+coeff_2(21+(n-1),:)+coeff_2(25+(n-1),:)+coeff_2(29+(n-1),:))/8;
end

coeff_mean_case1=(coeff_mean_1+coeff_mean_2)/2;

subplot(1,2,1);
% subplot(4,2,5);
bar(coeff_mean_case1,'DisplayName','coeff_mean_case1'); set(gca,'XTickLabel',{'F4-F3','C4-C3','P4-P3','O2-O1'});
title('(c) Two channel features from Case 1','FontWeight','bold');
xlabel('EEG channel pairs','FontWeight','bold');
ylabel('Kendall correlation coefficient','FontWeight','bold');
grid on;
% legend('Theta','Alpha','Beta','Gamma','Zcr','Shannon','Kurtosis','Skewness','Peak','Std','Rms','Sf','Cf','Pi');
legend('Theta','Alpha','Beta','Gamma');



%%%%(d) Two channel features from Case 2
%Session1
for ww=1:6
eval(['load F:\matlab\trial_procedure\study_1\features\ex2\s' num2str(ww) '_1'])

Y3=y(:,1);

for a=1:4
    X3=x(:,45+(a-1)*4:48+(a-1)*4);
%     coeff(j+(pp-1)*4,:)=corr(X,Y)';  
    %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','Spearman')'; %spearman相关系数   
    coeff_3(a+(ww-1)*4,:)=corr(X3,Y3,'type','kendall')'; %kendall相关系数 
end
end

for b=1:4
coeff_mean_3(b,:)=(coeff_3(1+(b-1),:)+coeff_3(5+(b-1),:)+coeff_3(9+(b-1),:)+coeff_3(13+(b-1),:)+coeff_3(17+(b-1),:)+coeff_3(21+(b-1),:))/6;
end


%Session2
for xx=1:6
eval(['load F:\matlab\trial_procedure\study_1\features\ex2\s' num2str(xx) '_2'])

Y4=y(:,1);

for c=1:4
    X4=x(:,45+(c-1)*4:48+(c-1)*4);
%     coeff(j+(pp-1)*4,:)=corr(X,Y)'; 
    %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','Spearman')'; %spearman相关系数   
    coeff_4(c+(xx-1)*4,:)=corr(X4,Y4,'type','kendall')'; %kendall相关系数 
end
end

for d=1:4
coeff_mean_4(d,:)=(coeff_4(1+(d-1),:)+coeff_4(5+(d-1),:)+coeff_4(9+(d-1),:)+coeff_4(13+(d-1),:)+coeff_4(17+(d-1),:)+coeff_4(21+(d-1),:))/6;
end

coeff_mean_case2=(coeff_mean_3+coeff_mean_4)/2;

subplot(1,2,2);
% subplot(4,2,6);
bar(coeff_mean_case2,'DisplayName','coeff_mean_case2'); set(gca,'XTickLabel',{'F4-F3','C4-C3','P4-P3','O2-O1'})
title('(d) Two channel features from Case 2','FontWeight','bold');
xlabel('EEG channel pairs','FontWeight','bold');
ylabel('Kendall correlation coefficient','FontWeight','bold');
grid on;
% legend('Theta','Alpha','Beta','Gamma','Zcr','Shannon','Kurtosis','Skewness','Peak','Std','Rms','Sf','Cf','Pi');
legend('Theta','Alpha','Beta','Gamma');