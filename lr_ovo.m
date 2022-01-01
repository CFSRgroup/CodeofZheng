%作者：郑展鹏
%lr_ovo 三分类
%lr跨被试（任务一8位）
%20190419
% clc;
% clear;
% close all;
% warning off;
% 
% % load F:\matlab\trial_procedure\study_1\features\ex1\s1_1
% % x1=x;
% % load F:\matlab\trial_procedure\study_1\features\ex1\s2_1
% % x2=x;
% % load F:\matlab\trial_procedure\study_1\features\ex1\s3_1
% % x3=x;
% % load F:\matlab\trial_procedure\study_1\features\ex1\s4_1
% % x4=x;
% % load F:\matlab\trial_procedure\study_1\features\ex1\s5_1
% % x5=x;
% % load F:\matlab\trial_procedure\study_1\features\ex1\s6_1
% % x6=x;
% % load F:\matlab\trial_procedure\study_1\features\ex1\s7_1
% % x7=x;
% % load F:\matlab\trial_procedure\study_1\features\ex1\s8_1
% % x8=x;
% 
% load F:\matlab\trial_procedure\study_1\features\ex1\s1_2
% x1=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s2_2
% x2=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s3_2
% x3=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s4_2
% x4=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s5_2
% x5=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s6_2
% x6=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s7_2
% x7=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s8_2
% x8=x;
% 
% x=[x1;x2;x3;x4;x5;x6;x7];
% y1=y;
% y=[y;y;y;y;y;y;y];
% 
% for i=1:7
% y_low(1+(i-1)*900:900+(i-1)*900,:)=y([1+(i-1)*2700:450+(i-1)*2700 2251+(i-1)*2700:2700+(i-1)*2700]);
% y_medium(1+(i-1)*900:900+(i-1)*900,:)=y([451+(i-1)*2700:900+(i-1)*2700 1801+(i-1)*2700:2250+(i-1)*2700]);
% y_high(1+(i-1)*900:900+(i-1)*900,:)=y([901+(i-1)*2700:1350+(i-1)*2700 1351+(i-1)*2700:1800+(i-1)*2700]);
% end
% 
% for i=1:7
% x_low(1+(i-1)*900:900+(i-1)*900,:)=x([1+(i-1)*2700:450+(i-1)*2700 2251+(i-1)*2700:2700+(i-1)*2700],:);
% x_medium(1+(i-1)*900:900+(i-1)*900,:)=x([451+(i-1)*2700:900+(i-1)*2700 1801+(i-1)*2700:2250+(i-1)*2700],:);
% x_high(1+(i-1)*900:900+(i-1)*900,:)=x([901+(i-1)*2700:1350+(i-1)*2700 1351+(i-1)*2700:1800+(i-1)*2700],:);
% end
% 
% %8位被试
% %classifier_1，前者“+”，后者“-”
% xtr_1=[x_low;x_medium];
% ytr_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];
% model_1=glmfit(xtr_1,[ytr_1 ones(12600,1)],'binomial','link','logit');
% 
% %classifier_2，前者“+”，后者“-”
% xtr_2=[x_low;x_high];
% ytr_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];
% model_2=glmfit(xtr_2,[ytr_2 ones(12600,1)],'binomial','link','logit');
% 
% %classifier_3，前者“+”，后者“-”
% xtr_3=[x_medium;x_high];
% ytr_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
% model_3=glmfit(xtr_3,[ytr_3 ones(12600,1)],'binomial','link','logit');
% 
% xte=x8;
% yte=y1;
% 
% yte_p_1=round(glmval(model_1,xte,'logit'));
% cte_1=cfmatrix(yte,yte_p_1);%计算混淆矩阵
% 
% yte_p_2=round(glmval(model_2,xte,'logit'));
% cte_2=cfmatrix(yte,yte_p_2);%计算混淆矩阵
% 
% yte_p_3=round(glmval(model_3,xte,'logit'));
% cte_3=cfmatrix(yte,yte_p_3);%计算混淆矩阵
% 
% yte_p_4=[yte_p_1 yte_p_2 yte_p_3];
% 
% for j=1:2700
% yte_p(j,:)=ovo_code(yte_p_1(1+(j-1)*1,:),yte_p_2(1+(j-1)*1,:),yte_p_3(1+(j-1)*1,:));
% end
% 
% %查看三类数目
% c1=size(find(yte_p==1));
% c2=size(find(yte_p==2));
% c3=size(find(yte_p==3));
% 
% %目标类别确定之后的混淆矩阵
% cte=cfmatrix(yte,yte_p);%计算混淆矩阵
% [acc_low,acc_medium,acc_high,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cte); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% 
% per_index=[acc_low;acc_medium;acc_high;mean_acc;sen;spe;pre;npv;f1;fnr;fpr;fdr;foe;mcc;bm;mk;ka];



% model=glmfit(xtr,ytr,'binomial','link','logit');
% yte_p=round(glmval(model,xte,'logit'));
% 
% cte=cfmatrix(yte,yte_p);%计算混淆矩阵
% 
% [acc_low,acc_medium,acc_high,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cte); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% 
% per_index=[acc_low;acc_medium;acc_high;mean_acc;sen;spe;pre;npv;f1;fnr;fpr;fdr;foe;mcc;bm;mk;ka];
% % time=[TestingAccuracy;TestingTime;TrainingAccuracy;TrainingTime];
% 
% % save F:\matlab\trial_procedure\study_1\baseline_performance\lr_cross_subject\task1_session1_H per_index 
% save F:\matlab\trial_procedure\study_1\baseline_performance\lr_cross_subject\task1_session2_H per_index 

% plot(yte);
% title('(a) Subject A, session 1, ACC=0.');
% axis on;  
% %设置坐标轴开启
% set(gca,'xtick',0:450:2700);   
% %gca是当前坐标轴的句柄，xtick表示我要设置x轴刻度要显示的位置
% % set(gca,'xticklabel',{0.1,0.2,0.3,0.4,0.5});  
% %xticklabel表示设置刻度上显示的东西，后面为希望显示的实际值
% set(gca,'ytick',1:1:3);   
% %这个地方注意y轴是从上往下数的，0在最上面
% set(gca,'yticklabel',{'LMW','MMW','HMW'});
% xlabel('Time index (2 s)');
% ylabel('OFS');
% hold on;
% plot(yte_p);
% legend('Target','Predicted');



%lr跨被试（任务二6位）
% clc;
% clear;
% close all;
% warning off;
% 
% % load F:\matlab\trial_procedure\study_1\features\ex2\s1_1
% % x1=x;
% % load F:\matlab\trial_procedure\study_1\features\ex2\s2_1
% % x2=x;
% % load F:\matlab\trial_procedure\study_1\features\ex2\s3_1
% % x3=x;
% % load F:\matlab\trial_procedure\study_1\features\ex2\s4_1
% % x4=x;
% % load F:\matlab\trial_procedure\study_1\features\ex2\s5_1
% % x5=x;
% % load F:\matlab\trial_procedure\study_1\features\ex2\s6_1
% % x6=x;
% 
% load F:\matlab\trial_procedure\study_1\features\ex2\s1_2
% x1=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s2_2
% x2=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s3_2
% x3=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s4_2
% x4=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s5_2
% x5=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s6_2
% x6=x;
% 
% x=[x1;x2;x3;x4;x5];
% y1=y;
% y=[y;y;y;y;y];
% 
% for i=1:5
% y_low(1+(i-1)*580:580+(i-1)*580,:)=y([1+(i-1)*1450:145+(i-1)*1450 436+(i-1)*1450:580+(i-1)*1450 871+(i-1)*1450:1015+(i-1)*1450 1306+(i-1)*1450:1450+(i-1)*1450]);
% y_medium(1+(i-1)*435:435+(i-1)*435,:)=y([146+(i-1)*1450:290+(i-1)*1450 581+(i-1)*1450:725+(i-1)*1450 1016+(i-1)*1450:1160+(i-1)*1450 ]);
% y_high(1+(i-1)*435:435+(i-1)*435,:)=y([291+(i-1)*1450:435+(i-1)*1450 726+(i-1)*1450:870+(i-1)*1450 1161+(i-1)*1450:1305+(i-1)*1450]);
% end
% 
% for i=1:5
% x_low(1+(i-1)*580:580+(i-1)*580,:)=x([1+(i-1)*1450:145+(i-1)*1450 436+(i-1)*1450:580+(i-1)*1450 871+(i-1)*1450:1015+(i-1)*1450 1306+(i-1)*1450:1450+(i-1)*1450],:);
% x_medium(1+(i-1)*435:435+(i-1)*435,:)=x([146+(i-1)*1450:290+(i-1)*1450 581+(i-1)*1450:725+(i-1)*1450 1016+(i-1)*1450:1160+(i-1)*1450],:);
% x_high(1+(i-1)*435:435+(i-1)*435,:)=x([291+(i-1)*1450:435+(i-1)*1450 726+(i-1)*1450:870+(i-1)*1450 1161+(i-1)*1450:1305+(i-1)*1450],:);
% end
% 
% %6位被试
% %classifier_1，前者“+”，后者“-”
% xtr_1=[x_low;x_medium];
% ytr_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];
% model_1=glmfit(xtr_1,[ytr_1 ones(5075,1)],'binomial','link','logit');
% 
% %classifier_2，前者“+”，后者“-”
% xtr_2=[x_low;x_high];
% ytr_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];
% model_2=glmfit(xtr_2,[ytr_2 ones(5075,1)],'binomial','link','logit');
% 
% %classifier_3，前者“+”，后者“-”
% xtr_3=[x_medium;x_high];
% ytr_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
% model_3=glmfit(xtr_3,[ytr_3 ones(4350,1)],'binomial','link','logit');
% 
% xte=x6;
% yte=y1;
% 
% yte_p_1=round(glmval(model_1,xte,'logit'));
% cte_1=cfmatrix(yte,yte_p_1);%计算混淆矩阵
% 
% yte_p_2=round(glmval(model_2,xte,'logit'));
% cte_2=cfmatrix(yte,yte_p_2);%计算混淆矩阵
% 
% yte_p_3=round(glmval(model_3,xte,'logit'));
% cte_3=cfmatrix(yte,yte_p_3);%计算混淆矩阵
% 
% yte_p_4=[yte_p_1 yte_p_2 yte_p_3];
% 
% for j=1:1450
% yte_p(j,:)=ovo_code(yte_p_1(1+(j-1)*1,:),yte_p_2(1+(j-1)*1,:),yte_p_3(1+(j-1)*1,:));
% end
% 
% %查看三类数目
% c1=size(find(yte_p==1));
% c2=size(find(yte_p==2));
% c3=size(find(yte_p==3));
% 
% %目标类别确定之后的混淆矩阵
% cte=cfmatrix(yte,yte_p);%计算混淆矩阵
% [acc_low,acc_medium,acc_high,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cte); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% 
% per_index=[acc_low;acc_medium;acc_high;mean_acc;sen;spe;pre;npv;f1;fnr;fpr;fdr;foe;mcc;bm;mk;ka];
% 
% % save F:\matlab\trial_procedure\study_1\baseline_performance\lr_cross_subject\task2_session1_N per_index 
% save F:\matlab\trial_procedure\study_1\baseline_performance\lr_cross_subject\task2_session2_N per_index 

% plot(yte);
% title('(k) Subject N, session 1, ACC=0.');
% axis on;  
% %设置坐标轴开启
% set(gca,'xtick',0:145:1450);   
% %xticklabel表示设置刻度上显示的东西，后面为希望显示的实际值
% set(gca,'ytick',1:1:3);   
% %这个地方注意y轴是从上往下数的，0在最上面
% set(gca,'yticklabel',{'LMW','MMW','HMW'});
% xlabel('Time index (2 s)');
% ylabel('OFS');
% hold on;



%%%%%%%
%程序改进
%lr_ovo跨被试（任务一8位）
clc;
clear;
close all;
warning off;

load F:\matlab\trial_procedure\study_1\features\ex1\s1_1
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_1
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_1
x3=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s4_1
x4=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s5_1
x5=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s6_1
x6=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s7_1
x7=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s8_1
x8=x;

x11=[x1;x2;x3;x4;x5;x6;x7;x8];

rr=70;
%进行LPP
fea =x11;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x11=fea*eigvector;

load F:\matlab\trial_procedure\study_1\features\ex1\s1_2
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_2
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_2
x3=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s4_2
x4=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s5_2
x5=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s6_2
x6=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s7_2
x7=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s8_2
x8=x;

x22=[x1;x2;x3;x4;x5;x6;x7;x8];

%进行LPP
fea =x22;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x22=fea*eigvector;

%任务一两个阶段的数据集
x1=x11;
x2=x22;

y=[y;y;y;y;y;y;y;y];

K=8;
low=size(find(y==1));
medium=size(find(y==2));
high=size(find(y==3));

[xtr_all_1, xte_all_1] = kfcv(x1,K,'off');
[ytr_all_1, yte_all_1] = kfcv(y,K,'off');

[xtr_all_2, xte_all_2] = kfcv(x2,K,'off');
[ytr_all_2, yte_all_2] = kfcv(y,K,'off');

%%%%%session1
for j=1:50
for i=1:K
    
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    
    for z=1:7
    y_low(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700]);
    y_medium(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700]);
    y_high(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700]);
    end

   for z=1:7
    x_low(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700],:);
    x_medium(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700],:);
    x_high(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700],:);
   end
    
%前者“+”，后者“-”
x_1=[x_low;x_medium];
y_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];

x_2=[x_low;x_high];
y_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];

x_3=[x_medium;x_high];
y_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
    
   rand('state',0)  %resets the generator to its initial state
start_time_train=cputime;   
model_1=glmfit(x_1,[y_1 ones(12600,1)],'binomial','link','logit');
model_2=glmfit(x_2,[y_2 ones(12600,1)],'binomial','link','logit');
model_3=glmfit(x_3,[y_3 ones(12600,1)],'binomial','link','logit');
end_time_train=cputime;
TrainingTime(i,j)=end_time_train-start_time_train;

start_time_test=cputime;
yte_p_1=round(glmval(model_1,xte_1,'logit'));
yte_p_2=round(glmval(model_2,xte_1,'logit'));
yte_p_3=round(glmval(model_3,xte_1,'logit'));
end_time_test=cputime;
TestingTime(i,j)=end_time_test-start_time_test; 

%训练精度
predict_train_1=round(glmval(model_1,x_1,'logit'));
TrainingAccuracy_1(i,j)=length(find(predict_train_1==y_1))/size(y_1,1);

%训练精度
predict_train_2=round(glmval(model_2,x_2,'logit'));
TrainingAccuracy_2(i,j)=length(find(predict_train_2==y_2))/size(y_2,1);

%训练精度
predict_train_3=round(glmval(model_3,x_3,'logit'));
TrainingAccuracy_3(i,j)=length(find(predict_train_3==y_3))/size(y_3,1);

TrainingAccuracy(i,j)=(TrainingAccuracy_1(i,j)+TrainingAccuracy_2(i,j)+TrainingAccuracy_3(i,j))/3;

for z=1:2700
yte_p(z,:)=ovo_code(yte_p_1(1+(z-1)*1,:),yte_p_2(1+(z-1)*1,:),yte_p_3(1+(z-1)*1,:));
end

% yte_p_perdict(1+(i-1)*2700:2700+(i-1)*2700,1)=yte_p;
% yte_p_label=yte_p_perdict;
% save F:\matlab\trial_procedure\study_1\roc_data\lr_cross_subject\task1_session1 yte_p_label y
%查看三类数目
c1=size(find(yte_p==1));
c2=size(find(yte_p==2));
c3=size(find(yte_p==3));

 %目标类别确定之后的混淆矩阵
cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵

[acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
mean_acc(i,j)=(acc_low(i,j)+acc_medium(i,j)+acc_high(i,j))/3;

TestingAccuracy(i,j)=Acc(i,j);
% [para_ind]=find(mean_acc==max(mean_acc));%找出最优参数

acc_low_mean=mean((acc_low'))';
acc_medium_mean=mean((acc_medium'))';
acc_high_mean=mean((acc_high'))';
Acc_mean=mean((Acc'))';  %对角线值除以总值的测试精度
acc_mean=mean((mean_acc'))';
sen_mean=mean((sen'))';
spe_mean=mean((spe'))';
pre_mean=mean((pre'))';
npv_mean=mean((npv'))';
f1_mean=mean((f1'))';
fnr_mean=mean((fnr'))';
fpr_mean=mean((fpr'))';
fdr_mean=mean((fdr'))';
foe_mean=mean((foe'))';
mcc_mean=mean((mcc'))';
bm_mean=mean((bm'))';
mk_mean=mean((mk'))';
ka_mean=mean((ka'))';

acc_low_sd=std((acc_low'))';
acc_medium_sd=std((acc_medium'))';
acc_high_sd=std((acc_high'))';
Acc_sd=std((Acc'))';  %对角线值除以总值的测试精度
acc_sd=std((mean_acc'))';
sen_sd=std((sen'))';
spe_sd=std((spe'))';
pre_sd=std((pre'))';
npv_sd=std((npv'))';
f1_sd=std((f1'))';
fnr_sd=std((fnr'))';
fpr_sd=std((fpr'))';
fdr_sd=std((fdr'))';
foe_sd=std((foe'))';
mcc_sd=std((mcc'))';
bm_sd=std((bm'))';
mk_sd=std((mk'))';
ka_sd=std((ka'))';

acc_TrainingAccuracy=mean((TrainingAccuracy'))';
acc_TestingAccuracy=mean((TestingAccuracy'))';

plot_TrainingAccuracy=mean(TrainingAccuracy);
plot_TestingAccuracy=mean(TestingAccuracy);

par_ave_mean=mean(mean(TrainingTime'+TestingTime')');
par_ave_sd=mean(std(TrainingTime'+TestingTime')');

accuracy_par_ave_mean=mean(mean(TestingAccuracy')');
accuracy_par_ave_sd=mean(std(TestingAccuracy')');
subject_average_accuracy1=[accuracy_par_ave_mean accuracy_par_ave_sd]; %重复50次实验

per_index_single1=[acc_low_mean,acc_medium_mean,acc_high_mean,Acc_mean,acc_mean,sen_mean,spe_mean,pre_mean,npv_mean,f1_mean,fnr_mean,fpr_mean,fdr_mean,foe_mean,mcc_mean,bm_mean,mk_mean,ka_mean];

per_index_ave1_mean=mean([acc_low_mean,acc_medium_mean,acc_high_mean,Acc_mean,acc_mean,sen_mean,spe_mean,pre_mean,npv_mean,f1_mean,fnr_mean,fpr_mean,fdr_mean,foe_mean,mcc_mean,bm_mean,mk_mean,ka_mean])';
per_index_ave1_sd=mean([acc_low_sd,acc_medium_sd,acc_high_sd,Acc_sd,acc_sd,sen_sd,spe_sd,pre_sd,npv_sd,f1_sd,fnr_sd,fpr_sd,fdr_sd,foe_sd,mcc_sd,bm_sd,mk_sd,ka_sd])';
per_index_ave1=[per_index_ave1_mean per_index_ave1_sd];

accuracy1=[acc_TrainingAccuracy acc_TestingAccuracy];
time1=[par_ave_mean par_ave_sd];

% save F:\matlab\trial_procedure\study_1\performance_comparison\lr_ovo_cross_subject\task1_session1 per_index accuracy time
% save F:\matlab\trial_procedure\study_1\data_analysis\lr_cross_subject\task1_session1 per_index_single1 per_index_ave1 accuracy1 time1 subject_average_accuracy1
save F:\matlab\trial_procedure\study_1\data_analysis\lpp_lr_cross_subject\task1_session1 per_index_single1 per_index_ave1 accuracy1 time1 subject_average_accuracy1

end
end

%%%%%session2
for j=1:50
for i=1:K
    
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    
    for z=1:7
    y_low(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700]);
    y_medium(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700]);
    y_high(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700]);
    end

   for z=1:7
    x_low(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700],:);
    x_medium(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700],:);
    x_high(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700],:);
   end
    
%前者“+”，后者“-”
x_1=[x_low;x_medium];
y_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];

x_2=[x_low;x_high];
y_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];

x_3=[x_medium;x_high];
y_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
    
   rand('state',0)  %resets the generator to its initial state
start_time_train=cputime;   
model_1=glmfit(x_1,[y_1 ones(12600,1)],'binomial','link','logit');
model_2=glmfit(x_2,[y_2 ones(12600,1)],'binomial','link','logit');
model_3=glmfit(x_3,[y_3 ones(12600,1)],'binomial','link','logit');
end_time_train=cputime;
TrainingTime(i,j)=end_time_train-start_time_train;

start_time_test=cputime;
yte_p_1=round(glmval(model_1,xte_2,'logit'));
yte_p_2=round(glmval(model_2,xte_2,'logit'));
yte_p_3=round(glmval(model_3,xte_2,'logit'));
end_time_test=cputime;
TestingTime(i,j)=end_time_test-start_time_test; 

%训练精度
predict_train_1=round(glmval(model_1,x_1,'logit'));
TrainingAccuracy_1(i,j)=length(find(predict_train_1==y_1))/size(y_1,1);

%训练精度
predict_train_2=round(glmval(model_2,x_2,'logit'));
TrainingAccuracy_2(i,j)=length(find(predict_train_2==y_2))/size(y_2,1);

%训练精度
predict_train_3=round(glmval(model_3,x_3,'logit'));
TrainingAccuracy_3(i,j)=length(find(predict_train_3==y_3))/size(y_3,1);

TrainingAccuracy(i,j)=(TrainingAccuracy_1(i,j)+TrainingAccuracy_2(i,j)+TrainingAccuracy_3(i,j))/3;

for z=1:2700
yte_p(z,:)=ovo_code(yte_p_1(1+(z-1)*1,:),yte_p_2(1+(z-1)*1,:),yte_p_3(1+(z-1)*1,:));
end

% yte_p_perdict(1+(i-1)*2700:2700+(i-1)*2700,1)=yte_p;
% yte_p_label=yte_p_perdict;
% save F:\matlab\trial_procedure\study_1\roc_data\lr_cross_subject\task1_session2 yte_p_label y
%查看三类数目
c1=size(find(yte_p==1));
c2=size(find(yte_p==2));
c3=size(find(yte_p==3));

 %目标类别确定之后的混淆矩阵
cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵

[acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
mean_acc(i,j)=(acc_low(i,j)+acc_medium(i,j)+acc_high(i,j))/3;

TestingAccuracy(i,j)=Acc(i,j);
% [para_ind]=find(mean_acc==max(mean_acc));%找出最优参数

acc_low_mean=mean((acc_low'))';
acc_medium_mean=mean((acc_medium'))';
acc_high_mean=mean((acc_high'))';
Acc_mean=mean((Acc'))';  %对角线值除以总值的测试精度
acc_mean=mean((mean_acc'))';
sen_mean=mean((sen'))';
spe_mean=mean((spe'))';
pre_mean=mean((pre'))';
npv_mean=mean((npv'))';
f1_mean=mean((f1'))';
fnr_mean=mean((fnr'))';
fpr_mean=mean((fpr'))';
fdr_mean=mean((fdr'))';
foe_mean=mean((foe'))';
mcc_mean=mean((mcc'))';
bm_mean=mean((bm'))';
mk_mean=mean((mk'))';
ka_mean=mean((ka'))';

acc_low_sd=std((acc_low'))';
acc_medium_sd=std((acc_medium'))';
acc_high_sd=std((acc_high'))';
Acc_sd=std((Acc'))';  %对角线值除以总值的测试精度
acc_sd=std((mean_acc'))';
sen_sd=std((sen'))';
spe_sd=std((spe'))';
pre_sd=std((pre'))';
npv_sd=std((npv'))';
f1_sd=std((f1'))';
fnr_sd=std((fnr'))';
fpr_sd=std((fpr'))';
fdr_sd=std((fdr'))';
foe_sd=std((foe'))';
mcc_sd=std((mcc'))';
bm_sd=std((bm'))';
mk_sd=std((mk'))';
ka_sd=std((ka'))';

acc_TrainingAccuracy=mean((TrainingAccuracy'))';
acc_TestingAccuracy=mean((TestingAccuracy'))';

plot_TrainingAccuracy=mean(TrainingAccuracy);
plot_TestingAccuracy=mean(TestingAccuracy);

par_ave_mean=mean(mean(TrainingTime'+TestingTime')');
par_ave_sd=mean(std(TrainingTime'+TestingTime')');

accuracy_par_ave_mean=mean(mean(TestingAccuracy')');
accuracy_par_ave_sd=mean(std(TestingAccuracy')');
subject_average_accuracy2=[accuracy_par_ave_mean accuracy_par_ave_sd]; %重复50次实验

per_index_single2=[acc_low_mean,acc_medium_mean,acc_high_mean,Acc_mean,acc_mean,sen_mean,spe_mean,pre_mean,npv_mean,f1_mean,fnr_mean,fpr_mean,fdr_mean,foe_mean,mcc_mean,bm_mean,mk_mean,ka_mean];

per_index_ave2_mean=mean([acc_low_mean,acc_medium_mean,acc_high_mean,Acc_mean,acc_mean,sen_mean,spe_mean,pre_mean,npv_mean,f1_mean,fnr_mean,fpr_mean,fdr_mean,foe_mean,mcc_mean,bm_mean,mk_mean,ka_mean])';
per_index_ave2_sd=mean([acc_low_sd,acc_medium_sd,acc_high_sd,Acc_sd,acc_sd,sen_sd,spe_sd,pre_sd,npv_sd,f1_sd,fnr_sd,fpr_sd,fdr_sd,foe_sd,mcc_sd,bm_sd,mk_sd,ka_sd])';
per_index_ave2=[per_index_ave2_mean per_index_ave2_sd];

accuracy2=[acc_TrainingAccuracy acc_TestingAccuracy];
time2=[par_ave_mean par_ave_sd];

% % save F:\matlab\trial_procedure\study_1\performance_comparison\lr_ovo_cross_subject\task1_session2 per_index accuracy time 
% save F:\matlab\trial_procedure\study_1\data_analysis\lr_cross_subject\task1_session2 per_index_single2 per_index_ave2 accuracy2 time2 subject_average_accuracy2
save F:\matlab\trial_procedure\study_1\data_analysis\lpp_lr_cross_subject\task1_session2 per_index_single2 per_index_ave2 accuracy2 time2 subject_average_accuracy2

end
end

%lr_ovo跨被试（任务二6位）
load F:\matlab\trial_procedure\study_1\features\ex2\s1_1
x1_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s2_1
x2_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s3_1
x3_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s4_1
x4_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s5_1
x5_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s6_1
x6_1=x;

x33=[x1_1;x2_1;x3_1;x4_1;x5_1;x6_1];

rr=70;
%进行LPP
fea =x33;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x33=fea*eigvector;

load F:\matlab\trial_procedure\study_1\features\ex2\s1_2
x1_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s2_2
x2_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s3_2
x3_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s4_2
x4_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s5_2
x5_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s6_2
x6_1=x;

x44=[x1_1;x2_1;x3_1;x4_1;x5_1;x6_1];

%进行LPP
fea =x44;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x44=fea*eigvector;

%任务二两个阶段的数据集
x3=x33;
x4=x44;

y_1=[y;y;y;y;y;y];

K=6;
low=size(find(y_1==1));
medium=size(find(y_1==2));
high=size(find(y_1==3));

[xtr_all_3, xte_all_3] = kfcv(x3,K,'off');
[ytr_all_3, yte_all_3] = kfcv(y_1,K,'off');

[xtr_all_4, xte_all_4] = kfcv(x4,K,'off');
[ytr_all_4, yte_all_4] = kfcv(y_1,K,'off');

%%%%%session1
for j=1:50
for i=1:K
    
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    
   for z=1:5
   y_low_1(1+(z-1)*580:580+(z-1)*580,:)=ytr_3([1+(z-1)*1450:145+(z-1)*1450 436+(z-1)*1450:580+(z-1)*1450 871+(z-1)*1450:1015+(z-1)*1450 1306+(z-1)*1450:1450+(z-1)*1450]);
   y_medium_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_3([146+(z-1)*1450:290+(z-1)*1450 581+(z-1)*1450:725+(z-1)*1450 1016+(z-1)*1450:1160+(z-1)*1450 ]);
   y_high_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_3([291+(z-1)*1450:435+(z-1)*1450 726+(z-1)*1450:870+(z-1)*1450 1161+(z-1)*1450:1305+(z-1)*1450]);
   end

   for z=1:5
   x_low_1(1+(z-1)*580:580+(z-1)*580,:)=xtr_3([1+(z-1)*1450:145+(z-1)*1450 436+(z-1)*1450:580+(z-1)*1450 871+(z-1)*1450:1015+(z-1)*1450 1306+(z-1)*1450:1450+(z-1)*1450],:);
   x_medium_1(1+(z-1)*435:435+(z-1)*435,:)=xtr_3([146+(z-1)*1450:290+(z-1)*1450 581+(z-1)*1450:725+(z-1)*1450 1016+(z-1)*1450:1160+(z-1)*1450],:);
   x_high_1(1+(z-1)*435:435+(z-1)*435,:)=xtr_3([291+(z-1)*1450:435+(z-1)*1450 726+(z-1)*1450:870+(z-1)*1450 1161+(z-1)*1450:1305+(z-1)*1450],:);
   end
    
%前者“+”，后者“-”
x_4=[x_low_1;x_medium_1];
y_4=[0*ones(length(y_low_1),1);1*ones(length(y_medium_1),1)];

x_5=[x_low_1;x_high_1];
y_5=[0*ones(length(y_low_1),1);1*ones(length(y_high_1),1)];

x_6=[x_medium_1;x_high_1];
y_6=[0*ones(length(y_medium_1),1);1*ones(length(y_high_1),1)];
    
   rand('state',0)  %resets the generator to its initial state
start_time_train_1=cputime;   
model_4=glmfit(x_4,[y_4 ones(5075,1)],'binomial','link','logit');
model_5=glmfit(x_5,[y_5 ones(5075,1)],'binomial','link','logit');
model_6=glmfit(x_6,[y_6 ones(4350,1)],'binomial','link','logit');
end_time_train_1=cputime;
TrainingTime_1(i,j)=end_time_train_1-start_time_train_1;

start_time_test_1=cputime;
yte_p_4=round(glmval(model_4,xte_3,'logit'));
yte_p_5=round(glmval(model_5,xte_3,'logit'));
yte_p_6=round(glmval(model_6,xte_3,'logit'));
end_time_test_1=cputime;
TestingTime_1(i,j)=end_time_test_1-start_time_test_1; 

%训练精度
predict_train_4=round(glmval(model_4,x_4,'logit'));
TrainingAccuracy_4(i,j)=length(find(predict_train_4==y_4))/size(y_4,1);

%训练精度
predict_train_5=round(glmval(model_5,x_5,'logit'));
TrainingAccuracy_5(i,j)=length(find(predict_train_5==y_5))/size(y_5,1);

%训练精度
predict_train_6=round(glmval(model_6,x_6,'logit'));
TrainingAccuracy_6(i,j)=length(find(predict_train_6==y_6))/size(y_6,1);

TrainingAccuracy_11(i,j)=(TrainingAccuracy_4(i,j)+TrainingAccuracy_5(i,j)+TrainingAccuracy_6(i,j))/3;

for z=1:1450
yte_p_11(z,:)=ovo_code(yte_p_4(1+(z-1)*1,:),yte_p_5(1+(z-1)*1,:),yte_p_6(1+(z-1)*1,:));
end

% yte_p_perdict_1(1+(i-1)*1450:1450+(i-1)*1450,1)=yte_p_11;
% yte_p_label_1=yte_p_perdict_1;
% save F:\matlab\trial_procedure\study_1\roc_data\lr_cross_subject\task2_session1 yte_p_label_1 y_1
%查看三类数目
c4=size(find(yte_p_11==1));
c5=size(find(yte_p_11==2));
c6=size(find(yte_p_11==3));

 %目标类别确定之后的混淆矩阵
cte_1(:,:,i,j)=cfmatrix(yte_3,yte_p_11);%计算混淆矩阵

[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
mean_acc_1(i,j)=(acc_low_1(i,j)+acc_medium_1(i,j)+acc_high_1(i,j))/3;

TestingAccuracy_11(i,j)=Acc_1(i,j);

% [para_ind]=find(mean_acc==max(mean_acc));%找出最优参数

acc_low_mean_1=mean((acc_low_1'))';
acc_medium_mean_1=mean((acc_medium_1'))';
acc_high_mean_1=mean((acc_high_1'))';
Acc_mean_1=mean((Acc_1'))';  %对角线值除以总值的测试精度
acc_mean_1=mean((mean_acc_1'))';
sen_mean_1=mean((sen_1'))';
spe_mean_1=mean((spe_1'))';
pre_mean_1=mean((pre_1'))';
npv_mean_1=mean((npv_1'))';
f1_mean_1=mean((f1_1'))';
fnr_mean_1=mean((fnr_1'))';
fpr_mean_1=mean((fpr_1'))';
fdr_mean_1=mean((fdr_1'))';
foe_mean_1=mean((foe_1'))';
mcc_mean_1=mean((mcc_1'))';
bm_mean_1=mean((bm_1'))';
mk_mean_1=mean((mk_1'))';
ka_mean_1=mean((ka_1'))';

acc_low_sd_1=std((acc_low_1'))';
acc_medium_sd_1=std((acc_medium_1'))';
acc_high_sd_1=std((acc_high_1'))';
Acc_sd_1=std((Acc_1'))';  %对角线值除以总值的测试精度
acc_sd_1=std((mean_acc_1'))';
sen_sd_1=std((sen_1'))';
spe_sd_1=std((spe_1'))';
pre_sd_1=std((pre_1'))';
npv_sd_1=std((npv_1'))';
f1_sd_1=std((f1_1'))';
fnr_sd_1=std((fnr_1'))';
fpr_sd_1=std((fpr_1'))';
fdr_sd_1=std((fdr_1'))';
foe_sd_1=std((foe_1'))';
mcc_sd_1=std((mcc_1'))';
bm_sd_1=std((bm_1'))';
mk_sd_1=std((mk_1'))';
ka_sd_1=std((ka_1'))';

acc_TrainingAccuracy_1=mean((TrainingAccuracy_11'))';
acc_TestingAccuracy_1=mean((TestingAccuracy_11'))';

plot_TrainingAccuracy_1=mean(TrainingAccuracy_11);
plot_TestingAccuracy_1=mean(TestingAccuracy_11);

par_ave_mean_1=mean(mean(TrainingTime_1'+TestingTime_1')');
par_ave_sd_1=mean(std(TrainingTime_1'+TestingTime_1')');

accuracy_par_ave_mean_1=mean(mean(TestingAccuracy_11')');
accuracy_par_ave_sd_1=mean(std(TestingAccuracy_11')');
subject_average_accuracy3=[accuracy_par_ave_mean_1 accuracy_par_ave_sd_1]; %重复50次实验

per_index_single3=[acc_low_mean_1,acc_medium_mean_1,acc_high_mean_1,Acc_mean_1,acc_mean_1,sen_mean_1,spe_mean_1,pre_mean_1,npv_mean_1,f1_mean_1,fnr_mean_1,fpr_mean_1,fdr_mean_1,foe_mean_1,mcc_mean_1,bm_mean_1,mk_mean_1,ka_mean_1];

per_index_ave3_mean=mean([acc_low_mean_1,acc_medium_mean_1,acc_high_mean_1,Acc_mean_1,acc_mean_1,sen_mean_1,spe_mean_1,pre_mean_1,npv_mean_1,f1_mean_1,fnr_mean_1,fpr_mean_1,fdr_mean_1,foe_mean_1,mcc_mean_1,bm_mean_1,mk_mean_1,ka_mean_1])';
per_index_ave3_sd=mean([acc_low_sd_1,acc_medium_sd_1,acc_high_sd_1,Acc_sd_1,acc_sd_1,sen_sd_1,spe_sd_1,pre_sd_1,npv_sd_1,f1_sd_1,fnr_sd_1,fpr_sd_1,fdr_sd_1,foe_sd_1,mcc_sd_1,bm_sd_1,mk_sd_1,ka_sd_1])';
per_index_ave3=[per_index_ave3_mean per_index_ave3_sd];

accuracy3=[acc_TrainingAccuracy_1 acc_TestingAccuracy_1];
time3=[par_ave_mean_1 par_ave_sd_1];
 
% % save F:\matlab\trial_procedure\study_1\performance_comparison\lr_ovo_cross_subject\task2_session1 per_index_1 accuracy_1 time_1 
% save F:\matlab\trial_procedure\study_1\data_analysis\lr_cross_subject\task2_session1 per_index_single3 per_index_ave3 accuracy3 time3 subject_average_accuracy3
save F:\matlab\trial_procedure\study_1\data_analysis\lpp_lr_cross_subject\task2_session1 per_index_single3 per_index_ave3 accuracy3 time3 subject_average_accuracy3

end
end

%%%%%session2
for j=1:50
for i=1:K
    
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));
    
   for z=1:5
   y_low_1(1+(z-1)*580:580+(z-1)*580,:)=ytr_4([1+(z-1)*1450:145+(z-1)*1450 436+(z-1)*1450:580+(z-1)*1450 871+(z-1)*1450:1015+(z-1)*1450 1306+(z-1)*1450:1450+(z-1)*1450]);
   y_medium_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_4([146+(z-1)*1450:290+(z-1)*1450 581+(z-1)*1450:725+(z-1)*1450 1016+(z-1)*1450:1160+(z-1)*1450 ]);
   y_high_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_4([291+(z-1)*1450:435+(z-1)*1450 726+(z-1)*1450:870+(z-1)*1450 1161+(z-1)*1450:1305+(z-1)*1450]);
   end

   for z=1:5
   x_low_1(1+(z-1)*580:580+(z-1)*580,:)=xtr_4([1+(z-1)*1450:145+(z-1)*1450 436+(z-1)*1450:580+(z-1)*1450 871+(z-1)*1450:1015+(z-1)*1450 1306+(z-1)*1450:1450+(z-1)*1450],:);
   x_medium_1(1+(z-1)*435:435+(z-1)*435,:)=xtr_4([146+(z-1)*1450:290+(z-1)*1450 581+(z-1)*1450:725+(z-1)*1450 1016+(z-1)*1450:1160+(z-1)*1450],:);
   x_high_1(1+(z-1)*435:435+(z-1)*435,:)=xtr_4([291+(z-1)*1450:435+(z-1)*1450 726+(z-1)*1450:870+(z-1)*1450 1161+(z-1)*1450:1305+(z-1)*1450],:);
   end
    
%前者“+”，后者“-”
x_4=[x_low_1;x_medium_1];
y_4=[0*ones(length(y_low_1),1);1*ones(length(y_medium_1),1)];

x_5=[x_low_1;x_high_1];
y_5=[0*ones(length(y_low_1),1);1*ones(length(y_high_1),1)];

x_6=[x_medium_1;x_high_1];
y_6=[0*ones(length(y_medium_1),1);1*ones(length(y_high_1),1)];
    
   rand('state',0)  %resets the generator to its initial state
start_time_train_1=cputime;   
model_4=glmfit(x_4,[y_4 ones(5075,1)],'binomial','link','logit');
model_5=glmfit(x_5,[y_5 ones(5075,1)],'binomial','link','logit');
model_6=glmfit(x_6,[y_6 ones(4350,1)],'binomial','link','logit');
end_time_train_1=cputime;
TrainingTime_1(i,j)=end_time_train_1-start_time_train_1;

start_time_test_1=cputime;
yte_p_4=round(glmval(model_4,xte_4,'logit'));
yte_p_5=round(glmval(model_5,xte_4,'logit'));
yte_p_6=round(glmval(model_6,xte_4,'logit'));
end_time_test_1=cputime;
TestingTime_1(i,j)=end_time_test_1-start_time_test_1; 

%训练精度
predict_train_4=round(glmval(model_4,x_4,'logit'));
TrainingAccuracy_4(i,j)=length(find(predict_train_4==y_4))/size(y_4,1);

%训练精度
predict_train_5=round(glmval(model_5,x_5,'logit'));
TrainingAccuracy_5(i,j)=length(find(predict_train_5==y_5))/size(y_5,1);

%训练精度
predict_train_6=round(glmval(model_6,x_6,'logit'));
TrainingAccuracy_6(i,j)=length(find(predict_train_6==y_6))/size(y_6,1);

TrainingAccuracy_11(i,j)=(TrainingAccuracy_4(i,j)+TrainingAccuracy_5(i,j)+TrainingAccuracy_6(i,j))/3;

for z=1:1450
yte_p_11(z,:)=ovo_code(yte_p_4(1+(z-1)*1,:),yte_p_5(1+(z-1)*1,:),yte_p_6(1+(z-1)*1,:));
end

% yte_p_perdict_1(1+(i-1)*1450:1450+(i-1)*1450,1)=yte_p_11;
% yte_p_label_1=yte_p_perdict_1;
% save F:\matlab\trial_procedure\study_1\roc_data\lr_cross_subject\task2_session2 yte_p_label_1 y_1
%查看三类数目
c4=size(find(yte_p_11==1));
c5=size(find(yte_p_11==2));
c6=size(find(yte_p_11==3));

 %目标类别确定之后的混淆矩阵
cte_1(:,:,i,j)=cfmatrix(yte_4,yte_p_11);%计算混淆矩阵

[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
mean_acc_1(i,j)=(acc_low_1(i,j)+acc_medium_1(i,j)+acc_high_1(i,j))/3;

TestingAccuracy_11(i,j)=Acc_1(i,j);
% [para_ind]=find(mean_acc==max(mean_acc));%找出最优参数

acc_low_mean_1=mean((acc_low_1'))';
acc_medium_mean_1=mean((acc_medium_1'))';
acc_high_mean_1=mean((acc_high_1'))';
Acc_mean_1=mean((Acc_1'))';  %对角线值除以总值的测试精度
acc_mean_1=mean((mean_acc_1'))';
sen_mean_1=mean((sen_1'))';
spe_mean_1=mean((spe_1'))';
pre_mean_1=mean((pre_1'))';
npv_mean_1=mean((npv_1'))';
f1_mean_1=mean((f1_1'))';
fnr_mean_1=mean((fnr_1'))';
fpr_mean_1=mean((fpr_1'))';
fdr_mean_1=mean((fdr_1'))';
foe_mean_1=mean((foe_1'))';
mcc_mean_1=mean((mcc_1'))';
bm_mean_1=mean((bm_1'))';
mk_mean_1=mean((mk_1'))';
ka_mean_1=mean((ka_1'))';

acc_low_sd_1=std((acc_low_1'))';
acc_medium_sd_1=std((acc_medium_1'))';
acc_high_sd_1=std((acc_high_1'))';
Acc_sd_1=std((Acc_1'))';  %对角线值除以总值的测试精度
acc_sd_1=std((mean_acc_1'))';
sen_sd_1=std((sen_1'))';
spe_sd_1=std((spe_1'))';
pre_sd_1=std((pre_1'))';
npv_sd_1=std((npv_1'))';
f1_sd_1=std((f1_1'))';
fnr_sd_1=std((fnr_1'))';
fpr_sd_1=std((fpr_1'))';
fdr_sd_1=std((fdr_1'))';
foe_sd_1=std((foe_1'))';
mcc_sd_1=std((mcc_1'))';
bm_sd_1=std((bm_1'))';
mk_sd_1=std((mk_1'))';
ka_sd_1=std((ka_1'))';

acc_TrainingAccuracy_1=mean((TrainingAccuracy_11'))';
acc_TestingAccuracy_1=mean((TestingAccuracy_11'))';

plot_TrainingAccuracy_1=mean(TrainingAccuracy_11);
plot_TestingAccuracy_1=mean(TestingAccuracy_11);

par_ave_mean_1=mean(mean(TrainingTime_1'+TestingTime_1')');
par_ave_sd_1=mean(std(TrainingTime_1'+TestingTime_1')');

accuracy_par_ave_mean_1=mean(mean(TestingAccuracy_11')');
accuracy_par_ave_sd_1=mean(std(TestingAccuracy_11')');

subject_average_accuracy4=[accuracy_par_ave_mean_1 accuracy_par_ave_sd_1]; %重复50次实验

per_index_single4=[acc_low_mean_1,acc_medium_mean_1,acc_high_mean_1,Acc_mean_1,acc_mean_1,sen_mean_1,spe_mean_1,pre_mean_1,npv_mean_1,f1_mean_1,fnr_mean_1,fpr_mean_1,fdr_mean_1,foe_mean_1,mcc_mean_1,bm_mean_1,mk_mean_1,ka_mean_1];

per_index_ave4_mean=mean([acc_low_mean_1,acc_medium_mean_1,acc_high_mean_1,Acc_mean_1,acc_mean_1,sen_mean_1,spe_mean_1,pre_mean_1,npv_mean_1,f1_mean_1,fnr_mean_1,fpr_mean_1,fdr_mean_1,foe_mean_1,mcc_mean_1,bm_mean_1,mk_mean_1,ka_mean_1])';
per_index_ave4_sd=mean([acc_low_sd_1,acc_medium_sd_1,acc_high_sd_1,Acc_sd_1,acc_sd_1,sen_sd_1,spe_sd_1,pre_sd_1,npv_sd_1,f1_sd_1,fnr_sd_1,fpr_sd_1,fdr_sd_1,foe_sd_1,mcc_sd_1,bm_sd_1,mk_sd_1,ka_sd_1])';
per_index_ave4=[per_index_ave4_mean per_index_ave4_sd];

accuracy4=[acc_TrainingAccuracy_1 acc_TestingAccuracy_1];
time4=[par_ave_mean_1 par_ave_sd_1];

% % save F:\matlab\trial_procedure\study_1\performance_comparison\lr_ovo_cross_subject\task2_session2 per_index_1 accuracy_1 time_1 
% save F:\matlab\trial_procedure\study_1\data_analysis\lr_cross_subject\task2_session2 per_index_single4 per_index_ave4 accuracy4 time4 subject_average_accuracy4
save F:\matlab\trial_procedure\study_1\data_analysis\lpp_lr_cross_subject\task2_session2 per_index_single4 per_index_ave4 accuracy4 time4 subject_average_accuracy4

end
end
