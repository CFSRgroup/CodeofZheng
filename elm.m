% %ELM跨被试（任务一8位）
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
% low=size(find(y==1));
% medium=size(find(y==2));
% high=size(find(y==3));
% 
% %8位被试
% xtr=x;
% ytr=y;
% xte=x8;
% yte=y1;
% 
% rand('state',0)  %resets the generator to its initial state
% %elm分类器
% % [TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(xtr,ytr, 1, 50, 'sig',2^0);%ELM只认1，2类标签，不认0，1
% % [TestingTime, TestingAccuracy, ty] = elm_predict(xte,yte,elm_model);  % TY: the actual output of the testing data
% 
% %elm_kernel分类器
% [ty] = elm_kernel_m(xtr, ytr, yte, xte, 1, 2^5, 'lin_kernel', 2^15);
% 
% for zz=1:size(ty, 2)
%     [~,label_index_actual(zz,1)]=max(ty(:,zz));
% end 
%     yte_p=label_index_actual;
%     
% cte=cfmatrix(yte,yte_p);%计算混淆矩阵
% 
% [acc_low,acc_medium,acc_high,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cte); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% 
% per_index=[acc_low;acc_medium;acc_high;mean_acc;sen;spe;pre;npv;f1;fnr;fpr;fdr;foe;mcc;bm;mk;ka];
% % time=[TestingAccuracy;TestingTime;TrainingAccuracy;TrainingTime];
% % 
% % save F:\matlab\trial_procedure\study_1\baseline_performance\elm_kernel_cross_subject\task1_session1_H per_index 
% save F:\matlab\trial_procedure\study_1\baseline_performance\elm_kernel_cross_subject\task1_session2_H per_index 
% 
% plot(yte);
% title('(p) Subject H, session 2, ACC=0.');
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





%ELM跨被试（任务二6位）
% clc;
% clear;
% close all;
% warning off;
% 
% load F:\matlab\trial_procedure\study_1\features\ex2\s1_1
% x1=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s2_1
% x2=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s3_1
% x3=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s4_1
% x4=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s5_1
% x5=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s6_1
% x6=x;
% 
% % load F:\matlab\trial_procedure\study_1\features\ex2\s1_2
% % x1=x;
% % load F:\matlab\trial_procedure\study_1\features\ex2\s2_2
% % x2=x;
% % load F:\matlab\trial_procedure\study_1\features\ex2\s3_2
% % x3=x;
% % load F:\matlab\trial_procedure\study_1\features\ex2\s4_2
% % x4=x;
% % load F:\matlab\trial_procedure\study_1\features\ex2\s5_2
% % x5=x;
% % load F:\matlab\trial_procedure\study_1\features\ex2\s6_2
% % x6=x;
% 
% 
% x=[x1;x2;x3;x4;x5];
% y1=y;
% y=[y;y;y;y;y];
% 
% low=size(find(y==1));
% medium=size(find(y==2));
% high=size(find(y==3));
% 
% %8位被试
% xtr=x;
% ytr=y;
% xte=x6;
% yte=y1;
% 
% rand('state',0)  %resets the generator to its initial state每次生成结果一致
% %elm分类器
% % [TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(xtr,ytr, 1, 50, 'sig',2^0);%ELM只认1，2类标签，不认0，1
% % [TestingTime, TestingAccuracy, ty] = elm_predict(xte,yte,elm_model);  % TY: the actual output of the testing data
% 
% %elm_kernel分类器
% [ty] = elm_kernel_m(xtr, ytr, yte, xte, 1, 2^15, 'lin_kernel', []);
% 
% for zz=1:size(ty, 2)
%     [~,label_index_actual(zz,1)]=max(ty(:,zz));
% end 
%     yte_p=label_index_actual;
%     
% cte=cfmatrix(yte,yte_p);%计算混淆矩阵
% 
% [acc_low,acc_medium,acc_high,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cte); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% 
% per_index=[acc_low;acc_medium;acc_high;mean_acc;sen;spe;pre;npv;f1;fnr;fpr;fdr;foe;mcc;bm;mk;ka];
% time=[TestingAccuracy;TestingTime;TrainingAccuracy;TrainingTime];

% % save F:\matlab\trial_procedure\study_1\baseline_performance\elm_kernel_cross_subject\task2_session1_N per_index 
% save F:\matlab\trial_procedure\study_1\baseline_performance\elm_kernel_cross_subject\task2_session2_N per_index 
% 
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
%ELM跨被试（任务一8位）
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

% rr=70;
% %进行LPP
% fea =x11;
% options = [];
% options.Metric = 'Euclidean';
% % options.NeighborMode = 'Supervised';
% % options.gnd = y;
% options.ReducedDim=rr;
% W = constructW(fea,options);      
% options.PCARatio = 1;
% [eigvector, eigvalue] = LPP(W, options, fea);
% x11=fea*eigvector;

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

% %进行LPP
% fea =x22;
% options = [];
% options.Metric = 'Euclidean';
% % options.NeighborMode = 'Supervised';
% % options.gnd = y;
% options.ReducedDim=rr;
% W = constructW(fea,options);      
% options.PCARatio = 1;
% [eigvector, eigvalue] = LPP(W, options, fea);
% x22=fea*eigvector;

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
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    
   rand('state',0)  %resets the generator to its initial state

%有放回随机抽样，具有重复性和遗漏性，每次改变训练样本，测试样本不变，针对一个被试测试，训练7个基分类器，最后用投票法，得出最终分类标签，任务一抽样7次
L=18900;
sampleindex=randi(L,18900,1);
xtr_1(1:18900,:)=xtr_1(sampleindex,:);
ytr_1(1:18900,:)=ytr_1(sampleindex,:);
   
   %elm分类器
[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p_1=label_index_actual;

%2次
sampleindex=randi(L,18900,1);
xtr_1(1:18900,:)=xtr_1(sampleindex,:);
ytr_1(1:18900,:)=ytr_1(sampleindex,:);
   
   %elm分类器
[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p_2=label_index_actual;

%3次
sampleindex=randi(L,18900,1);
xtr_1(1:18900,:)=xtr_1(sampleindex,:);
ytr_1(1:18900,:)=ytr_1(sampleindex,:);
   
   %elm分类器
[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p_3=label_index_actual;

    %4次
sampleindex=randi(L,18900,1);
xtr_1(1:18900,:)=xtr_1(sampleindex,:);
ytr_1(1:18900,:)=ytr_1(sampleindex,:);
   
   %elm分类器
[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p_4=label_index_actual;
    
    %5次
sampleindex=randi(L,18900,1);
xtr_1(1:18900,:)=xtr_1(sampleindex,:);
ytr_1(1:18900,:)=ytr_1(sampleindex,:);
   
   %elm分类器
[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p_5=label_index_actual;

%6次
sampleindex=randi(L,18900,1);
xtr_1(1:18900,:)=xtr_1(sampleindex,:);
ytr_1(1:18900,:)=ytr_1(sampleindex,:);
   
   %elm分类器
[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p_6=label_index_actual;

%7次
sampleindex=randi(L,18900,1);
xtr_1(1:18900,:)=xtr_1(sampleindex,:);
ytr_1(1:18900,:)=ytr_1(sampleindex,:);
   
   %elm分类器
[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p_7=label_index_actual;
    
y_1_1=[yte_p_1 yte_p_2 yte_p_3 yte_p_4 yte_p_5 yte_p_6 yte_p_7];    
%最终标签
yte_p=mode(y_1_1')';




%elm_kernel分类器
% [ty] = elm_kernel_m(xtr_1, ytr_1, yte_1, xte_1, 1, 2^5, 'lin_kernel', 2^15);
%原程序标签代码
% for zz=1:size(ty, 2)
%     [~,label_index_actual(zz,1)]=max(ty(:,zz));
% end 
%     yte_p=label_index_actual;
    
%     yte_p_label_predict(1+(i-1)*2700:2700+(i-1)*2700,1)=yte_p;
%     yte_p_label=yte_p_label_predict;
%     save F:\matlab\trial_procedure\study_1\roc_data\elm_cross_subject\task1_session1 yte_p_label y    
    
cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵

[acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
mean_acc(i,j)=(acc_low(i,j)+acc_medium(i,j)+acc_high(i,j))/3;


% % [para_ind]=find(mean_acc==max(mean_acc));%找出最优参数
% 
% acc_low_mean=mean((acc_low'))';
% acc_medium_mean=mean((acc_medium'))';
% acc_high_mean=mean((acc_high'))';
% Acc_mean=mean((Acc'))';  %对角线值除以总值的测试精度
% acc_mean=mean((mean_acc'))';
% sen_mean=mean((sen'))';
% spe_mean=mean((spe'))';
% pre_mean=mean((pre'))';
% npv_mean=mean((npv'))';
% f1_mean=mean((f1'))';
% fnr_mean=mean((fnr'))';
% fpr_mean=mean((fpr'))';
% fdr_mean=mean((fdr'))';
% foe_mean=mean((foe'))';
% mcc_mean=mean((mcc'))';
% bm_mean=mean((bm'))';
% mk_mean=mean((mk'))';
% ka_mean=mean((ka'))';
% 
% acc_low_sd=std((acc_low'))';
% acc_medium_sd=std((acc_medium'))';
% acc_high_sd=std((acc_high'))';
% Acc_sd=std((Acc'))';  %对角线值除以总值的测试精度
% acc_sd=std((mean_acc'))';
% sen_sd=std((sen'))';
% spe_sd=std((spe'))';
% pre_sd=std((pre'))';
% npv_sd=std((npv'))';
% f1_sd=std((f1'))';
% fnr_sd=std((fnr'))';
% fpr_sd=std((fpr'))';
% fdr_sd=std((fdr'))';
% foe_sd=std((foe'))';
% mcc_sd=std((mcc'))';
% bm_sd=std((bm'))';
% mk_sd=std((mk'))';
% ka_sd=std((ka'))';
% 
% acc_TrainingAccuracy=mean((TrainingAccuracy'))';
% acc_TestingAccuracy=mean((TestingAccuracy'))';
% 
% plot_TrainingAccuracy=mean(TrainingAccuracy);
% plot_TestingAccuracy=mean(TestingAccuracy);
% 
% par_ave_mean=mean(mean(TrainingTime'+TestingTime')');
% par_ave_sd=mean(std(TrainingTime'+TestingTime')');
% 
% % per_index=[acc_low_mean,acc_medium_mean,acc_high_mean,Acc_mean,acc_mean,sen_mean,spe_mean,pre_mean,npv_mean,f1_mean,fnr_mean,fpr_mean,fdr_mean,foe_mean,mcc_mean,bm_mean,mk_mean,ka_mean];
% % accuracy=[acc_TrainingAccuracy acc_TestingAccuracy];
% % time=[par_ave_mean par_ave_sd];
% 
% accuracy_par_ave_mean=mean(mean(TestingAccuracy')');
% accuracy_par_ave_sd=mean(std(TestingAccuracy')');
% subject_average_accuracy1=[accuracy_par_ave_mean accuracy_par_ave_sd]; %重复50次实验
% 
% per_index_single1=[acc_low_mean,acc_medium_mean,acc_high_mean,Acc_mean,acc_mean,sen_mean,spe_mean,pre_mean,npv_mean,f1_mean,fnr_mean,fpr_mean,fdr_mean,foe_mean,mcc_mean,bm_mean,mk_mean,ka_mean];
% 
% per_index_ave1_mean=mean([acc_low_mean,acc_medium_mean,acc_high_mean,Acc_mean,acc_mean,sen_mean,spe_mean,pre_mean,npv_mean,f1_mean,fnr_mean,fpr_mean,fdr_mean,foe_mean,mcc_mean,bm_mean,mk_mean,ka_mean])';
% per_index_ave1_sd=mean([acc_low_sd,acc_medium_sd,acc_high_sd,Acc_sd,acc_sd,sen_sd,spe_sd,pre_sd,npv_sd,f1_sd,fnr_sd,fpr_sd,fdr_sd,foe_sd,mcc_sd,bm_sd,mk_sd,ka_sd])';
% per_index_ave1=[per_index_ave1_mean per_index_ave1_sd];
% 
% accuracy1=[acc_TrainingAccuracy acc_TestingAccuracy];
% time1=[par_ave_mean par_ave_sd];

% save F:\matlab\trial_procedure\study_1\performance_comparison\elm_cross_subject\task1_session1 per_index accuracy time plot_TrainingAccuracy plot_TestingAccuracy
% save F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task1_session1 per_index_single1 per_index_ave1 accuracy1 time1 subject_average_accuracy1
% save F:\matlab\trial_procedure\study_1\data_analysis\lpp_elm_cross_subject\task1_session1 per_index_single1 per_index_ave1 accuracy1 time1 subject_average_accuracy1

end
end

per_1=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
save F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\task1_session1 per_1

% subplot(2,1,1);
% plot(plot_TrainingAccuracy,'s-g');
% title('(a) ELM: Case 1','FontWeight','bold');
% set(gca,'XTick',0:10:50);
% set(gca,'XTickLabel',{'0','100','200','300','400','500'});
% set(gca,'YTick',0:0.05:1);
% xlabel('Number of hidden neurons','FontWeight','bold');
% ylabel('Accuracy','FontWeight','bold');
% grid on;
% hold on;
% plot(plot_TestingAccuracy,'o-b');
% hold on;



%%%%%session2
for j=1
for i=1:K
    
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));

   rand('state',0)  %resets the generator to its initial state

%有放回随机抽样，具有重复性和遗漏性，每次改变训练样本，测试样本不变，针对一个被试测试，训练7个基分类器，最后用投票法，得出最终分类标签，任务一抽样7次
L=18900;
sampleindex=randi(L,18900,1);
xtr_2(1:18900,:)=xtr_2(sampleindex,:);
ytr_2(1:18900,:)=ytr_2(sampleindex,:);
   
    %elm分类器
[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p_1=label_index_actual;

%2次
sampleindex=randi(L,18900,1);
xtr_2(1:18900,:)=xtr_2(sampleindex,:);
ytr_2(1:18900,:)=ytr_2(sampleindex,:);
   
   %elm分类器
[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p_2=label_index_actual;

%3次
sampleindex=randi(L,18900,1);
xtr_2(1:18900,:)=xtr_2(sampleindex,:);
ytr_2(1:18900,:)=ytr_2(sampleindex,:);
   
  %elm分类器
[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p_3=label_index_actual;

    %4次
sampleindex=randi(L,18900,1);
xtr_2(1:18900,:)=xtr_2(sampleindex,:);
ytr_2(1:18900,:)=ytr_2(sampleindex,:);
   
   %elm分类器
[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p_4=label_index_actual;
    
    %5次
sampleindex=randi(L,18900,1);
xtr_2(1:18900,:)=xtr_2(sampleindex,:);
ytr_2(1:18900,:)=ytr_2(sampleindex,:);
   
  %elm分类器
[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p_5=label_index_actual;

%6次
sampleindex=randi(L,18900,1);
xtr_2(1:18900,:)=xtr_2(sampleindex,:);
ytr_2(1:18900,:)=ytr_2(sampleindex,:);
   
   %elm分类器
[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p_6=label_index_actual;

%7次
sampleindex=randi(L,18900,1);
xtr_2(1:18900,:)=xtr_2(sampleindex,:);
ytr_2(1:18900,:)=ytr_2(sampleindex,:);
   
   %elm分类器
[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p_7=label_index_actual;
    
y_1_2=[yte_p_1 yte_p_2 yte_p_3 yte_p_4 yte_p_5 yte_p_6 yte_p_7];    
%最终标签
yte_p=mode(y_1_2')';

%elm_kernel分类器
% [ty] = elm_kernel_m(xtr_1, ytr_1, yte_1, xte_1, 1, 2^5, 'lin_kernel', 2^15);

% for zz=1:size(ty, 2)
%     [~,label_index_actual(zz,1)]=max(ty(:,zz));
% end 
%     yte_p=label_index_actual;

%     yte_p_label_predict(1+(i-1)*2700:2700+(i-1)*2700,1)=yte_p;
% yte_p_label=yte_p_label_predict;
% save F:\matlab\trial_procedure\study_1\roc_data\elm_cross_subject\task1_session2 yte_p_label y    
    
cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵

% 
[acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
mean_acc(i,j)=(acc_low(i,j)+acc_medium(i,j)+acc_high(i,j))/3;
% 
% % [para_ind]=find(mean_acc==max(mean_acc));%找出最优参数
% 
% acc_low_mean=mean((acc_low'))';
% acc_medium_mean=mean((acc_medium'))';
% acc_high_mean=mean((acc_high'))';
% Acc_mean=mean((Acc'))';  %对角线值除以总值的测试精度
% acc_mean=mean((mean_acc'))';
% sen_mean=mean((sen'))';
% spe_mean=mean((spe'))';
% pre_mean=mean((pre'))';
% npv_mean=mean((npv'))';
% f1_mean=mean((f1'))';
% fnr_mean=mean((fnr'))';
% fpr_mean=mean((fpr'))';
% fdr_mean=mean((fdr'))';
% foe_mean=mean((foe'))';
% mcc_mean=mean((mcc'))';
% bm_mean=mean((bm'))';
% mk_mean=mean((mk'))';
% ka_mean=mean((ka'))';
% 
% acc_low_sd=std((acc_low'))';
% acc_medium_sd=std((acc_medium'))';
% acc_high_sd=std((acc_high'))';
% Acc_sd=std((Acc'))';  %对角线值除以总值的测试精度
% acc_sd=std((mean_acc'))';
% sen_sd=std((sen'))';
% spe_sd=std((spe'))';
% pre_sd=std((pre'))';
% npv_sd=std((npv'))';
% f1_sd=std((f1'))';
% fnr_sd=std((fnr'))';
% fpr_sd=std((fpr'))';
% fdr_sd=std((fdr'))';
% foe_sd=std((foe'))';
% mcc_sd=std((mcc'))';
% bm_sd=std((bm'))';
% mk_sd=std((mk'))';
% ka_sd=std((ka'))';
% 
% acc_TrainingAccuracy=mean((TrainingAccuracy'))';
% acc_TestingAccuracy=mean((TestingAccuracy'))';
% 
% plot_TrainingAccuracy=mean(TrainingAccuracy);
% plot_TestingAccuracy=mean(TestingAccuracy);
% 
% par_ave_mean=mean(mean(TrainingTime'+TestingTime')');
% par_ave_sd=mean(std(TrainingTime'+TestingTime')');
% 
% % per_index=[acc_low_mean,acc_medium_mean,acc_high_mean,Acc_mean,acc_mean,sen_mean,spe_mean,pre_mean,npv_mean,f1_mean,fnr_mean,fpr_mean,fdr_mean,foe_mean,mcc_mean,bm_mean,mk_mean,ka_mean];
% % accuracy=[acc_TrainingAccuracy acc_TestingAccuracy];
% % time=[par_ave_mean par_ave_sd];
% 
% accuracy_par_ave_mean=mean(mean(TestingAccuracy')');
% accuracy_par_ave_sd=mean(std(TestingAccuracy')');
% subject_average_accuracy2=[accuracy_par_ave_mean accuracy_par_ave_sd]; %重复50次实验
% 
% per_index_single2=[acc_low_mean,acc_medium_mean,acc_high_mean,Acc_mean,acc_mean,sen_mean,spe_mean,pre_mean,npv_mean,f1_mean,fnr_mean,fpr_mean,fdr_mean,foe_mean,mcc_mean,bm_mean,mk_mean,ka_mean];
% 
% per_index_ave2_mean=mean([acc_low_mean,acc_medium_mean,acc_high_mean,Acc_mean,acc_mean,sen_mean,spe_mean,pre_mean,npv_mean,f1_mean,fnr_mean,fpr_mean,fdr_mean,foe_mean,mcc_mean,bm_mean,mk_mean,ka_mean])';
% per_index_ave2_sd=mean([acc_low_sd,acc_medium_sd,acc_high_sd,Acc_sd,acc_sd,sen_sd,spe_sd,pre_sd,npv_sd,f1_sd,fnr_sd,fpr_sd,fdr_sd,foe_sd,mcc_sd,bm_sd,mk_sd,ka_sd])';
% per_index_ave2=[per_index_ave2_mean per_index_ave2_sd];
% 
% accuracy2=[acc_TrainingAccuracy acc_TestingAccuracy];
% time2=[par_ave_mean par_ave_sd];
% 
% % save F:\matlab\trial_procedure\study_1\performance_comparison\elm_cross_subject\task1_session2 per_index accuracy time plot_TrainingAccuracy plot_TestingAccuracy
% % save F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task1_session2 per_index_single2 per_index_ave2 accuracy2 time2 subject_average_accuracy2
% save F:\matlab\trial_procedure\study_1\data_analysis\lpp_elm_cross_subject\task1_session2 per_index_single2 per_index_ave2 accuracy2 time2 subject_average_accuracy2
end
end

per_2=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
save F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\task1_session2 per_2

% plot(plot_TrainingAccuracy,'*-r');
% hold on;
% plot(plot_TestingAccuracy,'+-y');
% 
% legend('Training: session 1','Testing: session 1','Training: session 2','Testing: session 2');


%ELM跨被试（任务二6位）
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

% rr=70;
% %进行LPP
% fea =x33;
% options = [];
% options.Metric = 'Euclidean';
% % options.NeighborMode = 'Supervised';
% % options.gnd = y;
% options.ReducedDim=rr;
% W = constructW(fea,options);      
% options.PCARatio = 1;
% [eigvector, eigvalue] = LPP(W, options, fea);
% x33=fea*eigvector;

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

% %进行LPP
% fea =x44;
% options = [];
% options.Metric = 'Euclidean';
% % options.NeighborMode = 'Supervised';
% % options.gnd = y;
% options.ReducedDim=rr;      
% W = constructW(fea,options);      
% options.PCARatio = 1;
% [eigvector, eigvalue] = LPP(W, options, fea);
% x44=fea*eigvector;

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
for j=1
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    
   rand('state',0)  %resets the generator to its initial state

%有放回随机抽样，具有重复性和遗漏性，每次改变训练样本，测试样本不变，针对一个被试测试，训练7个基分类器，最后用投票法，得出最终分类标签，任务一抽样7次
L=7250;
sampleindex=randi(L,7250,1);
xtr_3(1:7250,:)=xtr_3(sampleindex,:);
ytr_3(1:7250,:)=ytr_3(sampleindex,:);
   
    %elm分类器
[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_3,ytr_3, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_3,yte_3,elm_model_1);  % TY: the actual output of the testing data

for zz=1:size(ty_1, 2)
    [~,label_index_actual_1(zz,1)]=max(ty_1(:,zz));
end 
    yte_p1=label_index_actual_1;

%2次
sampleindex=randi(L,7250,1);
xtr_3(1:7250,:)=xtr_3(sampleindex,:);
ytr_3(1:7250,:)=ytr_3(sampleindex,:);
   
    %elm分类器
[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_3,ytr_3, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_3,yte_3,elm_model_1);  % TY: the actual output of the testing data

for zz=1:size(ty_1, 2)
    [~,label_index_actual_1(zz,1)]=max(ty_1(:,zz));
end 
    yte_p2=label_index_actual_1;

%3次
sampleindex=randi(L,7250,1);
xtr_3(1:7250,:)=xtr_3(sampleindex,:);
ytr_3(1:7250,:)=ytr_3(sampleindex,:);
   
    %elm分类器
[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_3,ytr_3, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_3,yte_3,elm_model_1);  % TY: the actual output of the testing data

for zz=1:size(ty_1, 2)
    [~,label_index_actual_1(zz,1)]=max(ty_1(:,zz));
end 
    yte_p3=label_index_actual_1;

    %4次
sampleindex=randi(L,7250,1);
xtr_3(1:7250,:)=xtr_3(sampleindex,:);
ytr_3(1:7250,:)=ytr_3(sampleindex,:);
   
    %elm分类器
[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_3,ytr_3, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_3,yte_3,elm_model_1);  % TY: the actual output of the testing data

for zz=1:size(ty_1, 2)
    [~,label_index_actual_1(zz,1)]=max(ty_1(:,zz));
end 
    yte_p4=label_index_actual_1;
    
    %5次
sampleindex=randi(L,7250,1);
xtr_3(1:7250,:)=xtr_3(sampleindex,:);
ytr_3(1:7250,:)=ytr_3(sampleindex,:);
   
    %elm分类器
[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_3,ytr_3, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_3,yte_3,elm_model_1);  % TY: the actual output of the testing data

for zz=1:size(ty_1, 2)
    [~,label_index_actual_1(zz,1)]=max(ty_1(:,zz));
end 
    yte_p5=label_index_actual_1;
    
y_1_3=[yte_p1 yte_p2 yte_p3 yte_p4 yte_p5];    
%最终标签
yte_p_1=mode(y_1_3')';
   
%    %elm分类器
% [TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_3,ytr_3, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
% [TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_3,yte_3,elm_model_1);  % TY: the actual output of the testing data

%elm_kernel分类器
% [ty] = elm_kernel_m(xtr_1, ytr_1, yte_1, xte_1, 1, 2^5, 'lin_kernel', 2^15);

% for zz=1:size(ty_1, 2)
%     [~,label_index_actual_1(zz,1)]=max(ty_1(:,zz));
% end 
%     yte_p_1=label_index_actual_1;
    
%     yte_p_label_predict_1(1+(i-1)*1450:1450+(i-1)*1450,1)=yte_p_1;
% yte_p_label_1=yte_p_label_predict_1;
% save F:\matlab\trial_procedure\study_1\roc_data\elm_cross_subject\task2_session1 yte_p_label_1 y_1    
    
cte_1(:,:,i,j)=cfmatrix(yte_3,yte_p_1);%计算混淆矩阵

[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
mean_acc_1(i,j)=(acc_low_1(i,j)+acc_medium_1(i,j)+acc_high_1(i,j))/3;
% 
% % [para_ind]=find(mean_acc==max(mean_acc));%找出最优参数
% 
% acc_low_mean_1=mean((acc_low_1'))';
% acc_medium_mean_1=mean((acc_medium_1'))';
% acc_high_mean_1=mean((acc_high_1'))';
% Acc_mean_1=mean((Acc_1'))';  %对角线值除以总值的测试精度
% acc_mean_1=mean((mean_acc_1'))';
% sen_mean_1=mean((sen_1'))';
% spe_mean_1=mean((spe_1'))';
% pre_mean_1=mean((pre_1'))';
% npv_mean_1=mean((npv_1'))';
% f1_mean_1=mean((f1_1'))';
% fnr_mean_1=mean((fnr_1'))';
% fpr_mean_1=mean((fpr_1'))';
% fdr_mean_1=mean((fdr_1'))';
% foe_mean_1=mean((foe_1'))';
% mcc_mean_1=mean((mcc_1'))';
% bm_mean_1=mean((bm_1'))';
% mk_mean_1=mean((mk_1'))';
% ka_mean_1=mean((ka_1'))';
% 
% acc_low_sd_1=std((acc_low_1'))';
% acc_medium_sd_1=std((acc_medium_1'))';
% acc_high_sd_1=std((acc_high_1'))';
% Acc_sd_1=std((Acc_1'))';  %对角线值除以总值的测试精度
% acc_sd_1=std((mean_acc_1'))';
% sen_sd_1=std((sen_1'))';
% spe_sd_1=std((spe_1'))';
% pre_sd_1=std((pre_1'))';
% npv_sd_1=std((npv_1'))';
% f1_sd_1=std((f1_1'))';
% fnr_sd_1=std((fnr_1'))';
% fpr_sd_1=std((fpr_1'))';
% fdr_sd_1=std((fdr_1'))';
% foe_sd_1=std((foe_1'))';
% mcc_sd_1=std((mcc_1'))';
% bm_sd_1=std((bm_1'))';
% mk_sd_1=std((mk_1'))';
% ka_sd_1=std((ka_1'))';
% 
% acc_TrainingAccuracy_1=mean((TrainingAccuracy_1'))';
% acc_TestingAccuracy_1=mean((TestingAccuracy_1'))';
% 
% plot_TrainingAccuracy_1=mean(TrainingAccuracy_1);
% plot_TestingAccuracy_1=mean(TestingAccuracy_1);
% 
% par_ave_mean_1=mean(mean(TrainingTime_1'+TestingTime_1')');
% par_ave_sd_1=mean(std(TrainingTime_1'+TestingTime_1')');
% 
% % per_index_1=[acc_low_mean_1,acc_medium_mean_1,acc_high_mean_1,Acc_mean_1,acc_mean_1,sen_mean_1,spe_mean_1,pre_mean_1,npv_mean_1,f1_mean_1,fnr_mean_1,fpr_mean_1,fdr_mean_1,foe_mean_1,mcc_mean_1,bm_mean_1,mk_mean_1,ka_mean_1];
% % accuracy_1=[acc_TrainingAccuracy_1 acc_TestingAccuracy_1];
% % time_1=[par_ave_mean_1 par_ave_sd_1];
% 
% accuracy_par_ave_mean_1=mean(mean(TestingAccuracy_1')');
% accuracy_par_ave_sd_1=mean(std(TestingAccuracy_1')');
% 
% subject_average_accuracy3=[accuracy_par_ave_mean_1 accuracy_par_ave_sd_1]; %重复50次实验
% 
% per_index_single3=[acc_low_mean_1,acc_medium_mean_1,acc_high_mean_1,Acc_mean_1,acc_mean_1,sen_mean_1,spe_mean_1,pre_mean_1,npv_mean_1,f1_mean_1,fnr_mean_1,fpr_mean_1,fdr_mean_1,foe_mean_1,mcc_mean_1,bm_mean_1,mk_mean_1,ka_mean_1];
% 
% per_index_ave3_mean=mean([acc_low_mean_1,acc_medium_mean_1,acc_high_mean_1,Acc_mean_1,acc_mean_1,sen_mean_1,spe_mean_1,pre_mean_1,npv_mean_1,f1_mean_1,fnr_mean_1,fpr_mean_1,fdr_mean_1,foe_mean_1,mcc_mean_1,bm_mean_1,mk_mean_1,ka_mean_1])';
% per_index_ave3_sd=mean([acc_low_sd_1,acc_medium_sd_1,acc_high_sd_1,Acc_sd_1,acc_sd_1,sen_sd_1,spe_sd_1,pre_sd_1,npv_sd_1,f1_sd_1,fnr_sd_1,fpr_sd_1,fdr_sd_1,foe_sd_1,mcc_sd_1,bm_sd_1,mk_sd_1,ka_sd_1])';
% per_index_ave3=[per_index_ave3_mean per_index_ave3_sd];
% 
% accuracy3=[acc_TrainingAccuracy_1 acc_TestingAccuracy_1];
% time3=[par_ave_mean_1 par_ave_sd_1];
% 
% % save F:\matlab\trial_procedure\study_1\performance_comparison\elm_cross_subject\task2_session1 per_index_1 accuracy_1 time_1 plot_TrainingAccuracy_1 plot_TestingAccuracy_1
% % save F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task2_session1 per_index_single3 per_index_ave3 accuracy3 time3 subject_average_accuracy3
% save F:\matlab\trial_procedure\study_1\data_analysis\lpp_elm_cross_subject\task2_session1 per_index_single3 per_index_ave3 accuracy3 time3 subject_average_accuracy3
end
end

per_3=[acc_low_1,acc_medium_1,acc_high_1,Acc_1,sen_1,spe_1,pre_1,npv_1,f1_1];
save F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\task2_session1 per_3

% subplot(2,1,2);
% plot(plot_TrainingAccuracy_1,'s-g');
% title('(b) ELM: Case 2','FontWeight','bold');
% set(gca,'XTick',0:10:50);
% set(gca,'XTickLabel',{'0','100','200','300','400','500'});
% set(gca,'YTick',0:0.05:1);
% xlabel('Number of hidden neurons','FontWeight','bold');
% ylabel('Accuracy','FontWeight','bold');
% grid on;
% hold on;
% plot(plot_TestingAccuracy_1,'o-b');
% hold on;



%%%%%session2
for j=1
for i=1:K
    
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));

    rand('state',0)  %resets the generator to its initial state

%有放回随机抽样，具有重复性和遗漏性，每次改变训练样本，测试样本不变，针对一个被试测试，训练7个基分类器，最后用投票法，得出最终分类标签，任务一抽样7次
L=7250;
sampleindex=randi(L,7250,1);
xtr_4(1:7250,:)=xtr_4(sampleindex,:);
ytr_4(1:7250,:)=ytr_4(sampleindex,:);
   
    %elm分类器
[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_4,ytr_4, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_4,yte_4,elm_model_1);  % TY: the actual output of the testing data

for zz=1:size(ty_1, 2)
    [~,label_index_actual_1(zz,1)]=max(ty_1(:,zz));
end 
    yte_p1=label_index_actual_1;

%2次
sampleindex=randi(L,7250,1);
xtr_4(1:7250,:)=xtr_4(sampleindex,:);
ytr_4(1:7250,:)=ytr_4(sampleindex,:);
   
    %elm分类器
[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_4,ytr_4, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_4,yte_4,elm_model_1);  % TY: the actual output of the testing data

for zz=1:size(ty_1, 2)
    [~,label_index_actual_1(zz,1)]=max(ty_1(:,zz));
end 
    yte_p2=label_index_actual_1;

%3次
sampleindex=randi(L,7250,1);
xtr_4(1:7250,:)=xtr_4(sampleindex,:);
ytr_4(1:7250,:)=ytr_4(sampleindex,:);
   
    %elm分类器
[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_4,ytr_4, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_4,yte_4,elm_model_1);  % TY: the actual output of the testing data
for zz=1:size(ty_1, 2)
    [~,label_index_actual_1(zz,1)]=max(ty_1(:,zz));
end 
    yte_p3=label_index_actual_1;

    %4次
sampleindex=randi(L,7250,1);
xtr_4(1:7250,:)=xtr_4(sampleindex,:);
ytr_4(1:7250,:)=ytr_4(sampleindex,:);
   
    %elm分类器
[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_4,ytr_4, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_4,yte_4,elm_model_1);  % TY: the actual output of the testing data
for zz=1:size(ty_1, 2)
    [~,label_index_actual_1(zz,1)]=max(ty_1(:,zz));
end 
    yte_p4=label_index_actual_1;
    
    %5次
sampleindex=randi(L,7250,1);
xtr_4(1:7250,:)=xtr_4(sampleindex,:);
ytr_4(1:7250,:)=ytr_4(sampleindex,:);
   
    %elm分类器
[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_4,ytr_4, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_4,yte_4,elm_model_1);  % TY: the actual output of the testing data
for zz=1:size(ty_1, 2)
    [~,label_index_actual_1(zz,1)]=max(ty_1(:,zz));
end 
    yte_p5=label_index_actual_1;
    
y_1_4=[yte_p1 yte_p2 yte_p3 yte_p4 yte_p5];    
%最终标签
yte_p_1=mode(y_1_4')';   
       
    
%     %elm分类器
% [TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_4,ytr_4, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
% [TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_4,yte_4,elm_model_1);  % TY: the actual output of the testing data
% 
% %elm_kernel分类器
% % [ty] = elm_kernel_m(xtr_1, ytr_1, yte_1, xte_1, 1, 2^5, 'lin_kernel', 2^15);
% 
% for zz=1:size(ty_1, 2)
%     [~,label_index_actual_1(zz,1)]=max(ty_1(:,zz));
% end 
%     yte_p_1=label_index_actual_1;
    
%     yte_p_label_predict_1(1+(i-1)*1450:1450+(i-1)*1450,1)=yte_p_1;
% yte_p_label_1=yte_p_label_predict_1;
% save F:\matlab\trial_procedure\study_1\roc_data\elm_cross_subject\task2_session2 yte_p_label_1 y_1    
    
cte_1(:,:,i,j)=cfmatrix(yte_4,yte_p_1);%计算混淆矩阵
% 
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
mean_acc_1(i,j)=(acc_low_1(i,j)+acc_medium_1(i,j)+acc_high_1(i,j))/3;
% 
% % [para_ind]=find(mean_acc==max(mean_acc));%找出最优参数
% 
% acc_low_mean_1=mean((acc_low_1'))';
% acc_medium_mean_1=mean((acc_medium_1'))';
% acc_high_mean_1=mean((acc_high_1'))';
% Acc_mean_1=mean((Acc_1'))';  %对角线值除以总值的测试精度
% acc_mean_1=mean((mean_acc_1'))';
% sen_mean_1=mean((sen_1'))';
% spe_mean_1=mean((spe_1'))';
% pre_mean_1=mean((pre_1'))';
% npv_mean_1=mean((npv_1'))';
% f1_mean_1=mean((f1_1'))';
% fnr_mean_1=mean((fnr_1'))';
% fpr_mean_1=mean((fpr_1'))';
% fdr_mean_1=mean((fdr_1'))';
% foe_mean_1=mean((foe_1'))';
% mcc_mean_1=mean((mcc_1'))';
% bm_mean_1=mean((bm_1'))';
% mk_mean_1=mean((mk_1'))';
% ka_mean_1=mean((ka_1'))';
% 
% acc_low_sd_1=std((acc_low_1'))';
% acc_medium_sd_1=std((acc_medium_1'))';
% acc_high_sd_1=std((acc_high_1'))';
% Acc_sd_1=std((Acc_1'))';  %对角线值除以总值的测试精度
% acc_sd_1=std((mean_acc_1'))';
% sen_sd_1=std((sen_1'))';
% spe_sd_1=std((spe_1'))';
% pre_sd_1=std((pre_1'))';
% npv_sd_1=std((npv_1'))';
% f1_sd_1=std((f1_1'))';
% fnr_sd_1=std((fnr_1'))';
% fpr_sd_1=std((fpr_1'))';
% fdr_sd_1=std((fdr_1'))';
% foe_sd_1=std((foe_1'))';
% mcc_sd_1=std((mcc_1'))';
% bm_sd_1=std((bm_1'))';
% mk_sd_1=std((mk_1'))';
% ka_sd_1=std((ka_1'))';
% 
% acc_TrainingAccuracy_1=mean((TrainingAccuracy_1'))';
% acc_TestingAccuracy_1=mean((TestingAccuracy_1'))';
% 
% plot_TrainingAccuracy_1=mean(TrainingAccuracy_1);
% plot_TestingAccuracy_1=mean(TestingAccuracy_1);
% 
% par_ave_mean_1=mean(mean(TrainingTime_1'+TestingTime_1')');
% par_ave_sd_1=mean(std(TrainingTime_1'+TestingTime_1')');
% 
% % per_index_1=[acc_low_mean_1,acc_medium_mean_1,acc_high_mean_1,Acc_mean_1,acc_mean_1,sen_mean_1,spe_mean_1,pre_mean_1,npv_mean_1,f1_mean_1,fnr_mean_1,fpr_mean_1,fdr_mean_1,foe_mean_1,mcc_mean_1,bm_mean_1,mk_mean_1,ka_mean_1];
% % accuracy_1=[acc_TrainingAccuracy_1 acc_TestingAccuracy_1];
% % time_1=[par_ave_mean_1 par_ave_sd_1];
% 
% accuracy_par_ave_mean_1=mean(mean(TestingAccuracy_1')');
% accuracy_par_ave_sd_1=mean(std(TestingAccuracy_1')');
% subject_average_accuracy4=[accuracy_par_ave_mean_1 accuracy_par_ave_sd_1]; %重复50次实验
% 
% per_index_single4=[acc_low_mean_1,acc_medium_mean_1,acc_high_mean_1,Acc_mean_1,acc_mean_1,sen_mean_1,spe_mean_1,pre_mean_1,npv_mean_1,f1_mean_1,fnr_mean_1,fpr_mean_1,fdr_mean_1,foe_mean_1,mcc_mean_1,bm_mean_1,mk_mean_1,ka_mean_1];
% 
% per_index_ave4_mean=mean([acc_low_mean_1,acc_medium_mean_1,acc_high_mean_1,Acc_mean_1,acc_mean_1,sen_mean_1,spe_mean_1,pre_mean_1,npv_mean_1,f1_mean_1,fnr_mean_1,fpr_mean_1,fdr_mean_1,foe_mean_1,mcc_mean_1,bm_mean_1,mk_mean_1,ka_mean_1])';
% per_index_ave4_sd=mean([acc_low_sd_1,acc_medium_sd_1,acc_high_sd_1,Acc_sd_1,acc_sd_1,sen_sd_1,spe_sd_1,pre_sd_1,npv_sd_1,f1_sd_1,fnr_sd_1,fpr_sd_1,fdr_sd_1,foe_sd_1,mcc_sd_1,bm_sd_1,mk_sd_1,ka_sd_1])';
% per_index_ave4=[per_index_ave4_mean per_index_ave4_sd];
% 
% accuracy4=[acc_TrainingAccuracy_1 acc_TestingAccuracy_1];
% time4=[par_ave_mean_1 par_ave_sd_1];
% 
% 
% % save F:\matlab\trial_procedure\study_1\performance_comparison\elm_cross_subject\task2_session2 per_index_1 accuracy_1 time_1 plot_TrainingAccuracy_1 plot_TestingAccuracy_1
% % save F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task2_session2 per_index_single4 per_index_ave4 accuracy4 time4 subject_average_accuracy4
% save F:\matlab\trial_procedure\study_1\data_analysis\lpp_elm_cross_subject\task2_session2 per_index_single4 per_index_ave4 accuracy4 time4 subject_average_accuracy4

end
end

per_4=[acc_low_1,acc_medium_1,acc_high_1,Acc_1,sen_1,spe_1,pre_1,npv_1,f1_1];
save F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\task2_session2 per_4

%      
% % plot(plot_TrainingAccuracy_1,'*-r');
% % hold on;
% % plot(plot_TestingAccuracy_1,'+-y');

