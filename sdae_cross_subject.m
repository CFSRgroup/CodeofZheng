%sdae跨被试（任务一8位）
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
for j=1:20
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    
   rand('state',0)  %resets the generator to its initial state

%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 100 5*j]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

% sae.ae{3}.activation_function       = 'sigm';
% sae.ae{3}.learningRate              = 0.01;
% % sae.ae{3}.inputZeroMaskedFraction   = 0;


opts.numepochs =   20;%重复迭代次数
opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_1, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 100 5*j 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};
% nn.W{3} = sae.ae{3}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 30;

start_time_train=cputime;  
nn = nntrain(nn, xtr_1, ytr_1, opts);
end_time_train=cputime;
TrainingTime(i,j)=end_time_train-start_time_train;

start_time_test=cputime;
% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_1);
end_time_test=cputime;
yte_p=labels;
TestingTime(i,j)=end_time_test-start_time_test; 

%训练精度
% [er, bad, labels] = nntest(nn, xtr_1, ytr_1);
labels = nnpredict(nn, xtr_1);
predict_train=labels;
TrainingAccuracy(i,j)=length(find(predict_train==ytr_1))/size(ytr_1,1);

% start_time_train=cputime;   
% nb=NaiveBayes.fit(xtr_1,ytr_1); 
% end_time_train=cputime;
% TrainingTime(i,j)=end_time_train-start_time_train;
% 
% start_time_test=cputime;
% % yte_p(1+(i-1)*2700:2700+(i-1)*2700,1)=predict(nb,xte_1);
% yte_p=predict(nb,xte_1);
% end_time_test=cputime;
% TestingTime(i,j)=end_time_test-start_time_test; 
% 
% % yte_p_label=yte_p;
% % save F:\matlab\trial_procedure\study_1\roc_data\nb_cross_subject\task1_session1 yte_p_label y
% 
% %训练精度
% predict_train=predict(nb,xtr_1);
% TrainingAccuracy(i,j)=length(find(predict_train==ytr_1))/size(ytr_1,1);
    
cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵

[acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
mean_acc(i,j)=(acc_low(i,j)+acc_medium(i,j)+acc_high(i,j))/3;
%测试精度
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



% save F:\matlab\trial_procedure\study_1\performance_comparison\nb_cross_subject\task1_session1 per_index accuracy time 
% save F:\matlab\trial_procedure\study_1\data_analysis\sae_cross_subject\task1_session1 per_index_single1 per_index_ave1 accuracy1 time1 subject_average_accuracy1
save F:\matlab\trial_procedure\study_1\stacking\lpp_sdae_para_data\task1_session1\h20 plot_TestingAccuracy plot_TrainingAccuracy


end
end

% subplot(2,1,1);
% % plot(plot_TrainingAccuracy,'s-g');
% C = gradient(plot_TestingAccuracy);
% mesh(plot_TestingAccuracy,C,'FaceLighting','gouraud','LineWidth',0.3);
% title('(k) Testing accuracy of SAE: Case 1','FontWeight','bold');
% set(gca,'XTick',0:5:20);
% set(gca,'XTickLabel',{'0','5','10','15','20'});
% set(gca,'YTick',0:25:100);
% set(gca,'YTickLabel',{'0','25','50','75','100'});
% xlabel('Number of epoches for training','bold');
% ylabel('Number of neurons in each hidden layer','FontWeight','bold');
% grid on;
% hold on;
% % plot(plot_TestingAccuracy,'o-b');
% % hold on;

%%%session2
for j=1:20
for i=1:K
    
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));

   rand('state',0)  %resets the generator to its initial state
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 100 5*j]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_2, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 100 5*j 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 30;

start_time_train=cputime;  
nn = nntrain(nn, xtr_2, ytr_2, opts);
end_time_train=cputime;
TrainingTime(i,j)=end_time_train-start_time_train;

start_time_test=cputime;
% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_2);
end_time_test=cputime;
yte_p=labels;
TestingTime(i,j)=end_time_test-start_time_test; 

%训练精度
% [er, bad, labels] = nntest(nn, xtr_1, ytr_1);
labels = nnpredict(nn, xtr_2);
predict_train=labels;
TrainingAccuracy(i,j)=length(find(predict_train==ytr_2))/size(ytr_2,1);   
      
% start_time_train=cputime;   
% nb=NaiveBayes.fit(xtr_2,ytr_2); 
% end_time_train=cputime;
% TrainingTime(i,j)=end_time_train-start_time_train;
% 
% start_time_test=cputime;
% % yte_p(1+(i-1)*2700:2700+(i-1)*2700,1)=predict(nb,xte_2);
% yte_p=predict(nb,xte_2);
% end_time_test=cputime;
% TestingTime(i,j)=end_time_test-start_time_test; 
% 
% % yte_p_label=yte_p;
% % save F:\matlab\trial_procedure\study_1\roc_data\nb_cross_subject\task1_session2 yte_p_label y
% % 
% % 训练精度
% predict_train=predict(nb,xtr_2);
% TrainingAccuracy(i,j)=length(find(predict_train==ytr_2))/size(ytr_2,1);
    
cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵

[acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
mean_acc(i,j)=(acc_low(i,j)+acc_medium(i,j)+acc_high(i,j))/3;

%测试精度
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


% save F:\matlab\trial_procedure\study_1\performance_comparison\nb_cross_subject\task1_session2 per_index accuracy time 
% save F:\matlab\trial_procedure\study_1\data_analysis\nb_cross_subject\task1_session2 per_index_single2 per_index_ave2 accuracy2 time2 subject_average_accuracy2
% save F:\matlab\trial_procedure\study_1\data_analysis\sae_cross_subject\task1_session2 per_index_single2 per_index_ave2 accuracy2 time2 subject_average_accuracy2
save F:\matlab\trial_procedure\study_1\stacking\lpp_sdae_para_data\task1_session2\h20 plot_TestingAccuracy plot_TrainingAccuracy

end
end
     
% %sae跨被试（任务二6位）
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

% 任务二两个阶段的数据集
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

%%%%session1
for j=1:20
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    
   rand('state',0)  %resets the generator to its initial state
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 100 5*j]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_3, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 100 5*j 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

start_time_train_1=cputime;  
nn = nntrain(nn, xtr_3, ytr_3, opts);
end_time_train_1=cputime;
TrainingTime_1(i,j)=end_time_train_1-start_time_train_1;

start_time_test_1=cputime;
% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_3);
end_time_test_1=cputime;
yte_p_1=labels;
TestingTime_1(i,j)=end_time_test_1-start_time_test_1; 

%训练精度
% [er, bad, labels] = nntest(nn, xtr_1, ytr_1);
labels = nnpredict(nn, xtr_3);
predict_train_1=labels;
TrainingAccuracy_1(i,j)=length(find(predict_train_1==ytr_3))/size(ytr_3,1);   
   
   
% start_time_train_1=cputime;   
% nb_1=NaiveBayes.fit(xtr_3,ytr_3); 
% end_time_train_1=cputime;
% TrainingTime_1(i,j)=end_time_train_1-start_time_train_1;
% 
% start_time_test_1=cputime;
% % yte_p_1(1+(i-1)*1450:1450+(i-1)*1450,1)=predict(nb_1,xte_3);
% yte_p_1=predict(nb_1,xte_3);
% end_time_test_1=cputime;
% TestingTime_1(i,j)=end_time_test_1-start_time_test_1; 
% 
% % yte_p_label_1=yte_p_1;
% % save F:\matlab\trial_procedure\study_1\roc_data\nb_cross_subject\task2_session1 yte_p_label_1 y_1
% % 
% % 训练精度
% predict_train_1=predict(nb_1,xtr_3);
% TrainingAccuracy_1(i,j)=length(find(predict_train_1==ytr_3))/size(ytr_3,1);
    
cte_1(:,:,i,j)=cfmatrix(yte_3,yte_p_1);%计算混淆矩阵

[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
mean_acc_1(i,j)=(acc_low_1(i,j)+acc_medium_1(i,j)+acc_high_1(i,j))/3;

%测试精度
TestingAccuracy_1(i,j)=Acc_1(i,j);
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

acc_TrainingAccuracy_1=mean((TrainingAccuracy_1'))';
acc_TestingAccuracy_1=mean((TestingAccuracy_1'))';

plot_TrainingAccuracy_1=mean(TrainingAccuracy_1);
plot_TestingAccuracy_1=mean(TestingAccuracy_1);

par_ave_mean_1=mean(mean(TrainingTime_1'+TestingTime_1')');
par_ave_sd_1=mean(std(TrainingTime_1'+TestingTime_1')');

accuracy_par_ave_mean_1=mean(mean(TestingAccuracy_1')');
accuracy_par_ave_sd_1=mean(std(TestingAccuracy_1')');

subject_average_accuracy3=[accuracy_par_ave_mean_1 accuracy_par_ave_sd_1]; %重复50次实验

per_index_single3=[acc_low_mean_1,acc_medium_mean_1,acc_high_mean_1,Acc_mean_1,acc_mean_1,sen_mean_1,spe_mean_1,pre_mean_1,npv_mean_1,f1_mean_1,fnr_mean_1,fpr_mean_1,fdr_mean_1,foe_mean_1,mcc_mean_1,bm_mean_1,mk_mean_1,ka_mean_1];

per_index_ave3_mean=mean([acc_low_mean_1,acc_medium_mean_1,acc_high_mean_1,Acc_mean_1,acc_mean_1,sen_mean_1,spe_mean_1,pre_mean_1,npv_mean_1,f1_mean_1,fnr_mean_1,fpr_mean_1,fdr_mean_1,foe_mean_1,mcc_mean_1,bm_mean_1,mk_mean_1,ka_mean_1])';
per_index_ave3_sd=mean([acc_low_sd_1,acc_medium_sd_1,acc_high_sd_1,Acc_sd_1,acc_sd_1,sen_sd_1,spe_sd_1,pre_sd_1,npv_sd_1,f1_sd_1,fnr_sd_1,fpr_sd_1,fdr_sd_1,foe_sd_1,mcc_sd_1,bm_sd_1,mk_sd_1,ka_sd_1])';
per_index_ave3=[per_index_ave3_mean per_index_ave3_sd];

accuracy3=[acc_TrainingAccuracy_1 acc_TestingAccuracy_1];
time3=[par_ave_mean_1 par_ave_sd_1];

% save F:\matlab\trial_procedure\study_1\performance_comparison\nb_cross_subject\task2_session1 per_index_1 accuracy_1 time_1 
% save F:\matlab\trial_procedure\study_1\data_analysis\nb_cross_subject\task2_session1 per_index_single3 per_index_ave3 accuracy3 time3 subject_average_accuracy3
% save F:\matlab\trial_procedure\study_1\data_analysis\sae_cross_subject\task2_session1 per_index_single3 per_index_ave3 accuracy3 time3 subject_average_accuracy3
save F:\matlab\trial_procedure\study_1\stacking\lpp_sdae_para_data\task2_session1\h20 plot_TestingAccuracy_1 plot_TrainingAccuracy_1
end
end

% %%%%session2
for j=1:20
for i=1:K
    
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));

    rand('state',0)  %resets the generator to its initial state
    
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 100 5*j]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_4, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 100 5*j 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

start_time_train_1=cputime;  
nn = nntrain(nn, xtr_4, ytr_4, opts);
end_time_train_1=cputime;
TrainingTime_1(i,j)=end_time_train_1-start_time_train_1;

start_time_test_1=cputime;
% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_4);
end_time_test_1=cputime;
yte_p_1=labels;
TestingTime_1(i,j)=end_time_test_1-start_time_test_1; 

%训练精度
% [er, bad, labels] = nntest(nn, xtr_1, ytr_1);
labels = nnpredict(nn, xtr_4);
predict_train_1=labels;
TrainingAccuracy_1(i,j)=length(find(predict_train_1==ytr_4))/size(ytr_4,1);    
    
    
% start_time_train_1=cputime;   
% nb_1=NaiveBayes.fit(xtr_4,ytr_4); 
% end_time_train_1=cputime;
% TrainingTime_1(i,j)=end_time_train_1-start_time_train_1;
% 
% start_time_test_1=cputime;
% % yte_p_1(1+(i-1)*1450:1450+(i-1)*1450,1)=predict(nb_1,xte_4);
% yte_p_1=predict(nb_1,xte_4);
% end_time_test_1=cputime;
% TestingTime_1(i,j)=end_time_test_1-start_time_test_1; 
% 
% % yte_p_label_1=yte_p_1;
% % save F:\matlab\trial_procedure\study_1\roc_data\nb_cross_subject\task2_session2 yte_p_label_1 y_1
% % 
% % 训练精度
% predict_train_1=predict(nb_1,xtr_4);
% TrainingAccuracy_1(i,j)=length(find(predict_train_1==ytr_4))/size(ytr_4,1);
    
cte_1(:,:,i,j)=cfmatrix(yte_4,yte_p_1);%计算混淆矩阵

[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
mean_acc_1(i,j)=(acc_low_1(i,j)+acc_medium_1(i,j)+acc_high_1(i,j))/3;

%测试精度
TestingAccuracy_1(i,j)=Acc_1(i,j);
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

acc_TrainingAccuracy_1=mean((TrainingAccuracy_1'))';
acc_TestingAccuracy_1=mean((TestingAccuracy_1'))';

plot_TrainingAccuracy_1=mean(TrainingAccuracy_1);
plot_TestingAccuracy_1=mean(TestingAccuracy_1);

par_ave_mean_1=mean(mean(TrainingTime_1'+TestingTime_1')');
par_ave_sd_1=mean(std(TrainingTime_1'+TestingTime_1')');

accuracy_par_ave_mean_1=mean(mean(TestingAccuracy_1')');
accuracy_par_ave_sd_1=mean(std(TestingAccuracy_1')');

subject_average_accuracy4=[accuracy_par_ave_mean_1 accuracy_par_ave_sd_1]; %重复50次实验

per_index_single4=[acc_low_mean_1,acc_medium_mean_1,acc_high_mean_1,Acc_mean_1,acc_mean_1,sen_mean_1,spe_mean_1,pre_mean_1,npv_mean_1,f1_mean_1,fnr_mean_1,fpr_mean_1,fdr_mean_1,foe_mean_1,mcc_mean_1,bm_mean_1,mk_mean_1,ka_mean_1];

per_index_ave4_mean=mean([acc_low_mean_1,acc_medium_mean_1,acc_high_mean_1,Acc_mean_1,acc_mean_1,sen_mean_1,spe_mean_1,pre_mean_1,npv_mean_1,f1_mean_1,fnr_mean_1,fpr_mean_1,fdr_mean_1,foe_mean_1,mcc_mean_1,bm_mean_1,mk_mean_1,ka_mean_1])';
per_index_ave4_sd=mean([acc_low_sd_1,acc_medium_sd_1,acc_high_sd_1,Acc_sd_1,acc_sd_1,sen_sd_1,spe_sd_1,pre_sd_1,npv_sd_1,f1_sd_1,fnr_sd_1,fpr_sd_1,fdr_sd_1,foe_sd_1,mcc_sd_1,bm_sd_1,mk_sd_1,ka_sd_1])';
per_index_ave4=[per_index_ave4_mean per_index_ave4_sd];

accuracy4=[acc_TrainingAccuracy_1 acc_TestingAccuracy_1];
time4=[par_ave_mean_1 par_ave_sd_1];

% save F:\matlab\trial_procedure\study_1\performance_comparison\nb_cross_subject\task2_session2 per_index_1 accuracy_1 time_1 
% save F:\matlab\trial_procedure\study_1\data_analysis\nb_cross_subject\task2_session2 per_index_single4 per_index_ave4 accuracy4 time4 subject_average_accuracy4
% save F:\matlab\trial_procedure\study_1\data_analysis\sae_cross_subject\task2_session2 per_index_single4 per_index_ave4 accuracy4 time4 subject_average_accuracy4
save F:\matlab\trial_procedure\study_1\stacking\lpp_sdae_para_data\task2_session2\h20 plot_TestingAccuracy_1 plot_TrainingAccuracy_1

end
end