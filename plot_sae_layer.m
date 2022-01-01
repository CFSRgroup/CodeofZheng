%%
% %取数据
% %sae跨被试（任务一8位）
clc;
clear;
close all;
warning off;
% 
% load F:\matlab\trial_procedure\study_1\features\ex1\s1_1
% x1=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s2_1
% x2=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s3_1
% x3=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s4_1
% x4=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s5_1
% x5=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s6_1
% x6=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s7_1
% x7=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s8_1
% x8=x;
% 
% x11=[x1;x2;x3;x4;x5;x6;x7;x8];
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
% x22=[x1;x2;x3;x4;x5;x6;x7;x8];
% 
% %任务一两个阶段的数据集
% x1=x11;
% x2=x22;
% 
% y=[y;y;y;y;y;y;y;y];
% 
% K=8;
% low=size(find(y==1));
% medium=size(find(y==2));
% high=size(find(y==3));
% 
% [xtr_all_1, xte_all_1] = kfcv(x1,K,'off');
% [ytr_all_1, yte_all_1] = kfcv(y,K,'off');
% 
% [xtr_all_2, xte_all_2] = kfcv(x2,K,'off');
% [ytr_all_2, yte_all_2] = kfcv(y,K,'off');
% 
% %%%%%session1
% for j=1
% for i=1:K
%     xtr_1=cell2mat(xtr_all_1(i,:));
%     xte_1=cell2mat(xte_all_1(i,:));
%     ytr_1=cell2mat(ytr_all_1(i,:));
%     yte_1=cell2mat(yte_all_1(i,:));
%     
%    rand('state',0)  %resets the generator to its initial state
% 
% %  Setup and train a stacked denoising autoencoder (SDAE)
% sae = saesetup([170 80 80 80 80]);
% sae.ae{1}.activation_function       = 'sigm';   %第一隐层
% sae.ae{1}.learningRate              = 0.01;%定义学习率
% % sae.ae{1}.inputZeroMaskedFraction   = 0;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力
% 
% sae.ae{2}.activation_function       = 'sigm';   %第二隐层
% sae.ae{2}.learningRate              = 0.01;%定义学习率
% % sae.ae{2}.inputZeroMaskedFraction   = 0;
% 
% sae.ae{3}.activation_function       = 'sigm';
% sae.ae{3}.learningRate              = 0.01;
% sae.ae{4}.activation_function       = 'sigm';
% sae.ae{4}.learningRate              = 0.01;
% % sae.ae{5}.activation_function       = 'sigm';
% % sae.ae{5}.learningRate              = 0.01;
% % sae.ae{6}.activation_function       = 'sigm';
% % sae.ae{6}.learningRate              = 0.01;
% % sae.ae{7}.activation_function       = 'sigm';
% % sae.ae{7}.learningRate              = 0.01;
% % sae.ae{8}.activation_function       = 'sigm';
% % sae.ae{8}.learningRate              = 0.01;
% % sae.ae{9}.activation_function       = 'sigm';
% % sae.ae{9}.learningRate              = 0.01;
% % sae.ae{10}.activation_function       = 'sigm';
% % sae.ae{10}.learningRate              = 0.01;
% % sae.ae{11}.activation_function       = 'sigm';
% % sae.ae{11}.learningRate              = 0.01;
% % sae.ae{12}.activation_function       = 'sigm';
% % sae.ae{12}.learningRate              = 0.01;
% % sae.ae{13}.activation_function       = 'sigm';
% % sae.ae{13}.learningRate              = 0.01;
% opts.numepochs =   20;%重复迭代次数
% opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
% sae = saetrain(sae, xtr_1, opts);   
% 
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([170 80 80 80 80 3]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 0.01;
% nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
% nn.W{2} = sae.ae{2}.W{1};
% nn.W{3} = sae.ae{3}.W{1};
% nn.W{4} = sae.ae{4}.W{1};
% % nn.W{5} = sae.ae{5}.W{1};
% % nn.W{6} = sae.ae{6}.W{1};
% % nn.W{7} = sae.ae{7}.W{1};
% % nn.W{8} = sae.ae{8}.W{1};
% % nn.W{9} = sae.ae{9}.W{1};
% % nn.W{10} = sae.ae{10}.W{1};
% % nn.W{11} = sae.ae{11}.W{1};
% % nn.W{12} = sae.ae{12}.W{1};
% % nn.W{13} = sae.ae{13}.W{1};
% 
% % Train the FFNN
% opts.numepochs =   20;
% opts.batchsize = 30;
% 
% nn = nntrain(nn, xtr_1, ytr_1, opts);
% labels = nnpredict(nn, xte_1);
% yte_p=labels;
% 
% %测试精度
% TestingAccuracy(i,j)=length(find(yte_p==yte_1))/size(yte_1,1);
% 
% %训练精度
% labels = nnpredict(nn, xtr_1);
% predict_train=labels;
% TrainingAccuracy(i,j)=length(find(predict_train==ytr_1))/size(ytr_1,1);
% 
% TrainingAccuracy_case1_session1=mean(TrainingAccuracy);
% TestingAccuracy_case1_session1=mean(TestingAccuracy);
% 
% save F:\matlab\trial_procedure\study_1\data_analysis\plot_sae_layer\layer4_case1_session1 TrainingAccuracy_case1_session1 TestingAccuracy_case1_session1
% end
% end
% 
% %%%session2
% for j=1
% for i=1:K
%     
%     xtr_2=cell2mat(xtr_all_2(i,:));
%     xte_2=cell2mat(xte_all_2(i,:));
%     ytr_2=cell2mat(ytr_all_2(i,:));
%     yte_2=cell2mat(yte_all_2(i,:));
% 
%    rand('state',0)  %resets the generator to its initial state
%    
% %  Setup and train a stacked denoising autoencoder (SDAE)
% sae = saesetup([170 80 80 80 80]);
% sae.ae{1}.activation_function       = 'sigm';   %第一隐层
% sae.ae{1}.learningRate              = 0.01;%定义学习率
% % sae.ae{1}.learningRate              = 1;%定义学习率
% % sae.ae{1}.inputZeroMaskedFraction   = 0.01;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力
% 
% sae.ae{2}.activation_function       = 'sigm';   %第二隐层
% sae.ae{2}.learningRate              = 0.01;%定义学习率
% % sae.ae{2}.learningRate              = 1;%定义学习率
% % sae.ae{2}.inputZeroMaskedFraction   = 0.01;
% 
% sae.ae{3}.activation_function       = 'sigm';
% sae.ae{3}.learningRate              = 0.01;
% sae.ae{4}.activation_function       = 'sigm';
% sae.ae{4}.learningRate              = 0.01;
% % sae.ae{5}.activation_function       = 'sigm';
% % sae.ae{5}.learningRate              = 0.01;
% % sae.ae{6}.activation_function       = 'sigm';
% % sae.ae{6}.learningRate              = 0.01;
% % sae.ae{7}.activation_function       = 'sigm';
% % sae.ae{7}.learningRate              = 0.01;
% % sae.ae{8}.activation_function       = 'sigm';
% % sae.ae{8}.learningRate              = 0.01;
% % sae.ae{9}.activation_function       = 'sigm';
% % sae.ae{9}.learningRate              = 0.01;
% % sae.ae{10}.activation_function       = 'sigm';
% % sae.ae{10}.learningRate              = 0.01;
% % sae.ae{11}.activation_function       = 'sigm';
% % sae.ae{11}.learningRate              = 0.01;
% % sae.ae{12}.activation_function       = 'sigm';
% % sae.ae{12}.learningRate              = 0.01;
% % sae.ae{13}.activation_function       = 'sigm';
% % sae.ae{13}.learningRate              = 0.01;
% 
% 
% opts.numepochs =   20;%重复迭代次数
% opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
% sae = saetrain(sae, xtr_2, opts);   
% 
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([170 80 80 80 80 3]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 0.01;
% nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
% nn.W{2} = sae.ae{2}.W{1};
% nn.W{3} = sae.ae{3}.W{1};
% nn.W{4} = sae.ae{4}.W{1};
% % nn.W{5} = sae.ae{5}.W{1};
% % nn.W{6} = sae.ae{6}.W{1};
% % nn.W{7} = sae.ae{7}.W{1};
% % nn.W{8} = sae.ae{8}.W{1};
% % nn.W{9} = sae.ae{9}.W{1};
% % nn.W{10} = sae.ae{10}.W{1};
% % nn.W{11} = sae.ae{11}.W{1};
% % nn.W{12} = sae.ae{12}.W{1};
% % nn.W{13} = sae.ae{13}.W{1};
% 
% 
% % Train the FFNN
% opts.numepochs =   20;
% opts.batchsize = 30;
% 
% nn = nntrain(nn, xtr_2, ytr_2, opts);
% labels = nnpredict(nn, xte_2);
% yte_p=labels;
% 
% %测试精度
% TestingAccuracy(i,j)=length(find(yte_p==yte_2))/size(yte_2,1);
% 
% %训练精度
% labels = nnpredict(nn, xtr_2);
% predict_train=labels;
% TrainingAccuracy(i,j)=length(find(predict_train==ytr_2))/size(ytr_2,1);  
% 
% TrainingAccuracy_case1_session2=mean(TrainingAccuracy);
% TestingAccuracy_case1_session2=mean(TestingAccuracy);
% 
% save F:\matlab\trial_procedure\study_1\data_analysis\plot_sae_layer\layer4_case1_session2 TrainingAccuracy_case1_session2 TestingAccuracy_case1_session2
% 
% end
% end
% 
% 
% % %sae跨被试（任务二6位）
% load F:\matlab\trial_procedure\study_1\features\ex2\s1_1
% x1_1=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s2_1
% x2_1=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s3_1
% x3_1=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s4_1
% x4_1=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s5_1
% x5_1=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s6_1
% x6_1=x;
% 
% x33=[x1_1;x2_1;x3_1;x4_1;x5_1;x6_1];
% 
% load F:\matlab\trial_procedure\study_1\features\ex2\s1_2
% x1_1=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s2_2
% x2_1=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s3_2
% x3_1=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s4_2
% x4_1=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s5_2
% x5_1=x;
% load F:\matlab\trial_procedure\study_1\features\ex2\s6_2
% x6_1=x;
% 
% x44=[x1_1;x2_1;x3_1;x4_1;x5_1;x6_1];
% 
% % 任务二两个阶段的数据集
% x3=x33;
% x4=x44;
% 
% y_1=[y;y;y;y;y;y];
% 
% K=6;
% low=size(find(y_1==1));
% medium=size(find(y_1==2));
% high=size(find(y_1==3));
% 
% [xtr_all_3, xte_all_3] = kfcv(x3,K,'off');
% [ytr_all_3, yte_all_3] = kfcv(y_1,K,'off');
% 
% [xtr_all_4, xte_all_4] = kfcv(x4,K,'off');
% [ytr_all_4, yte_all_4] = kfcv(y_1,K,'off');
% 
% %%%%session1
% for j=1
% for i=1:K
%     xtr_3=cell2mat(xtr_all_3(i,:));
%     xte_3=cell2mat(xte_all_3(i,:));
%     ytr_3=cell2mat(ytr_all_3(i,:));
%     yte_3=cell2mat(yte_all_3(i,:));
%     
%    rand('state',0)  %resets the generator to its initial state
%    
% %  Setup and train a stacked denoising autoencoder (SDAE)
% sae = saesetup([170 80 80 80 80]);
% sae.ae{1}.activation_function       = 'sigm';   %第一隐层
% sae.ae{1}.learningRate              = 0.01;%定义学习率
% % sae.ae{1}.learningRate              = 1;%定义学习率
% % sae.ae{1}.inputZeroMaskedFraction   = 0.01;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力
% 
% sae.ae{2}.activation_function       = 'sigm';   %第二隐层
% sae.ae{2}.learningRate              = 0.01;%定义学习率
% % sae.ae{2}.learningRate              = 1;%定义学习率
% % sae.ae{2}.inputZeroMaskedFraction   = 0.01;
% 
% sae.ae{3}.activation_function       = 'sigm';
% sae.ae{3}.learningRate              = 0.01;
% sae.ae{4}.activation_function       = 'sigm';
% sae.ae{4}.learningRate              = 0.01;
% % sae.ae{5}.activation_function       = 'sigm';
% % sae.ae{5}.learningRate              = 0.01;
% % sae.ae{6}.activation_function       = 'sigm';
% % sae.ae{6}.learningRate              = 0.01;
% % sae.ae{7}.activation_function       = 'sigm';
% % sae.ae{7}.learningRate              = 0.01;
% % sae.ae{8}.activation_function       = 'sigm';
% % sae.ae{8}.learningRate              = 0.01;
% % sae.ae{9}.activation_function       = 'sigm';
% % sae.ae{9}.learningRate              = 0.01;
% % sae.ae{10}.activation_function       = 'sigm';
% % sae.ae{10}.learningRate              = 0.01;
% % sae.ae{11}.activation_function       = 'sigm';
% % sae.ae{11}.learningRate              = 0.01;
% % sae.ae{12}.activation_function       = 'sigm';
% % sae.ae{12}.learningRate              = 0.01;
% % sae.ae{13}.activation_function       = 'sigm';
% % sae.ae{13}.learningRate              = 0.01;
% 
% 
% opts.numepochs =   20;%重复迭代次数
% opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
% sae = saetrain(sae, xtr_3, opts);   
% 
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([170 80 80 80 80 3]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 0.01;
% nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
% nn.W{2} = sae.ae{2}.W{1};
% nn.W{3} = sae.ae{3}.W{1};
% nn.W{4} = sae.ae{4}.W{1};
% % nn.W{5} = sae.ae{5}.W{1};
% % nn.W{6} = sae.ae{6}.W{1};
% % nn.W{7} = sae.ae{7}.W{1};
% % nn.W{8} = sae.ae{8}.W{1};
% % nn.W{9} = sae.ae{9}.W{1};
% % nn.W{10} = sae.ae{10}.W{1};
% % nn.W{11} = sae.ae{11}.W{1};
% % nn.W{12} = sae.ae{12}.W{1};
% % nn.W{13} = sae.ae{13}.W{1};
% 
% 
% % Train the FFNN
% opts.numepochs =   20;
% opts.batchsize = 25;
% 
% nn = nntrain(nn, xtr_3, ytr_3, opts);
% labels = nnpredict(nn, xte_3);
% yte_p_1=labels;
% 
% %测试精度
% TestingAccuracy_1(i,j)=length(find(yte_p_1==yte_3))/size(yte_3,1);
% 
% %训练精度
% labels = nnpredict(nn, xtr_3);
% predict_train_1=labels;
% TrainingAccuracy_1(i,j)=length(find(predict_train_1==ytr_3))/size(ytr_3,1);   
% 
% TrainingAccuracy_case2_session1=mean(TrainingAccuracy_1);
% TestingAccuracy_case2_session1=mean(TestingAccuracy_1);
% 
% save F:\matlab\trial_procedure\study_1\data_analysis\plot_sae_layer\layer4_case2_session1 TrainingAccuracy_case2_session1 TestingAccuracy_case2_session1
% 
% end
% end
% 
% 
% % %%%%session2
% for j=1
% for i=1:K
%     
%     xtr_4=cell2mat(xtr_all_4(i,:));
%     xte_4=cell2mat(xte_all_4(i,:));
%     ytr_4=cell2mat(ytr_all_4(i,:));
%     yte_4=cell2mat(yte_all_4(i,:));
% 
%     rand('state',0)  %resets the generator to its initial state
%     
% %  Setup and train a stacked denoising autoencoder (SDAE)
% sae = saesetup([170 80 80 80 80]);
% sae.ae{1}.activation_function       = 'sigm';   %第一隐层
% sae.ae{1}.learningRate              = 0.01;%定义学习率
% % sae.ae{1}.learningRate              = 1;%定义学习率
% % sae.ae{1}.inputZeroMaskedFraction   = 0.01;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力
% 
% sae.ae{2}.activation_function       = 'sigm';   %第二隐层
% sae.ae{2}.learningRate              = 0.01;%定义学习率
% % sae.ae{2}.learningRate              = 1;%定义学习率
% % sae.ae{2}.inputZeroMaskedFraction   = 0.01;
% 
% sae.ae{3}.activation_function       = 'sigm';
% sae.ae{3}.learningRate              = 0.01;
% sae.ae{4}.activation_function       = 'sigm';
% sae.ae{4}.learningRate              = 0.01;
% % sae.ae{5}.activation_function       = 'sigm';
% % sae.ae{5}.learningRate              = 0.01;
% % sae.ae{6}.activation_function       = 'sigm';
% % sae.ae{6}.learningRate              = 0.01;
% % sae.ae{7}.activation_function       = 'sigm';
% % sae.ae{7}.learningRate              = 0.01;
% % sae.ae{8}.activation_function       = 'sigm';
% % sae.ae{8}.learningRate              = 0.01;
% % sae.ae{9}.activation_function       = 'sigm';
% % sae.ae{9}.learningRate              = 0.01;
% % sae.ae{10}.activation_function       = 'sigm';
% % sae.ae{10}.learningRate              = 0.01;
% % sae.ae{11}.activation_function       = 'sigm';
% % sae.ae{11}.learningRate              = 0.01;
% % sae.ae{12}.activation_function       = 'sigm';
% % sae.ae{12}.learningRate              = 0.01;
% % sae.ae{13}.activation_function       = 'sigm';
% % sae.ae{13}.learningRate              = 0.01;
% 
% 
% opts.numepochs =   20;%重复迭代次数
% opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
% sae = saetrain(sae, xtr_4, opts);   
% 
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([170 80 80 80 80 3]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 0.01;
% nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
% nn.W{2} = sae.ae{2}.W{1};
% nn.W{3} = sae.ae{3}.W{1};
% nn.W{4} = sae.ae{4}.W{1};
% % nn.W{5} = sae.ae{5}.W{1};
% % nn.W{6} = sae.ae{6}.W{1};
% % nn.W{7} = sae.ae{7}.W{1};
% % nn.W{8} = sae.ae{8}.W{1};
% % nn.W{9} = sae.ae{9}.W{1};
% % nn.W{10} = sae.ae{10}.W{1};
% % nn.W{11} = sae.ae{11}.W{1};
% % nn.W{12} = sae.ae{12}.W{1};
% % nn.W{13} = sae.ae{13}.W{1};
% 
% 
% % Train the FFNN
% opts.numepochs =   20;
% opts.batchsize = 25;
% 
% nn = nntrain(nn, xtr_4, ytr_4, opts);
% labels = nnpredict(nn, xte_4);
% yte_p_1=labels;
% 
% %测试精度
% TestingAccuracy_1(i,j)=length(find(yte_p_1==yte_4))/size(yte_4,1);
% 
% %训练精度
% labels = nnpredict(nn, xtr_4);
% predict_train_1=labels;
% TrainingAccuracy_1(i,j)=length(find(predict_train_1==ytr_4))/size(ytr_4,1);      
% 
% TrainingAccuracy_case2_session2=mean(TrainingAccuracy_1);
% TestingAccuracy_case2_session2=mean(TestingAccuracy_1);
% 
% save F:\matlab\trial_procedure\study_1\data_analysis\plot_sae_layer\layer4_case2_session2 TrainingAccuracy_case2_session2 TestingAccuracy_case2_session2
% 
% end
% end


%%
%画图
%case1_session1
for i=2:14
eval(['load F:\matlab\trial_procedure\study_1\data_analysis\plot_sae_layer\layer' num2str(i) '_case1_session1'])
train_acc_case1_session1(1,i-1)=TrainingAccuracy_case1_session1;
test_acc_case1_session1(1,i-1)=TestingAccuracy_case1_session1;
end

%case1_session2
for i=2:14
eval(['load F:\matlab\trial_procedure\study_1\data_analysis\plot_sae_layer\layer' num2str(i) '_case1_session2'])
train_acc_case1_session2(1,i-1)=TrainingAccuracy_case1_session2;
test_acc_case1_session2(1,i-1)=TestingAccuracy_case1_session2;
end

%case2_session1
for i=2:14
eval(['load F:\matlab\trial_procedure\study_1\data_analysis\plot_sae_layer\layer' num2str(i) '_case2_session1'])
train_acc_case2_session1(1,i-1)=TrainingAccuracy_case2_session1;
test_acc_case2_session1(1,i-1)=TestingAccuracy_case2_session1;
end

%case2_session2
for i=2:14
eval(['load F:\matlab\trial_procedure\study_1\data_analysis\plot_sae_layer\layer' num2str(i) '_case2_session2'])
train_acc_case2_session2(1,i-1)=TrainingAccuracy_case2_session2;
test_acc_case2_session2(1,i-1)=TestingAccuracy_case2_session2;
end

%case1
subplot(2,1,1);
plot(train_acc_case1_session1,'s-g');
title('(k) SAE: Case 1','FontWeight','bold');
set(gca,'XTick',1:2:13);
set(gca,'XTickLabel',{'2','4','6','8','10','12','14'});
% set(gca,'YTick',0:0.05:1);
xlabel('Number of layers','FontWeight','bold');
ylabel('Accuracy','FontWeight','bold');
grid on;
hold on;
plot(test_acc_case1_session1,'o-b');
hold on;

plot(train_acc_case1_session2,'*-r');
hold on;
plot(test_acc_case1_session2,'+-y');


%case2
subplot(2,1,2);
plot(train_acc_case2_session1,'s-g');
title('(l) SAE: Case 2','FontWeight','bold');
set(gca,'XTick',1:2:13);
set(gca,'XTickLabel',{'2','4','6','8','10','12','14'});
% set(gca,'YTick',0:0.05:1);
xlabel('Number of layers','FontWeight','bold');
ylabel('Accuracy','FontWeight','bold');
grid on;
hold on;
plot(test_acc_case2_session1,'o-b');
hold on;

plot(train_acc_case2_session2,'*-r');
hold on;
plot(test_acc_case2_session2,'+-y');


test_acc_case1=(test_acc_case1_session1+test_acc_case1_session2)./2;
test_acc_case2=(test_acc_case2_session1+test_acc_case2_session2)./2;

[maxVal1,maxInd1] = max(test_acc_case1)
[maxVal2,maxInd2] = max(test_acc_case2)