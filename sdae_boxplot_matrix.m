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
options.PCARatio = 0.8;
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
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 10 85]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_1, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 10 85 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_1, ytr_1, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_1);
yte_p_1=labels;

%2次
sampleindex=randi(L,18900,1);
xtr_1(1:18900,:)=xtr_1(sampleindex,:);
ytr_1(1:18900,:)=ytr_1(sampleindex,:);   
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 10 85]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_1, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 10 85 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_1, ytr_1, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_1);
yte_p_2=labels;

%3次
sampleindex=randi(L,18900,1);
xtr_1(1:18900,:)=xtr_1(sampleindex,:);
ytr_1(1:18900,:)=ytr_1(sampleindex,:);   
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 10 85]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_1, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 10 85 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_1, ytr_1, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_1);
yte_p_3=labels;

%4次
sampleindex=randi(L,18900,1);
xtr_1(1:18900,:)=xtr_1(sampleindex,:);
ytr_1(1:18900,:)=ytr_1(sampleindex,:);   
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 10 85]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_1, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 10 85 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_1, ytr_1, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_1);
yte_p_4=labels;

%5次
sampleindex=randi(L,18900,1);
xtr_1(1:18900,:)=xtr_1(sampleindex,:);
ytr_1(1:18900,:)=ytr_1(sampleindex,:);   
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 10 85]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_1, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 10 85 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_1, ytr_1, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_1);
yte_p_5=labels;

%6次
sampleindex=randi(L,18900,1);
xtr_1(1:18900,:)=xtr_1(sampleindex,:);
ytr_1(1:18900,:)=ytr_1(sampleindex,:);   
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 10 85]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_1, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 10 85 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_1, ytr_1, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_1);
yte_p_6=labels;

%7次
sampleindex=randi(L,18900,1);
xtr_1(1:18900,:)=xtr_1(sampleindex,:);
ytr_1(1:18900,:)=ytr_1(sampleindex,:);   
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 10 85]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_1, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 10 85 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_1, ytr_1, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_1);
yte_p_7=labels;

y_1_1=[yte_p_1 yte_p_2 yte_p_3 yte_p_4 yte_p_5 yte_p_6 yte_p_7];    
%最终标签
yte_p=mode(y_1_1')';
    
cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵

[acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
mean_acc(i,j)=(acc_low(i,j)+acc_medium(i,j)+acc_high(i,j))/3;

end
end

per_1=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
save F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\task1_session1 per_1

%%%session2
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
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 10 85]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_2, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 10 85 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_2, ytr_2, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_2);
yte_p_1=labels;

%2次
sampleindex=randi(L,18900,1);
xtr_2(1:18900,:)=xtr_2(sampleindex,:);
ytr_2(1:18900,:)=ytr_2(sampleindex,:);   
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 10 85]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_2, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 10 85 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_2, ytr_2, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_2);
yte_p_2=labels;

%3次
sampleindex=randi(L,18900,1);
xtr_2(1:18900,:)=xtr_2(sampleindex,:);
ytr_2(1:18900,:)=ytr_2(sampleindex,:);   
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 10 85]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_2, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 10 85 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_2, ytr_2, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_2);
yte_p_3=labels;

%4次
sampleindex=randi(L,18900,1);
xtr_2(1:18900,:)=xtr_2(sampleindex,:);
ytr_2(1:18900,:)=ytr_2(sampleindex,:);   
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 10 85]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_2, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 10 85 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_2, ytr_2, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_2);
yte_p_4=labels;

%5次
sampleindex=randi(L,18900,1);
xtr_2(1:18900,:)=xtr_2(sampleindex,:);
ytr_2(1:18900,:)=ytr_2(sampleindex,:);   
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 10 85]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_2, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 10 85 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_2, ytr_2, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_2);
yte_p_5=labels;

%6次
sampleindex=randi(L,18900,1);
xtr_2(1:18900,:)=xtr_2(sampleindex,:);
ytr_2(1:18900,:)=ytr_2(sampleindex,:);   
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 10 85]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_2, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 10 85 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_2, ytr_2, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_2);
yte_p_6=labels;

%7次
sampleindex=randi(L,18900,1);
xtr_2(1:18900,:)=xtr_2(sampleindex,:);
ytr_2(1:18900,:)=ytr_2(sampleindex,:);   
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 10 85]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_2, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 10 85 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_2, ytr_2, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_2);
yte_p_7=labels;

y_1_2=[yte_p_1 yte_p_2 yte_p_3 yte_p_4 yte_p_5 yte_p_6 yte_p_7];    
%最终标签
yte_p=mode(y_1_2')';

cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵

[acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
mean_acc(i,j)=(acc_low(i,j)+acc_medium(i,j)+acc_high(i,j))/3;
end
end

per_2=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
save F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\task1_session2 per_2

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
options.PCARatio = 0.8;
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
options.PCARatio = 0.8;
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
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 15 5]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_3, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 15 5 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_3, ytr_3, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_3);
yte_p1=labels;

%2次
sampleindex=randi(L,7250,1);
xtr_3(1:7250,:)=xtr_3(sampleindex,:);
ytr_3(1:7250,:)=ytr_3(sampleindex,:);   
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 15 5]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_3, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 15 5 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_3, ytr_3, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_3);
yte_p2=labels;
    
%3次
sampleindex=randi(L,7250,1);
xtr_3(1:7250,:)=xtr_3(sampleindex,:);
ytr_3(1:7250,:)=ytr_3(sampleindex,:);   
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 15 5]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_3, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 15 5 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_3, ytr_3, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_3);
yte_p3=labels;

%4次
sampleindex=randi(L,7250,1);
xtr_3(1:7250,:)=xtr_3(sampleindex,:);
ytr_3(1:7250,:)=ytr_3(sampleindex,:);   
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 15 5]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_3, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 15 5 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_3, ytr_3, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_3);
yte_p4=labels;

%5次
sampleindex=randi(L,7250,1);
xtr_3(1:7250,:)=xtr_3(sampleindex,:);
ytr_3(1:7250,:)=ytr_3(sampleindex,:);   
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 15 5]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_3, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 15 5 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_3, ytr_3, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_3);
yte_p5=labels;

y_1_3=[yte_p1 yte_p2 yte_p3 yte_p4 yte_p5];    
%最终标签
yte_p_1=mode(y_1_3')';

cte_1(:,:,i,j)=cfmatrix(yte_3,yte_p_1);%计算混淆矩阵

[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
mean_acc_1(i,j)=(acc_low_1(i,j)+acc_medium_1(i,j)+acc_high_1(i,j))/3;

end
end

per_3=[acc_low_1,acc_medium_1,acc_high_1,Acc_1,sen_1,spe_1,pre_1,npv_1,f1_1];
save F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\task2_session1 per_3

% %%%%session2
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
    
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 15 5]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_4, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 15 5 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_4, ytr_4, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_4);
yte_p1=labels;

%2次
sampleindex=randi(L,7250,1);
xtr_4(1:7250,:)=xtr_4(sampleindex,:);
ytr_4(1:7250,:)=ytr_4(sampleindex,:);    
    
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 15 5]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_4, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 15 5 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_4, ytr_4, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_4);
yte_p2=labels;

%3次
sampleindex=randi(L,7250,1);
xtr_4(1:7250,:)=xtr_4(sampleindex,:);
ytr_4(1:7250,:)=ytr_4(sampleindex,:);    
    
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 15 5]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_4, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 15 5 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_4, ytr_4, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_4);
yte_p3=labels;

%4次
sampleindex=randi(L,7250,1);
xtr_4(1:7250,:)=xtr_4(sampleindex,:);
ytr_4(1:7250,:)=ytr_4(sampleindex,:);    
    
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 15 5]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_4, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 15 5 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_4, ytr_4, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_4);
yte_p4=labels;

%5次
sampleindex=randi(L,7250,1);
xtr_4(1:7250,:)=xtr_4(sampleindex,:);
ytr_4(1:7250,:)=ytr_4(sampleindex,:);    
    
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([70 15 5]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.2;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
sae.ae{2}.inputZeroMaskedFraction   = 0.2;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_4, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([70 15 5 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_4, ytr_4, opts);

% [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
labels = nnpredict(nn, xte_4);
yte_p5=labels;

y_1_4=[yte_p1 yte_p2 yte_p3 yte_p4 yte_p5];    
%最终标签
yte_p_1=mode(y_1_4')';   

cte_1(:,:,i,j)=cfmatrix(yte_4,yte_p_1);%计算混淆矩阵

[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
mean_acc_1(i,j)=(acc_low_1(i,j)+acc_medium_1(i,j)+acc_high_1(i,j))/3;

end
end

per_4=[acc_low_1,acc_medium_1,acc_high_1,Acc_1,sen_1,spe_1,pre_1,npv_1,f1_1];
save F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\task2_session2 per_4
