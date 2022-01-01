% function test_example_SAE
load s1_1expdata;
sum(train_y==NaN)
train_x = zscore(train_x(:,1:10));
test_x  = zscore(test_x(:,1:10));
train_y = double(train_y);
test_y  = double(test_y);

%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
% rand('state',0)
% sae = saesetup([10 5]);
% sae.ae{1}.activation_function       = 'sigm';
% sae.ae{1}.learningRate              = 0.1;%定义学习率
% sae.ae{1}.inputZeroMaskedFraction   = 0.01;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力
% % sae.ae{2}.activation_function       = 'sigm';
% % sae.ae{2}.learningRate              = 1;%定义学习率
% % sae.ae{2}.inputZeroMaskedFraction   = 0.01;
% % sae.ae{3}.activation_function       = 'sigm';
% % sae.ae{3}.learningRate              = 1;
% % sae.ae{3}.inputZeroMaskedFraction   = 0.01;
% % sae.ae{4}.activation_function       = 'sigm';
% % sae.ae{4}.learningRate              = 1;
% % sae.ae{4}.inputZeroMaskedFraction   = 0.01;
% % sae.ae{5}.activation_function       = 'sigm';
% % sae.ae{5}.learningRate              = 1;
% % sae.ae{5}.inputZeroMaskedFraction   = 0.01;
% 
% opts.numepochs =   100;%重复迭代次数
% opts.batchsize = 100;%数据块大小--每一块用于一次梯度下降算法
% sae = saetrain(sae, train_x, opts);



% Use the SDAE to initialize a FFNN
nn = nnsetup([10 5 2]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
% nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
% nn.W{2} = sae.ae{2}.W{1};
% nn.W{3} = sae.ae{3}.W{1};
% nn.W{4} = sae.ae{4}.W{1};
% nn.W{5} = sae.ae{5}.W{1};

% Train the FFNN
opts.numepochs =   100;
opts.batchsize = 10;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);%测试
% assert(er < 2, 'Too big error');
