clear all; close all; clc;
load C:\users\hp\desktop\demo\dataset\stacking\sdae\dataset.mat
tic
rng(0);
sae = saesetup([137 110 20]);            
sae.ae{1}.activation_function  = 'sigm'; 
sae.ae{1}.learningRate = 1;               
sae.ae{1}.inputZeroMaskedFraction = 0.2;  
sae.ae{2}.activation_function = 'sigm';  
sae.ae{2}.learningRate = 1;  
sae.ae{2}.inputZeroMaskedFraction = 0.2;  
opts.numepochs = 10;        
opts.batchsize = 32;      
nn = nnsetup([137 110 20 2]);  
nn.activation_function = 'sigm';  
nn.learningRate = 1;  
nn.W{1} = sae.ae{1}.W{1};  
nn.W{2} = sae.ae{2}.W{1};  
opts.numepochs = 20; 
opts.batchsize =32;  
nn = nntrain(nn,train_x,train_y, opts);
[er, bad] = nntest(nn, test_x, test_y); 
eb=1-er;
str = sprintf('测试精度 is: %f',eb); 
disp(str)
 toc
disp(['运行时间: ',num2str(toc)]);