clc;
clear;
close all;

for pp=1:8
%%
%载入数据
%eval(['load D:\exp\features\ex1\s' num2str(pp) '_1'])
eval(['load F:\matlab\trial_procedure\study_1\features\ex1\s' num2str(pp) '_1'])
x1=x;
y1=y;
%eval(['load D:\exp\features\ex1\s' num2str(pp) '_2'])
eval(['load F:\matlab\trial_procedure\study_1\features\ex1\s' num2str(pp) '_2'])
x2=x;
y2=y;
x=[x1;x2];  
y=[y1;y2];
%抽取训练数据
x_train_1=x([1:1150 1551:2700],:);
x_train_2=x([1+2700:1550+2700 1551+2700:2700+2700],:);
y_train_1=y([1:1150 1551:2700],:);
y_train_2=y([1+2700:1550+2700 1551+2700:2700+2700],:);
%抽取测试数据
x_test_1=x(1151:1550,:);
x_test_2=x(1151+2700:1550+2700,:);
y_test_1=y(1151:1550,:);
y_test_2=y(1151+2700:1550+2700,:);

train_x=[x_train_1;x_train_2];
train_y=[y_train_1;y_train_2];
test_x=[x_test_1;x_test_2];
test_y=[y_test_1;y_test_2];
% train_y=ind2vec(train_y')';
% test_y=ind2vec(test_y')';
%train_y(train_y==2)=0;
%test_y(test_y==2)=0;

train_x = double(train_x);
test_x  = double(test_x);
train_y = double(train_y);
test_y  = double(test_y);
%进行逻辑回归
a =glmfit(train_x,train_y,'binomial', 'link', 'logit');  
logitFit = glmval(a,test_x, 'logit'); 
logitFit(logitFit<0.5)=1;
logitFit(logitFit>0.5)=3;
accuracy=length(find(logitFit == test_y))/length(test_y)*100
%eval(['save D:\exp\logistic_results\s' num2str(pp) ' accuracy '])
eval(['save F:\matlab\trial_procedure\study_1\baseline_performance\logistic_results\s' num2str(pp) ' accuracy '])
end
