clc;
clear;
close all;

for pp=1:8
%%
%载入数据
eval(['load D:\exp\features\ex1\s' num2str(pp) '_1'])
x1=x;
y1=y;
eval(['load D:\exp\features\ex1\s' num2str(pp) '_2'])
x2=x;
y2=y;
x=[x1;x2];
y=[y1;y2];
%抽取训练数据
x_train_1=x([1:250 651:1800],:);
x_train_2=x([1+1800:250+1800 651+1800:1800+1800],:);
y_train_1=y([1:250 651:1800],:);
y_train_2=y([1+1800:250+1800 651+1800:1800+1800],:);
%抽取测试数据
x_test_1=x(251:650,:);
x_test_2=x(251+1800:650+1800,:);
y_test_1=y(251:650,:);
y_test_2=y(251+1800:650+1800,:);

train_x=[x_train_1;x_train_2];
train_y=[y_train_1;y_train_2];
test_x=[x_test_1;x_test_2];
test_y=[y_test_1;y_test_2];
% train_y=ind2vec(train_y')';
% test_y=ind2vec(test_y')';
train_y(train_y==2)=0;
test_y(test_y==2)=0;
%进行KNN
for i=1:100
mdl = ClassificationKNN.fit(train_x,train_y,'NumNeighbors',i);
predict_label   =       predict(mdl, test_x);
accuracy        =       length(find(predict_label == test_y))/length(test_y)*100;
eval(['save D:\exp\KNN_results\s' num2str(pp) ' accuracy '])
end 
end