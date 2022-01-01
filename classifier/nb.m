clc;
clear;
close all;

for pp=1:8
eval(['load D:\exp\result1-8\S1-8data\s1' num2str(pp) ''])
load D:\exp\label1;
nb1=fitcnb(train_x,training_label);
nb1.Prior=[0.2,0.8];
predict_label=predict(nb1,test_x);
accuracy=length(find(predict_label==testing_label))/length(testing_label);


eval(['save D:\exp\nb_accuracy\\0.9\s1' num2str(pp) ''])
end