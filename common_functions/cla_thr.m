%3类混淆矩阵分类器性能指标评估
%author:郑展鹏
%time:2019/4/4

%sensitivity,recall,true positive rate(TPR)  [sen]
%precision,positive predictive value(PPV)  阳性预测值,查准率  [pre]
%accuracy(ACC)  [acc]
%F1 score  [f1]

function [sen_low,sen_medium,sen_high,pre_low,pre_medium,pre_high,acc,f1_low,f1_medium,f1_high] = cla_thr(cfmatrix) 
A=cfmatrix(1,1);
B=cfmatrix(1,2);
C=cfmatrix(1,3);
D=cfmatrix(2,1);
E=cfmatrix(2,2);
F=cfmatrix(2,3);
G=cfmatrix(3,1);
H=cfmatrix(3,2);
I=cfmatrix(3,3);

sen_low=A/(A+D+G);
sen_medium=E/(B+E+H);
sen_high=I/(C+F+I);
pre_low=A/(A+B+C);
pre_medium=E/(D+E+F);
pre_high=I/(G+H+I);
acc=(A+B+C)/(A+B+C+D+E+F+G+H+I);
f1_low=(2*pre_low*sen_low)/(pre_low+sen_low);
f1_medium=(2*pre_medium*sen_medium)/(pre_medium+sen_medium);
f1_high=(2*pre_high*sen_high)/(pre_high+sen_high);

if isnan(pre_low)==1
    pre_low=0;
end

if isnan(pre_medium)==1
    pre_medium=0;
end

if isnan(pre_high)==1
    pre_high=0;
end

if isnan(f1_low)==1
    f1_low=0;
end

if isnan(f1_medium)==1
    f1_medium=0;
end

if isnan(f1_high)==1
    f1_high=0;
end