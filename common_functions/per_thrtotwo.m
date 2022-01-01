%混淆矩阵分类器性能指标评估
%三分类转换为二分类
%author:郑展鹏
%time:2019/4/8
%notes:P:数据中真正例个数  N;数据中真反例个数  TP:真正例  TN;真反例  FP;假正例  FN;假反例
%多类结果编码为两种OFS类别，即正常OFS（negative class）和异常OFS（positive class）。
%LMW编码为负类（表明正常OFS），而MMW和HMW类是positive class（表明异常OFS）。

%sensitivity,recall,true positive rate(TPR)  [sen]
%specificity,true negative rate(TNR)  [spe]
%precision,positive predictive value(PPV)  阳性预测值,查准率  [pre]
%negative predictive value(NPV)  阴性预测值  [npv]
%miss rate or false negative rate(FNR)  [fnr]
%fall-out or false positive rate(FPR)  [fpr]
%false discovery rate(FDR)  [fdr]
%false omission rate(FOR)  [foe]
%accuracy(ACC)  [acc]
%F1 score  [f1]
%Matthews correlation coefficient(MCC)  马修斯相关系数  [mcc]
%Informedness or Bookmaker Informedness(BM)  [bm]
%Markedness(MK)  [mk]
%kappa系数 一致性检验

function [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cfmatrix) 
TN=cfmatrix(1,1);
FN=cfmatrix(1,2)+cfmatrix(1,3);
FP=cfmatrix(2,1)+cfmatrix(3,1);
TP=cfmatrix(2,2)+cfmatrix(2,3)+cfmatrix(3,2)+cfmatrix(3,3);
target_low=cfmatrix(1,1);
target_medium=cfmatrix(2,2);
target_high=cfmatrix(3,3);
low=cfmatrix(1,1)+cfmatrix(2,1)+cfmatrix(3,1);
medium=cfmatrix(1,2)+cfmatrix(2,2)+cfmatrix(3,2);
high=cfmatrix(1,3)+cfmatrix(2,3)+cfmatrix(3,3);
p0=(cfmatrix(1,1)+cfmatrix(2,2)+cfmatrix(3,3))/(cfmatrix(1,1)+cfmatrix(1,2)+cfmatrix(1,3)+cfmatrix(2,1)+cfmatrix(2,2)+cfmatrix(2,3)+cfmatrix(3,1)+cfmatrix(3,2)+cfmatrix(3,3));
pc=((cfmatrix(1,1)+cfmatrix(1,2)+cfmatrix(1,3))*(cfmatrix(1,1)+cfmatrix(2,1)+cfmatrix(3,1))+(cfmatrix(2,1)+cfmatrix(2,2)+cfmatrix(2,3))*(cfmatrix(1,2)+cfmatrix(2,2)+cfmatrix(3,2))+(cfmatrix(3,1)+cfmatrix(3,2)+cfmatrix(3,3))*(cfmatrix(1,3)+cfmatrix(2,3)+cfmatrix(3,3)))/(cfmatrix(1,1)+cfmatrix(1,2)+cfmatrix(1,3)+cfmatrix(2,1)+cfmatrix(2,2)+cfmatrix(2,3)+cfmatrix(3,1)+cfmatrix(3,2)+cfmatrix(3,3))^2;

acc_low=target_low/low;
acc_medium=target_medium/medium;
acc_high=target_high/high;
Acc=(target_low+target_medium+target_high)/(TN+FN+FP+TP);
sen=TP/(TP+FN);
spe=TN/(TN+FP);
pre=TP/(TP+FP); 
npv=TN/(TN+FN); 
f1=(2*TP)/(2*TP+FP+FN);
fnr=FN/(FN+TP);
fpr=FP/(FP+TN);
fdr=FP/(FP+TP);
foe=FN/(FN+TN);
mcc=(TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
bm=sen+spe-1;
mk=pre+npv-1;
ka=(p0-pc)/(1-pc);


% mean_acc=(acc_low+acc_medium+acc_high)/3;

if isnan(pre)==1
    pre=0;
end

if isnan(npv)==1
    npv=0;
end

if isnan(f1)==1
    f1=0;
end

if isnan(fnr)==1
    fnr=0;
end

if isnan(fpr)==1
    fpr=0;
end

if isnan(fdr)==1
    fdr=0;
end

if isnan(foe)==1
    foe=0;
end

if isnan(ka)==1
    ka=0;
end

if isnan(sen)==1
    sen=0;
end

if isnan(spe)==1
    spe=0;
end

if isnan(mcc)==1
    mcc=0;
end

if isnan(bm)==1
    bm=0;
end

if isnan(mk)==1
    mk=0;
end
