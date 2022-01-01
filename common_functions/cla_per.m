%混淆矩阵分类器性能指标评估
%author:郑展鹏
%time:2019/3/30
%notes:P:数据中真正例个数  N;数据中真反例个数  TP:真正例  TN;真反例  FP;假正例  FN;假反例

%sensitivity,recall,true positive rate(TPR)  [sen]
%specificity,true negative rate(TNR)  [spe]
%precision,positive predictive value(PPV)  阳性预测值,查准率  [pre]
%negative predictive value(NPV)  阴性预测值  [npv]
%miss rate or false negative rate(FNR)  [fnr]
%fall-out or false positive rate(FPR)  [fpr]
%false discovery rate(FDR)  [fdr]
%false omission rate(FOR)  [for]
%accuracy(ACC)  [acc]
%F1 score  [f1]
%Matthews correlation coefficient(MCC)  马修斯相关系数  [mcc]
%Informedness or Bookmaker Informedness(BM)  [bm]
%Markedness(MK)  [mk]

function [acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk] = cla_per(cfmatrix) 
TP=cfmatrix(1,1);
TN=cfmatrix(2,2);
FP=cfmatrix(1,2);
FN=cfmatrix(2,1);

acc=(TP+TN)/(TP+TN+FP+FN);
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
