%基于2类混淆矩阵的分类精度计算指标评价系统
function [sen,spe,acc,pre,npv,f1] = per_eva(cfmatrix) 
TP=cfmatrix(1,1);
TN=cfmatrix(2,2);
FP=cfmatrix(1,2);
FN=cfmatrix(2,1);

sen=TP/(TP+FN);
spe=TN/(TN+FP);
pre=TP/(TP+FP); %pre是指精确度,即对正类样本的查准率。
npv=TN/(TN+FN); %npv是指negative predictive value,即对负类样本的查准率。
acc=(TP+TN)/(TN+FP+TP+FN);

if isnan(pre)==1
    pre=0;
end

if isnan(npv)==1
    npv=0;
end


f1=(2*(sen+spe)/2*(pre+npv)/2)/((sen+spe)/2+(pre+npv)/2);%平均F1分数
if isnan(f1)==1
    f1=0;
end

