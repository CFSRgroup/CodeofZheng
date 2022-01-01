%����2���������ķ��ྫ�ȼ���ָ������ϵͳ
function [sen,spe,acc,pre,npv,f1] = per_eva(cfmatrix) 
TP=cfmatrix(1,1);
TN=cfmatrix(2,2);
FP=cfmatrix(1,2);
FN=cfmatrix(2,1);

sen=TP/(TP+FN);
spe=TN/(TN+FP);
pre=TP/(TP+FP); %pre��ָ��ȷ��,�������������Ĳ�׼�ʡ�
npv=TN/(TN+FN); %npv��ָnegative predictive value,���Ը��������Ĳ�׼�ʡ�
acc=(TP+TN)/(TN+FP+TP+FN);

if isnan(pre)==1
    pre=0;
end

if isnan(npv)==1
    npv=0;
end


f1=(2*(sen+spe)/2*(pre+npv)/2)/((sen+spe)/2+(pre+npv)/2);%ƽ��F1����
if isnan(f1)==1
    f1=0;
end

