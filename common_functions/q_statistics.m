%q_statistics
%������ת��Ϊ������
%author:֣չ��
%notes:P:����������������  N;�������淴������  TP:������  TN;�淴��  FP;������  FN;�ٷ���
%����TP��TNΪ��ȷ����ĸ�����FP��FNΪ�������ĸ���
%����������Ϊ����OFS��𣬼�����OFS��negative class�����쳣OFS��positive class����
%LMW����Ϊ���ࣨ��������OFS������MMW��HMW����positive class�������쳣OFS����

function Q = q_statistics(cfmatrix)
TN=cfmatrix(1,1);
FN=cfmatrix(1,2)+cfmatrix(1,3);
FP=cfmatrix(2,1)+cfmatrix(3,1);
TP=cfmatrix(2,2)+cfmatrix(2,3)+cfmatrix(3,2)+cfmatrix(3,3);
Q = ((TN*TP)-(FN*FP))/((TN*TP)+(FN*FP));