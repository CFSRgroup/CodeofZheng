%q_statistics
%三分类转换为二分类
%author:郑展鹏
%notes:P:数据中真正例个数  N;数据中真反例个数  TP:真正例  TN;真反例  FP;假正例  FN;假反例
%其中TP和TN为正确分类的个数，FP和FN为错误分类的个数
%多类结果编码为两种OFS类别，即正常OFS（negative class）和异常OFS（positive class）。
%LMW编码为负类（表明正常OFS），而MMW和HMW类是positive class（表明异常OFS）。

function Q = q_statistics(cfmatrix)
TN=cfmatrix(1,1);
FN=cfmatrix(1,2)+cfmatrix(1,3);
FP=cfmatrix(2,1)+cfmatrix(3,1);
TP=cfmatrix(2,2)+cfmatrix(2,3)+cfmatrix(3,2)+cfmatrix(3,3);
Q = ((TN*TP)-(FN*FP))/((TN*TP)+(FN*FP));