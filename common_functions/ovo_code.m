%功能：一对一分类器类别确定
%作者：郑展鹏
%20190419

function [y]=ovo_code(x1,x2,x3)
x=x1*2^2+x2*2^1+x3*2^0;
switch x
    case 0
        y=1;
    case 1
        y=1;
    case 2
        y=1;
    case 3
        y=3;
    case 4
        y=2;
    case 5
        y=1;
    case 6
        y=2;
    case 7
        y=3;
end
end