%kappaÏµÊı¼ÆËã
function [a]=kappa(b)
po=trace(b)/sum(sum(b));
pc=(sum(b,1)*sum(b,2))/sum(sum(b))^2;
a=(po-pc)/(1-pc);

