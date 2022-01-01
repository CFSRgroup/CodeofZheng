function [y]=gaussian_kernel_function(x1,x2)
for i=1:size(x1,1)
y(i,1)=exp(-0.5*norm(x1(i,:)-x2(i,:))^2/(1*2));
end
end