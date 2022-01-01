function [y]=sigmoid_function(x)
y=1./(1+exp(-x/500));
end