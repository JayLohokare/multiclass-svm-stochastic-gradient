load q3_1_data.mat

C=100;

options.iterations=1000;
options.eta0 = 1;
options.eta1 = 100;
options.C = C;


[w,obj] = SGMSVM(trD,trLb, options);



function [W, obj] = SGMSVM(X, Y,options)
trLb = Y;
eta0 = 1;
eta1 = 100;
n = size(X, 1);
W = zeros(size(X, 1), k);
L = size(X, 1);
obj = zeros(iterations, 1);
w_dimension = size(W, 2);


end