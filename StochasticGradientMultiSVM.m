load q3_1_data.mat

C=100;
options.iterations=1;
options.k = 2;
options.eta0 = 1;
options.eta1 = 100;
options.C = C;
[options.featureCount,~] = size(trD);




[w,obj] = SGMSVM(trD, trLb, options)


function [W, objective] = SGMSVM(X, Y, options)
labels = Y;
data = X;
C = options.C;
iterations = options.iterations;
n_zero = options.eta0;
n_one = options.eta1;

d = size(X,1)
n = size(X, 2);

k = options.featureCount;

W = zeros(d, k);

loss = size(X, 1);

objective = zeros(iterations, 1);



for epoch = 1:iterations
    n_res = n_zero / (n_one + epoch);
    %indx = randperm(size(data, 1));
    j = rand(size(X,2));
    
    %for j = indx
        
        
       for i = 1:n   
           newW = W;      
           newW(:,i) = -inf;    
           [~, yi_hat_index] = max(newW' * X(:,i));
        
           
           condition = W(:, i)' * X(:,i) - W(:,yi_hat_index)' * X(:,i) + 1;
        
           if i == j
                if condition > 0
                    W(:,i) = W(:,i) - n_res * (W(:,i) / n);
                else
                    W(:,i) = W(:,i) - n_res * ( W(:,i) / n - C * X(:,i));
                end
           elseif i == yi_hat_index
               if condition > 0
                    W(:,i) = W(:,i) - n_res * (W(:,yi_hat_index) / n);
                else
                    W(:,i) = W(:,i) - n_res * ( W(:,yi_hat_index) / n + C * X(:,i));
               end
           else
               W(:,i) = W(:,i) - n_res * (W(:,i) / n);
               
           end
               
           
           
       end
       
       
 
end

end
   
    