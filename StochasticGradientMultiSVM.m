load q3_1_data.mat

C=100;
options.iterations=1;
options.eta0 = 1;
options.eta1 = 100;
options.C = C;
[options.featureCount,~] = size(trD);


%[w,objective] = SGMSVM(trD, trLb, options);

function [W, obj] = SGMSVM(X, Y, options)
labels = Y;
data = X;
C = options.C;
iterations = options.iterations;
n_zero = options.eta0;
n_one = options.eta1;
n = size(X, 1);
k = options.featureCount;

W = zeros(size(X, 2));
loss = size(X, 2);
objective = zeros(iterations);
w_dim = size(W, 2);



for epoch = 1:iterations
    n_res = n_zero / (n_one + epoch);
    indx = randperm(size(X, 1));
    
    for i = indx
        newW = W;
        newW(:,(i)) = -inf;
        
        argmax = newW' * X(:,i)
        
        [~, yi_hat_index] = max(argmax);
        yi_hat = yi_hat_index;
        
        %w_tmp = W(:,yi_hat)';
        
        condition = W(:,yi_hat)' * X(:,i) - W(:,(i))' * X(:,i) + 1;
        
        loss(i) = max(condition, 0);
        
        for j = 1:k
           if j == (i)
               if condition > 0
                   if size(W, 2) == 1
                        W(j) = W(j) - n_res * (sum(W')' / n - C * X(:, i));
                   else
                        W(:,j) = W(:,j) - n_res * (sum(W')' / n - C * X(:, i));
                   end
               else
                   if size(W, 2) == 1
                        W(j) = W(j) - n_res * (sum(W')' / n);
                   else
                        W(:,j) = W(:,j) - n_res * (sum(W')' / n);
                   end
               end
           elseif j == yi_hat
               if condition > 0
                   if size(W, 2) == 1
                        W(j) = W(j) - n_res * (sum(W')' / n +  C * X(:, i));
                   else
                        W(:,j) = W(:,j) - n_res * (sum(W')' / n +  C * X(:, i));
                   end
               else
                   if size(W, 2) == 1
                        W(j) = W(j) - n_res * (sum(W')' / n);
                   else
                        W(:,j) = W(:,j) - n_res * (sum(W')' / n);
                   end
               end
           else
               W(:,j) = W(:,j) - n_res * (sum(W')' / n);
           end
        end
    end
    if w_dim ~= 2
        Y_tr = X' * W;
    else
        Y_tr = X' * W(:,1);
    end

    objective(epoch) = 1/2 * sum(sum(W'*W)) + C * sum(loss);
end
end

