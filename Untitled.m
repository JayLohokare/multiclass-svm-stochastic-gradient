load q3_1_data.mat

C=100;
X = trD;
Y = trLb;
k = 2;
iterations = 2;

[w] = SGDSVM(X, Y, C, k, iterations)


function [W, obj] = SGDSVM(X, Y, C, k, iterations)
trLb = Y;
eta0 = 1;
eta1 = 100;
n = size(X, 1);
W = zeros(size(X, 1), k);
L = size(X, 1);
obj = zeros(iterations, 1);
w_dimension = size(W, 2);
for epoch = 1:iterations
    eta = eta0 * (eta1+epoch);
    indexes = randperm(size(X, 2));
    for i = indexes
        tmpW = W;
        tmpW(:,Y(i)) = -inf;
        product = tmpW' * X(:,i);
        [~, yi_hat_index] = max(product);
        yi_hat = yi_hat_index;
        %yi_hat = Y(yi_hat_index);
        if size(W, 2) == 1
            w_tmp = W(yi_hat)';
        else
            w_tmp = W(:, yi_hat)';
        end
        condition = w_tmp * X(:,i) - W(:,Y(i))' * X(:,i) + 1;
        L(i) = max(condition, 0);
        for j = 1:k
           if j == Y(i)
               if condition > 0
                   if size(W, 2) == 1
                        W(j) = W(j) - eta * (sum(W')' / n - C * X(:, i));
                   else
                        W(:,j) = W(:,j) - eta * (sum(W')' / n - C * X(:, i));
                   end
               else
                   if size(W, 2) == 1
                        W(j) = W(j) - eta * (sum(W')' / n);
                   else
                        W(:,j) = W(:,j) - eta * (sum(W')' / n);
                   end
               end
           elseif j == yi_hat
               if condition > 0
                   if size(W, 2) == 1
                        W(j) = W(j) - eta * (sum(W')' / n +  C * X(:, i));
                   else
                        W(:,j) = W(:,j) - eta * (sum(W')' / n +  C * X(:, i));
                   end
               else
                   if size(W, 2) == 1
                        W(j) = W(j) - eta * (sum(W')' / n);
                   else
                        W(:,j) = W(:,j) - eta * (sum(W')' / n);
                   end
               end
           else
               W(:,j) = W(:,j) - eta * (sum(W')' / n);
           end
        end
    end
    if w_dimension == 2
        Y_tr = X' * W(:,1);
    else
        Y_tr = X' * W;
    end

    obj(epoch) = 1/2 * sum(sum(W'*W)) + C * sum(L);
    %fprintf('Objective %f\n', obj(epoch));
end
end