function Combination = sparse_combination(X, Dim, lambda)
%%
% X: featureMatrix, i.e. X = {x1, x2, ..., xn}
% Dim: dimension of Si
% lambda: the threshold in the paper

warning off;
tic;
delta = 0.03;
i = 1;
count = 0;
rows = size(X, 1);                  % rows of each feature vector
tot = size(X, 2);                   % total features
remains_index = 1: tot;             % OMEGAc in the paper
gamma = zeros(tot, tot);            % gamma in the paper
beta() = struct('val');             % beta in the paper
D() = struct('val');                % the S = {Si} in the paper

while ~isempty(remains_index)
    ee = remains_index;
    gamma(i, :) = zeros(1, tot);
    gamma(i, ee) = 1;
    remains_ones_index = ee;        % the ones in the index
    len = length(ee);
    
    % initialize Si
    if length(remains_index) < Dim
        S_cur = rand(rows, Dim);
    else
        [~, S_cur] = kmeans(X(:, remains_index)', Dim, 'start', zeros(Dim, rows), ...
            'emptyaction', 'singleton', 'options', statset('MaxIter', 10));
        S_cur = S_cur';
        S_cur = S_cur / norm(S_cur);
    end
    
    t = Inf;
    accu = accuracy(len);
    while t > accu
        tmp = (S_cur' * S_cur) \ S_cur';
        for jj = remains_ones_index
            beta(i, jj).val = tmp * X(:, jj);                   % update beta, beta(i, jj).val is a column vector with size of Dim * 1
        end
        grad_S = grad_of_L(beta(i, :), S_cur, gamma(i, :));     % calculate gradient
        S_cur = S_cur - grad_S .* delta;                        % update Si
        t = norm(grad_S);
    end
    
    S_cur = S_cur / norm(S_cur);
    tmp = (S_cur' * S_cur) \ S_cur';
    for jj = ee
        beta(i, jj).val = tmp * X(:, jj);                       % update beta with fixed Si
    end
    
    % update gamma with fixed Si and beta
    remains_ones_index = [];
    for jj = ee
        t = norm(X(:, jj) - S_cur * beta(i, jj).val)^2;
        if t <= lambda
            gamma(i, jj) = 1;
            remains_ones_index = [remains_ones_index, jj];
        else
            gamma(i, jj) = 0;
        end                                                     
    end
    
    if ~isempty(remains_index)
        remains_index = setdiff(remains_index, remains_ones_index); % take out the features that fit to the model
        count = count + length(remains_ones_index);
        if ~isempty(remains_ones_index)
            fprintf('%d features have been trained, %d features remain, %.2f seconds elapsed\n', count, tot - count, toc)
            D(i).val = S_cur;
            i = i + 1;
        end
    end
    disp('********************************');
end
Combination = D;
toc

%%
    % caculate the gradient of L function
    % S: D(i).val
    % beta: beta(i, :)
    % gamma: gamma(i, :)
    function val = grad_of_L(beta, S, gamma)
        rows = size(S, 1);
        val = 0;
        for j = remains_ones_index
            val = val + gamma(i, j) * ((S * beta(j).val - X(:, j)) * beta(j).val' .* 2);
        end
    end

%%
    % caculate the value of L function
    % S: D(i).val
    % beta: beta(i, :)
    % gamma: gamma(i, :)
    function val = L(beta, S, gamma)
        val = 0;
        for j = remains_ones_index
            val = val + gamma(i, j) * (norm(X(:, j) - S * beta(j).val)^2);
        end
    end

    function val = accuracy(t)
        if t < 500
            val = 0.3;
        elseif t <= 1000
            val = 1;
        elseif t <= 5000
            val = 3;
        elseif t <= 10000
            val = 6;
        elseif t <= 20000
            val = 15;
        elseif t <= 50000
            val = 20;
        else
            val = 50;
        end
    end
end

