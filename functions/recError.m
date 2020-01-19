function  Re = recError(X, R, ThrTest)
   K = length(R);
   reSet = 1 : size(X, 2);
   Re = ones(1, size(X, 2));
   for ii = 1 : K
       Re(reSet) = sum((R(ii).val * X(:, reSet)) .^ 2);
       % if reconstruction error is small, X is regarded as normal.
       % idx: index of normal X
       idx = find(Re <= ThrTest); 
       % reSet: index of abnormal X
       % only choose the index of abnormal X, enter next loop
       reSet = setdiff(reSet, idx);
   end 
end