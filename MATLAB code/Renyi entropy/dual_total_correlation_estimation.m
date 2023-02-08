function [dual_total_correlation] = dual_total_correlation_estimation(variable,sigma,alpha)

num_variable = size(variable,2); % number of variables
K_x = cell(num_variable,1); % reserve all gram matrices for each variable
H_x = zeros(num_variable,1); % reserve all entropy values for each variable

for i=1:num_variable

%% estimate entropy for the i-th variable, i.e., H(Si)
K_x{i} = real(guassianMatrix(variable(:,i),sigma))/size(variable,1);
[~, L_x] = eig(K_x{i});
lambda_x = abs(diag(L_x));
H_x(i) = (1/(1-alpha))*log((sum(lambda_x.^alpha)));

end

%% estimate joint entropy H(S1,S2,S3,...)
K_all = K_x{1};
for i=2:num_variable
    K_all = K_all.*K_x{i}.*size(variable,1);
end
[~,L_all] = eig(K_all);
lambda_all = abs(diag(L_all));
H_all =  (1/(1-alpha))*log((sum(lambda_all.^alpha)));
    
%% estimate partial joint entropy H(S1,S2,S_{i-1},S_{i+1},...,Sn)
partial_joint_entropy = zeros(num_variable,1);
for i=1:num_variable
    index_remain = [1:num_variable];
    index_remain(i) = [];
    K_partial = K_x{index_remain(1)};
    for j=2:numel(index_remain)
        K_partial = K_partial.*K_x{index_remain(j)}.*size(variable,1);
    end
    [~,L_partial] = eig(K_partial);
    lambda_partial = abs(diag(L_partial));
    H_partial =  (1/(1-alpha))*log((sum(lambda_partial.^alpha)));
    partial_joint_entropy(i) = H_partial;
end

%% estimate dual total correlation
dual_total_correlation = sum(partial_joint_entropy) - (num_variable-1)*H_all;

end