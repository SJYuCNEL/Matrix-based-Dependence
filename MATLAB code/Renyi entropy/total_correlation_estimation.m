function [total_correlation] = total_correlation_estimation(variable,sigma,alpha)

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
    
%% estimate total correlation TC(S1,S2,...,Sn)
total_correlation = sum(H_x) - H_all;

end