function [K_final,K] = guassianMatrix(X,sigma)
%N = size(X,1);
G = X*X';
%K = exp(-(1/(2*sigma^2))*(diag(G)*ones(1,N)+ones(N,1)*diag(G)'-2*G));
K = bsxfun(@minus, 2*G, diag(G)');
K_final = exp((1/(2*sigma^2))*bsxfun(@minus, K, diag(G)));
% K = K-diag(G);
end