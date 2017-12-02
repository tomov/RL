function pi = GEM(alpha, K)
% stick-breaking construct of a Dirichlet process
% pi ~ GEM(alpha)
% 
% INPUT:
% alpha = concentration parameter
% K = # of clusters
%
% OUTPUT:
% pi = column vector of probabilities corresponding to the mixing proportions = the probability of each cluster (normalized to 1 since K < infinity)
%
beta = nan(K,1); % where we broke each stick
pi = nan(K,1); % mixing proportion = probability of each cluster

for k = 1:K % for each cluster k
    beta(k) = betarnd(1, alpha); % draw stick-breaking portion from beta distribution
    pi(k) = beta(k) * prod(1 - beta(1:k-1)); % mixing proportion of cluster k
end

pi = pi / sum(pi); % the sum of pi converges to 1 as K --> infinity, however here we are working with limited K


