function G = DP(alpha, H)
% Dirichlet process witha discrete base distribution H. Returns 
% G ~ DP(alpha, H)
% Used in HDP and iHMM
%
% INPUT:
% alpha = concentration parameter
% H = base distribution as a (1 x K) vector of probabilities
%
% OUTPUT:
% G = (1 x K) vector of probabilities


% use stick-breaking construct
% remember that G = sum of pi_k delta_theta*_k
% where pi_k = the mixing coefficients, and
% theta*_k = the parameters for cluster k
% however in this case, the base distribution H is a categorical distribution.
% so a draw theta*_k ~ H is an integer 1..K;
% hence delta_theta*_k is a one-hot vector
% e.g. delta_theta*_k = [0 0 0 0 1 0 0 0 ...]
% the rows are the one-hot vectors (so each row corresponds to a new atom
% delta_theta*_k)
% Note that the same atom can be picked more than once since H is a discrete
% distribution, unlike the case when the base distribution H is
% continuous and all clusters have unique parameters with probability 1
%

K = numel(H); % # of clusters

pi = GEM(alpha, K); % pi ~ GEM(alpha)

% theta*_k ~ H (implicit)
% delta_theta*_k = atom = Kronecher delta function at theta*_k; in this case ends up being a
% one-hot vector since H is discrete
%
delta = zeros(K,K);
for k = 1:K
    delta(k,:) = mnrnd(1, H); % delta_theta*_k
end

G = pi' * delta; % G = sum of pi_k * delta_theta*_k

assert(abs(sum(G) - 1) < 1e-10); % should sum to 1

