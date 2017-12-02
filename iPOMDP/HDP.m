% Hierarchical Dirichlet process (HDP) using stick-breaking construction
% (GEM)
% Following nomenclature of Teh 2006
%

H = @() rand(1,2) * 20; % base distribution = prior distribution over component parameters. Here, 2D uniform random variable in the square between [0, 0] and [10, 10] 
F = @(theta) mvnrnd(theta, [1 0; 0 1]); % component distribution, parametrized by theta. Here, 2D Gaussain with fixed covariance and parametrized mean. So we have a 2D Gaussian mixture

gamma = 10; % concentration parameter for shared clusters
alpha_0 = 5; % concentration parameter for clusters in each group
J = 5; % # of groups
N = 50; % # observations
K = 1000; % # of shared active clusters (across groups). Note we must specify in advance b/c we're using the stick-breaking construction

b = nan(K,1); % where we broke each stick
phi = nan(K,2); % parameters for each shared cluster
pi = nan(K,J); % mixing proportions = probability of each shared cluster (row) for each group (column)
n = zeros(K,J); % # of observations assigned to each cluster (row) for each group (column)
z = nan(N,J); % cluster assignment for each observation (row) for each group (column)
theta = nan(N,J,2); % parameters for each observation (row) for each group (column) = phi(z(i))
x = nan(N,J,2); % observation (row) for each group (column)


% draw shared cluster mixing proportions 
% beta ~ GEM(gamma)
%
beta = GEM(gamma, K);
    
% draw cluster params from base distribution
% phi_k ~ H
%
for k = 1:K
    phi(k,:) = H();
end

% draw mixing proportion for each group
% pi_j ~ DP(alpha_0, beta) 
%
for j = 1:J % for each group j
    % use stick-breaking construct for each group
    % remember that pi_j = sum of w_k delta_c_k
    % where w_k = the mixing coefficients, and
    % c_k = the "parameters" for "cluster" k
    % however in this case, the base distribution is a categorical
    % distribution beta
    % so a draw c_k ~ beta is an integer 1..K, hence delta_c_k is a one-hot vector
    % e.g. delta_c_k = [0 0 0 0 1 0 0 0 ...]
    % corresponding to a shared cluster from the higher-level DP
    % the rows are the one-hot vectors (so each row corresponds to a new
    % "cluster atom" delta_c_k)
    % note that the same "cluster atom" (i.e. shared cluster in the
    % higher-level DP) can be picked more than once since beta is a discrete
    % distribution, unlike the case when the base distribution H is
    % continuous and all clusters have unique parameters with probability 1
    %
    w = GEM(alpha_0, K);
    d = zeros(K,K); 
    for k = 1:K
        d(k,:) = mnrnd(1, beta);
    end
    p = w' * d;
    assert(abs(sum(p) - 1) < 1e-10); % should sum to 1
    pi(:,j) = p';
end

% draw cluster assignments and observations for each group
% z_ji ~ pi_j
% x_ji ~ F(phi_z_ji)
%
for j = 1:J % for each group
    for i = 1:N % for each observation
        z(i,j) = find(mnrnd(1, pi(:,j))); % draw (shared) cluster assignment

        theta(i,j,:) = phi(z(i,j),:); % copy over cluster parameters
        x(i,j,:) = F(reshape(theta(i,j,:), 2, 1)); % draw observation from component distribution
    end
end

% plot the clusters
%
C = colormap;
C = C(randperm(size(C,1)),:); % use different color for each cluster

figure;

for j = 1:J
    subplot(1,J,j);
    hold on;
    for k = 1:max(z(:,j)) % for each cluster
        if sum(z(:,j) == k) == 0, continue; end % NOTE: not all clusters are used!!! unlike the CRP construct
        c = C(mod(k,64),:);
        scatter(x(z(:,j) == k, j, 1), x(z(:,j) == k, j, 2), 4, c); % plot all the observations
        circle(phi(k,1), phi(k,2), 2, c); % and a circle around the center (mean of observations)
    end
    hold off;
    
    xlabel(['group ', num2str(j)]);
    
    axis([0 20 0 20]);
end