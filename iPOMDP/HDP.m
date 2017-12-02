% Hierarchical Dirichlet process (HDP) using stick-breaking construction (GEM)
% Following nomenclature of Teh 2006
% see DP_GEM.m for comparison
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


% draw cluster params from base distribution
% phi_k ~ H
%
for k = 1:K % for each (shared) cluster k
    phi(k,:) = H();
end

% draw shared cluster mixing proportions 
% beta ~ GEM(gamma)
%
beta = GEM(gamma, K);
    
% for each group,
% draw mixing proportions, then the cluster assignments and the observations
% pi_j ~ DP(alpha_0, beta) 
% z_ji ~ pi_j
% x_ji ~ F(phi_z_ji)
% note groups are i.i.d. given beta
%
for j = 1:J % for each group j
    pi(:,j) = DP(alpha_0, beta);
    [x(:,j,:), z(:,j,:), theta(:,j,:)] = DP_mix(pi(:,j), N, phi, F);
end


% plot the clusters
%
C = colormap;
C = C(randperm(size(C,1)),:); % use different color for each cluster

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