% DP mixture as stick-breaking construction (GEM)
% yields probability of assigning each observation to a cluster
% Following nomenclature of Teh 2010
%
% G ~ DP(alpha, H)
% G = sum of pi_k * delta_theta_k, where delta_theta_k is a Kronecher delta f'n at theta_k
% theta_i ~ G
% theta_i = the parameters (hence cluster assignment) for observation x_i


%rng default;

H = @() rand(1,2) * 20; % base distribution = prior distribution over component parameters. Here, 2D uniform random variable in the square between [0, 0] and [10, 10] 
F = @(theta) mvnrnd(theta, [1 0; 0 1]); % component distribution, parametrized by theta. Here, 2D Gaussain with fixed covariance and parametrized mean. So we have a 2D Gaussian mixture

alpha = 10; % concentration parameter
N = 50; % # observations
K = 1000; % # of active clusters NOTE -- we must specify in advance here

n = []; % # of observations assigned to each cluster
z = nan(N,1); % cluster assignment for each observation
theta_star = nan(K,2); % parameters for each cluster
theta = nan(N,2); % parameters for each observation = theta_star{z(i)}
x = nan(N,2); % observations

% draw cluster mixing proportions 
% pi ~ GEM(alpha)
%
pi = GEM(alpha, K);

% draw cluster params from base distribution
% theta*_k ~ H
%
for k = 1:K
    theta_star(k,:) = H();
end

% draw cluster assignments and observations
% z_i ~ pi
% x_i ~ F(theta*_z_i)
%
[x, z, theta] = DP_mix(pi, N, theta_star, F);

% plot the clusters
%
C = colormap;
C = C(randperm(size(C,1)),:); % use different color for each cluster

hold on;
for k = 1:max(z) % for each cluster
    if sum(z == k) == 0, continue; end % NOTE: not all clusters are used!!! unlike the CRP construct
    c = C(mod(k,64),:);
    scatter(x(z == k, 1), x(z == k, 2), 4, c); % plot all the observations
    circle(theta_star(k,1), theta_star(k,2), 2, c); % and a circle around the center (mean of observations)
end
hold off;

