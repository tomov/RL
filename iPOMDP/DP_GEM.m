% DP mixture as stick-breaking construction (GEM)
% yields probability of assigning each observation to a cluster
% Following nomenclature of Teh 2010
%

%rng default;

H = @() rand(1,2) * 20; % base distribution = prior distribution over component parameters. Here, 2D uniform random variable in the square between [0, 0] and [10, 10] 
F = @(theta) mvnrnd(theta, [1 0; 0 1]); % component distribution, parametrized by theta. Here, 2D Gaussain with fixed covariance and parametrized mean. So we have a 2D Gaussian mixture

alpha = 10; % concentration parameter
N = 50; % # observations
K = 1000; % # of active clusters NOTE -- we must specify in advance here

n = []; % # of observations assigned to each cluster
z = nan(N,1); % cluster assignment for each observation
theta_star = []; % parameters for each cluster
theta = nan(N,2); % parameters for each observation = theta_star{z(i)}
x = nan(N,2);

beta = nan(K,1); % where we broke each stick
pi = nan(K,1); % mixing proportion = probability of each cluster

% draw cluster mixing proportions 
% pi ~ GEM(alpha)
% theta*_k ~ H
%
for k = 1:K % for each cluster k
    beta(k) = betarnd(1, alpha); % draw stick-breaking portion from beta distribution
    pi(k) = beta(k) * prod(1 - beta(1:k-1)); % mixing proportion of cluster k
    theta_star(k,:) = H(); % draw cluster params from base distribution
end

pi = pi / sum(pi); % the sum of pi converges to 1 as K --> infinity, however here we are working with limited K

% draw observations
% x_i ~ F(theta*_z_i)
%
for i = 1:N % for each observation i
    z(i) = find(mnrnd(1, pi)); % sample cluster assignment from categorical distribution with probabilities = the mixing proportions
    
    % draw actual observation based on its cluster parameters
    % this is where the "mixture" part comes in; up to now, it's just a DP
    % with stick-breaking construct
    %
    theta(i,:) = theta_star(z(i),:); % observation parameters = its cluster parameters
    x(i,:) = F(theta(i,:)); % draw observation
end


% plot the clusters
%
C = colormap;
C = C(randperm(size(C,1)),:); % use different color for each cluster

figure;
hold on;
for k = 1:max(z) % for each cluster
    if sum(z == k) == 0, continue; end % NOTE: not all clusters are used!!! unlike the CRP construct
    c = C(mod(k,64),:);
    scatter(x(z == k, 1), x(z == k, 2), 4, c); % plot all the observations
    circle(theta_star(k,1), theta_star(k,2), 2, c); % and a circle around the center (mean of observations)
end
hold off;

