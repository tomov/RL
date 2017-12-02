% Dirichlet process (DP) mixture model as Chinese restaurant process (CRP) mixture
% directly yields cluster assignments of each observation
% Following nomenclature of Teh 2010
%

%rng default;

H = @() rand(1,2) * 20; % base distribution = prior distribution over component parameters. Here, 2D uniform random variable in the square between [0, 0] and [10, 10] 
F = @(theta) mvnrnd(theta, [1 0; 0 1]); % component distribution, parametrized by theta. Here, 2D Gaussain with fixed covariance and parametrized mean. So we have a 2D Gaussian mixture

alpha = 10; % concentration parameter
N = 50; % # observations
K = 0; % # of active clusters

n = []; % # of observations assigned to each cluster
z = nan(N,1); % cluster assignment for each observation
theta_star = []; % parameters for each cluster
theta = nan(N,2); % parameters for each observation = theta_star{z(i)}
x = nan(N,2);

for i = 1:N % for each observation i
    % with probability proportional to n(k), assign i to cluster k;
    % with probability proportional to alpha, assign i to new cluster
    %
    p = [n(1:K), alpha];
    p = p / sum(p);
    
    z(i) = find(mnrnd(1, p)); % sample cluster assignment from categorical distribution with probabilities = the # of observations in each cluster, and alpha for the new cluster
    
    if z(i) == K + 1
        % new cluster
        %
        K = K + 1;
        n(K) = 1;
        theta_star(K,:) = H(); % draw new cluster params from base distribution
    else
        % old cluster
        %
        n(z(i)) = n(z(i)) + 1;
    end
    
    assert(isequal(K, numel(n)));
    assert(isequal(K, size(theta_star, 1)));
    
    % draw actual observation based on its cluster parameters
    % this is where the "mixture" part comes in; up to now, it's just a DP
    % with CRP construct
    %
    theta(i,:) = theta_star(z(i),:); % observation parameters = its cluster parameters
    x(i,:) = F(theta(i,:)); % draw observation from component distribution
end


% plot the clusters
%
C = colormap;
C = C(randperm(size(C,1)),:); % use different color for each cluster

figure;
hold on;
for k = 1:K % for each cluster
    scatter(x(z == k, 1), x(z == k, 2), 4, C(k,:)); % plot all the observations
    circle(theta_star(k,1), theta_star(k,2), 2, C(k,:)); % and a circle around the center (mean of observations)
end
hold off;
