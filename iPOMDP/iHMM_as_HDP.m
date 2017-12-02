% INCOMPLETE Infinite hidden Markov model (iHMM) based on HDP using stick-breaking construction (GEM)
% Following nomenclature of Teh 2006
% incomplete b/c I wanted to show the close correspondence between HDP and
% iHMM.
% see HDP.m for comparison
%

H = @() rand(1,2) * 20; % base distribution = prior distribution over component parameters. Here, 2D uniform random variable in the square between [0, 0] and [10, 10] 
F = @(theta) mvnrnd(theta, [1 0; 0 1]); % component distribution, parametrized by theta. Here, 2D Gaussain with fixed covariance and parametrized mean. So we have a 2D Gaussian mixture

gamma = 10; % concentration parameter for shared clusters
alpha_0 = 5; % concentration parameter for clusters in each group
J = 50; % # of groups
N = 100; % # observations
K = 50; % # of shared active clusters (across groups). Note we must specify in advance b/c we're using the stick-breaking construction

assert(J == K); % groups = clusters = states in iHMM

pi = nan(K,J); % mixing proportions = probability of each shared cluster (row) for each group (column)
phi = nan(K,2); % parameters for each shared cluster

% draw shared cluster mixing proportions = popularity of each state
% beta ~ GEM(gamma)
%
beta = GEM(gamma, K);
    
% draw cluster params from base distribution
% phi_k ~ H
%
for k = 1:K
    phi(k,:) = H();
end

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


% convert to HMM terminology
% groups = previous states
% (shared) clusters = next states
% T(i,j) = T(s_t = i | s_t-1 = j)
%
T = pi;

figure;
subplot(1,2,1);
imagesc(T);
xlabel('s_{t-1}');
ylabel('s_t');
title('T(s_t|s_{t-1})');

subplot(1,2,2);
plot(beta(end:-1:1), 1:K);
ylabel('s_t');
xlabel('mean T(s_t | s_{t-1})');
title('"popularity" of state s_t');

%{

% plot the clusters
%
C = colormap;
C = C(randperm(size(C,1)),:); % use different color for each cluster


for j = 1:5
    subplot(1,5,j);
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

%}