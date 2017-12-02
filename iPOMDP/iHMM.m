% Infinite hidden Markov model (iHMM) based on HDP using stick-breaking construction (GEM)
% Following nomenclature of Teh 2006
% see HDP.m for comparison
%

H = @() rand(1,2) * 20; % base distribution = prior distribution over observation distribution parameters. Here, 2D uniform random variable in the square between [0, 0] and [10, 10] 
O = @(theta) mvnrnd(theta, [1 0; 0 1]); % observation distribution, parametrized by theta. Here, 2D Gaussain with fixed covariance and parametrized mean. So we have a 2D Gaussian mixture

gamma = 10; % concentration parameter for shared clusters
alpha_0 = 30; % concentration parameter for clusters in each group

N = 100; % # observations = # of time points
J = 50; % # of groups = # of states
K = 50; % # of shared active clusters (across groups) = # of states

assert(J == K); % groups = clusters = states in iHMM

T = nan(J,K); % transition probabilities: T(j,i) = T(s_t = i | s_t-1 = j) 
phi = nan(K,2); % parameters for observation distribution for each state: o ~ O(.|s) = O(phi_s)

% draw popularity = average transition probability to each state
% T_mean ~ GEM(gamma)
%
T_mean = GEM(gamma, K);
    
% draw observation distribut params from base distribution
% phi_k ~ H
%
for k = 1:K
    phi(k,:) = H();
end

% for each state j, draw transition probabilities
% T_j ~ DP(alpha_0, T_mean) 
%
for j = 1:J % for each previous state j
    T(j,:) = DP(alpha_0, T_mean);
end

% draw first state based on popularity, as well as its observation
% s_1 ~ T_mean
% o_1 ~ F()
%
s(1) = find(mnrnd(1, T_mean));
o(1,:) = O(phi(s(1),:));

% draw the sequence of states and observations
% s_t ~ T_s_t-1
%
for t = 2:N
    s(t) = find(mnrnd(1, T(s(t-1),:)))
    o(t,:) = O(phi(s(t),:));
end


% show transition matrix
%
figure;

subplot(2,1,1);
plot(T_mean);
ylabel('s_t');
xlabel('mean T(s_t | s_{t-1})');
title('"popularity" of state s_t');

subplot(2,1,2);
imagesc(T);
xlabel('s_t');
ylabel('s_{t-1}');
title('T(s_t|s_{t-1})');


% plot the observations
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

