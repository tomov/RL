% Hierarchical iHMM, version 1 -- clusters of states that share the same
% average transition vector.
% Following nomenclature of Teh 2006
% see iHMM.m for comparison
%
clear all; close all;

rng(15);

H = @() rand(1,2) * 20; % base distribution = prior distribution over observation distribution parameters. Here, 2D uniform random variable in the square between [0, 0] and [10, 10] 
O = @(theta) mvnrnd(theta, [1 0; 0 1]); % observation distribution, parametrized by theta. Here, 2D Gaussain with fixed covariance and parametrized mean. So we have a 2D Gaussian mixture

gamma = 100; % concentration parameter for state "communities" (clusters of states)
alpha_0 = 10; % concentration parameter for state popularities (mean transition distribution)
alpha_1 = 3; % concentration parameter for mean transition distributions (one for each community of states)
alpha_2 = 10; % concentration parameter for the individual transition distributions (for each previous state)

C = 10; % # of "communities" = clusters of states
N = 100; % # observations = # of time points
J = 50; % # of groups = # of states
K = 50; % # of shared active clusters (across groups) = # of states

assert(J == K); % groups = clusters = states in hiHMM

T_means = nan(C,K); % average transition probabilities for each community of states
z = nan(J,1); % community assignment for each state
T = nan(J,K); % transition probabilities: T(j,i) = T(s_t = i | s_t-1 = j) 
phi = nan(K,2); % parameters for observation distribution for each state: o ~ O(.|s) = O(phi_s)

    
% draw observation distribution params from base distribution; one for each state k
% phi_k ~ H
%
for k = 1:K % for each state k
    phi(k,:) = H();
end

% draw state cluster mixing proportions
% beta ~ GEM(gamma)
%
beta = GEM(gamma, C);

% draw popularity = average transition probability to each state
% T_mean ~ GEM(alpha_0)
%
T_mean = GEM(alpha_0, K);

% draw average transition probability to each state; one for each *cluster* of states c
% T_means_c ~ DP(alpha_1, T_mean)
%
for c = 1:C % for each cluster of states c
    T_means(c,:) = DP(alpha_1, T_mean);
end

% for each state j, draw its community and transition probabilities; see DP_mix.m 
% z_j ~ beta
% T_j ~ DP(alpha_2, T_means_z_j) 
%
for j = 1:J % for each previous state j
    z(j) = find(mnrnd(1, beta)); 
    T(j,:) = DP(alpha_2, T_means(z(j),:));
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


%% show transition matrix
%
figure;

subplot(3,2,1);
plot(T_mean);
xlabel('s_t');
ylabel('mean T(s_t | s_{t-1})');
title('$\bar{T}$ = popularity of each state', 'interpreter','Latex');

subplot(3,2,3);
imagesc(T_means);
xlabel('s_t');
ylabel('community');
title('$\bar{T}_c$ = popularity for each state within each community $c$', 'interpreter','Latex');

subplot(3,2,4);
plot(beta, 1:C);
ylabel('community');
xlabel('beta');
title('$\beta$ = popularity of each community', 'interpreter','Latex');
set(gca,'Ydir','reverse');


subplot(3,2,5);
imagesc(T);
xlabel('s_t');
ylabel('s_{t-1}');
title('$T_s$ = transition vector for each state $s$', 'interpreter','Latex');


%{

%% plot the observations
%
figure;
C = colormap;
C = C(randperm(size(C,1)),:); % use different color for each cluster

hold on;
for k = 1:max(s) % for each state
    if sum(s == k) == 0, continue; end % NOTE: not all states are used!!! unlike the CRP construct
    c = C(mod(k,64),:);
    scatter(o(s == k, 1), o(s == k, 2), 4, c); % plot all the observations for that state
    circle(phi(k,1), phi(k,2), 2, c); % and a circle around the center of the state (mean of observations)
end
hold off;


%% plot observations sequentially
%
figure;
C = colormap;
C = C(randperm(size(C,1)),:); % use different color for each cluster

hold on;
for t = 1:N % for each time point
    k = s(t);
    c = C(mod(k,64),:);
    scatter(o(t, 1), o(t, 2), 4, c); % plot all the observations for that state
    if t > 1
        plot([o(t-1, 1) o(t, 1)], [o(t-1, 2) o(t, 2)], 'Color', c);
    end
    circle(phi(k,1), phi(k,2), 2, c); % and a circle around the center of the state (mean of observations)
    
    pause;
end
hold off;

%}
