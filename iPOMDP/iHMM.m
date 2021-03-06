% Infinite hidden Markov model (iHMM) based on HDP using stick-breaking construction (GEM)
% Following nomenclature of Teh 2006
% see HDP.m for comparison
%
clear all; close all;

rng(15);

H = @() rand(1,2) * 20; % base distribution = prior distribution over observation distribution parameters. Here, 2D uniform random variable in the square between [0, 0] and [10, 10] 
O = @(theta) mvnrnd(theta, [1 0; 0 1]); % observation distribution, parametrized by theta. Here, 2D Gaussain with fixed covariance and parametrized mean. So we have a 2D Gaussian mixture

gamma = 10; % concentration parameter for state popularities (mean transition distribution)
alpha_0 = 30; % concentration parameter for the individual transition distributions (for each previous state)

N = 100; % # observations = # of time points
J = 50; % # of groups = # of states
K = 50; % # of shared active clusters (across groups) = # of states

assert(J == K); % groups = clusters = states in iHMM

T = nan(J,K); % transition probabilities: T(j,i) = T(s_t = i | s_t-1 = j) 
phi = nan(K,2); % parameters for observation distribution for each state: o ~ O(.|s) = O(phi_s)


% draw observation distribution params from base distribution; one for each state k
% phi_k ~ H
%
for k = 1:K % for each state k
    phi(k,:) = H();
end

% draw popularity = average transition probability to each state
% T_mean ~ GEM(gamma)
%
T_mean = GEM(gamma, K);
    
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


%% show transition matrix
%
figure;

subplot(2,1,1);
plot(T_mean);
xlabel('s_t');
ylabel('\beta_{s_t}');
title('\beta = "popularity" of state s_t = mean T(s_t | s_{t-1}) across s_{t-1}');

subplot(2,1,2);
imagesc(T);
xlabel('s_t');
ylabel('s_{t-1}');
title('T(s_t|s_{t-1})');


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

title('observations');


%{
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
