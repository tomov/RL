% Hierarchical iHMM, version 2 -- iHMM within a iHMM. Example for two-layer
% iPOMDP with subgoals
% Following nomenclature of Teh 2006
% see iHMM.m for comparison
%
clear all; close all;

rng(15);

H = @() rand(1,2) * 20; % base distribution
O = @(theta) mvnrnd(theta, [1 0; 0 1]); % observation distribution

% concentration parameters
alpha_0 = 10;
alpha_1 = 10;
alpha_2 = 10;
alpha_3 = 10;
alpha_4 = 10;
alpha_5 = 10;
alpha_6 = 10;

C = 5; % # of "communities" = clusters of states
S = 20; % # of states in each community
N = 100; % # observations = # of time points

T_mean = nan(1,C); % average transition for communities = "popularity" of each community
T_c = nan(C,C); % transition functions between communities
T_mean_c = nan(C,S); % popularity of each state in community c
T_c_s = nan(S,S,C); % T(s,s',c) = T(s'|s,c) if s,s' in c

T_mean_c_dot = nan(C,S); % exit state popularities
T_mean_dot_c = nan(C,S); % entrance state popoularities
subgoal = nan(C,C); % (c,c') = subgoal state from c to c'
T_c_c_s = zeros(S,S,C,C); % T(s,s',c,c') = T(s'|s,c,c') if s in c, s' in c'

phi = nan(S,2,C); % parameters for observation distribution for each state: o ~ O(.|s) = O(phi_s)

xi = 1; % preference for within-community transitions

% draw observation distribution params
%
for c = 1:C % for each community
    for s = 1:S % for each state
        phi(s,:,c) = H();
    end
end

% draw average community transition
%
T_mean = GEM(alpha_0, C);

% draw community-to-community transition probabilities
%
for c = 1:C % for each previous community c
    T_c(c,:) = DP(alpha_1, T_mean);
    
    T_c(c,c) = T_c(c,c) + xi; %  * (1 + xi); % TODO FIXME THIS IS A HACK!!!
    T_c(c,:) = T_c(c,:) / sum(T_c(c,:));
end

% draw average state transition within each community
% and then the actual state-to-state transition within each community
%
for c = 1:C
    T_mean_c(c,:) = GEM(alpha_2, S);
    
    T_mean_c_dot(c,:) = GEM(alpha_4, S);
    T_mean_dot_c(c,:) = GEM(alpha_5, S);
    
    for s = 1:S
        T_c_s(s,:,c) = DP(alpha_3, T_mean_c(c,:));
    end
end

% draw exit states and cross-module state transitions
%
for c = 1:C
    for c_next = 1:C
        if c ~= c_next
            subgoal(c,c_next) = find(mnrnd(1, T_mean_c_dot(c, :)));
            
            s = subgoal(c,c_next);
            T_c_c_s(s,:,c,c_next) = DP(alpha_6, T_mean_dot_c(c_next, :));
        end
    end
end

% draw first community based on popularity, then state based on popularity, as well as its observation
% s_1 ~ T_mean
% o_1 ~ F()
%
c = [];
s = [];
c(1) = find(mnrnd(1, T_mean));
s(1) = find(mnrnd(1, T_mean_c(c(1),:)));
o(1,:) = O(phi(s(1),:,c(1)));

% draw the sequence of communities, states and observations
%
for t = 2:N
    c(t) = find(mnrnd(1, T_c(c(t-1),:)))
    if c(t) == c(t-1)
        % same community -> pick from within-community transition f'n
        %
        s(t) = find(mnrnd(1, T_c_s(s(t-1),:,c(t))));
    else
        % new community -> pick from its average transition f'n
        %
        s(t) = find(mnrnd(1, T_mean_c(c(t),:)));
    end
    o(t,:) = O(phi(s(t),:,c(t)));
end



%% show transition matrix
%
figure;

subplot(3,1,1);
plot(T_mean);
xlabel('c_t');
ylabel('mean T(c_t | c_{t-1})');
title('$\bar{T}$ = popularity of each community', 'interpreter','Latex');


subplot(3,1,2);
imagesc(T_c);
xlabel('c_t');
ylabel('c_{t-1}');
title('$T_c$ = transition vector from community $c$ to other communities', 'interpreter','Latex');

subplot(3,1,3);
imagesc(T_mean_c);
xlabel('s_t');
ylabel('c_t');
title('$\bar{T}_{c,\cdot}$ = state popularity within community $c$', 'interpreter','Latex');



figure;

state = @(c,s) (c - 1) * S + s;
T_fun = @(c1,s1,c2,s2) (c1 == c2) * T_c(c1,c2) * T_c_s(s1,s2,c1) + (c1 ~= c2) * T_c(c1,c2) * T_c_c_s(s1,s2,c1,c2); % transition probability from s1,c1 to s2,c2

T = nan(C*S, C*S);
for c1 = 1:C
    for s1 = 1:S
        for c2 = 1:C
            for s2 = 1:S
                T(state(c1,s1), state(c2,s2)) = T_fun(c1,s1,c2,s2);
            end
        end
    end
end
imagesc(T);
xlabel('s_t');
ylabel('s_{t-1}');
title('$T_s$ = state transitions within and across communities', 'interpreter','Latex');


%{
subplot(3,2,4);
plot(beta, 1:C);
ylabel('community');
xlabel('beta');
title('"popularity" of each community');
set(gca,'Ydir','reverse');


subplot(3,2,5);
imagesc(T);
xlabel('s_t');
ylabel('s_{t-1}');
title('T(s_t|s_{t-1})');

%}


figure;

for c = 1:C
    subplot(1,C,c);
    imagesc(T_c_s(:,:,c));
    title(['$T_{s,c=', num2str(c), '}$'], 'interpreter','Latex');
end



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
