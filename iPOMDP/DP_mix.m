function [x, z, theta] = DP_mix(pi, N, theta_star, F)
% Dirichlet process mixture.
% Draw cluster assignments z_i and observations x_i given the mixing proportions pi:
% z_i ~ pi
% x_i ~ F(theta*_z_i)
%
% INPUT:
% pi = (1 x K) vector of mixing proportions = the probability of each cluster; assumes pi ~ GEM(..) 
% N = # of observations
% theta_star = (K x P) parameters for each of the K clusters
% F = component distribution; function that takes a (1 x P) vector of parameters and returns a single observation as a (1 x B) vector
%
% OUTPUT:
% x = (N x B) vector of N observations
% z = (1 x N) vector of cluster assignments for the observations
% theta = (N x P) vector of parameters for each observation
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

