function p = eps_greedy(Q, eps)
% Return epsilon-greedy policy pi(s) = PF over actions,
% given Q(s,.) and eps
%
[~, a] = max(Q);
if numel(unique(Q)) == 1
    % no max => choose at random
    %
    p = ones(size(Q));
    p = p / sum(p);
else
    % return best action
    % with small probability eps, return another action at random
    %
    p = ones(size(Q)) * eps / (numel(Q) - 1);
    p(a) = 1 - eps;
    assert(abs(sum(p) - 1) < 1e-8);
end