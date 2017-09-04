function p = eps_greedy(Q, eps)
% Return epsilon-greedy policy pi(s) = PF over actions,
% given Q(s,.) and eps
%
actualQ = Q(~isinf(Q)); % only consider allowed actions

if numel(unique(actualQ)) == 1
    % no max => choose at random
    %
    p = ones(size(actualQ));
    p = p / sum(p);
else
    % return best action
    % with small probability eps, return another action at random
    %
    [~, a] = max(actualQ);
    p = ones(size(actualQ)) * eps / (numel(actualQ) - 1);
    p(a) = 1 - eps;
    assert(abs(sum(p) - 1) < 1e-8);
end

% re-insert disallowed actions with probability 0
pp = zeros(size(Q));
pp(~isinf(Q)) = p;
p = pp;
