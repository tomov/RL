% Genetic TD-learning (SARSA, Q-learning, and actor-critic) as in Sutton & Barto (2013)
% TODO dedupe with LMDP
%
classdef MDP < handle

    properties (Constant = true)
        alpha = 0.1; % learning rate
        gamma = 0.9; % discount rate
        eps = 0.1; % eps for eps-greedy
        beta = 0.1; % learning rate for policy (actor-critic)
        GPI_threshold = 0.1; % threshold for convergence of V(s) during policy evaluation
    end

    properties (Access = public)
        S = []; % all states
        I = []; % interior states
        B = []; % boundary states
        A = []; % all actions
        R = []; % R(s) = instantaneous reward at state s

        P = []; % P(s', s, a) = P(s'|s,a) = probability of going to state s' from state s after taking action a

        Q = []; % Q(s, a) = state-action value, for SARSA and Q-learning
        V = []; % V(s) = state value, for actor-critic
        H = []; % H(s, a) = modifiable policy parameters, for actor-critic 
        pi = []; % pi(s) = policy = what action to take in state s (deterministic), for policy iteration

        lambda = 0; % constant for eligibility traces
        E_V = []; % E(s) = eligibility trace for state values
        E_Q = []; % E(s, a) = eligibility trace for action values
    end
   
    methods

        % Initialize an MDP from a maze
        %
        function self = MDP(lambda)
            if ~exist('lambda', 'var')
                lambda = 0;
            end
            self.lambda = lambda;
        end

        % Solve using generalized policy iteration.
        % Notice that the resulting policy is deterministic
        %
        function solveGPI(self, do_print)
            if ~exist('do_print', 'var')
                do_print = false;
            end
            N = numel(self.S);
            pi = randi(numel(self.A), [N 1]);

            self.V(self.B) = self.R(self.B); % boundary V(s) = R(s)

            assert(self.gamma < 1); % doesn't work with gamma = 1 -> the states start going around in circles after a few iterations

            policy_stable = false;
            iter = 0;
            while ~policy_stable
                % policy evaluation
                %
                delta = Inf;
                while delta > self.GPI_threshold
                    delta = 0;
                    if do_print, fprintf('\n'); end
                    for s = self.I
                        v = self.V(s);
                        a = pi(s);
                        self.V(s) = sum(self.P(:, s, a) .* (self.R(s) + self.gamma * self.V(:)));
                        delta = max(delta, abs(v - self.V(s)));
                        if do_print, fprintf('%d: %.2f -> %.2f (action %d), delta %.2f\n', s, v, self.V(s), a, delta); end
                    end
                end
                if do_print, disp(self.V'); end

                % policy improvement
                %
                policy_stable = true;
                for s = self.I
                    a = pi(s);
                    r = squeeze(sum(self.P(:, s, :) .* (self.R(s) + self.gamma * self.V(:)), 1));
                    r(isinf(self.Q(s,:))) = -Inf; % mark illegal actions as illegal
                    if do_print, fprintf('  -- %d: %s\n', s, sprintf('%d ', r)); end
                    assert(numel(r) == numel(self.A));
                    [~, pi(s)] = max(eps_greedy(r, MDP.eps)); % takes care of illegal actions
                    if pi(s) ~= a
                        policy_stable = false;
                    end
                end
                if do_print, disp(pi'); end
            end

            self.pi = pi;
        end

        %
        % Sample paths from deterministic policy pi generated using generalized policy iteration
        %

        function [Rtot, path] = sampleGPI(varargin)
            self = varargin{1};
            [Rtot, path] = self.sample_helper(@self.init_sampleGPI, @self.stepGPI, varargin{2:end});
        end

        function state = init_sampleGPI(self, s)
            if ~exist('s', 'var')
                s = find(self.map == self.agent_symbol);
            end
            assert(numel(find(self.I == s)) == 1);

            state.Rtot = 0;
            state.path = [];
            state.rs = [];
            state.pes = [];
            state.s = s;
            state.a = self.pi(s);
            state.pi = zeros(size(self.A));
            state.pi(state.a) = 1;
            state.done = false;
            state.method = 'GPI';
        end

        function state = stepGPI(self, state)
            s = state.s;
            a = state.a;

            state.Rtot = state.Rtot + self.R(s);
            state.path = [state.path, s];

            new_s = samplePF(self.P(:, s, a));
            new_a = self.pi(new_s);
            r = self.R(new_s);
            pi = zeros(size(self.A));
            pi(new_a) = 1;

            if ismember(new_s, self.B)
                % Boundary state
                %
                state.Rtot = state.Rtot + self.R(new_s);
                state.path = [state.path, new_s];
                state.done = true;
            else
                % Internal state
                %
                state.s = new_s;
                state.a = new_a;
                state.pi = pi;
            end
            state.rs = [state.rs, r];
            state.pes = [state.pes, 0]; % no PEs; we're not learning any more
        end

        %
        % Run an episode and update Q-values using SARSA
        %

        function [Rtot, path] = sampleSARSA(varargin)
            self = varargin{1};
            [Rtot, path] = self.sample_helper(@self.init_sampleSARSA, @self.stepSARSA, varargin{2:end});
        end

        function state = init_sampleSARSA(self, s)
            if ~exist('s', 'var')
                s = find(self.map == self.agent_symbol);
            end
            assert(numel(find(self.I == s)) == 1);

            state.Rtot = 0;
            state.path = [];
            state.rs = [];
            state.pes = [];
            state.s = s;
            pi = self.eps_greedy(s);
            state.pi = pi;
            state.a = samplePF(pi);
            state.done = false;
            state.method = 'SARSA';
            state.r = 0;
            state.pe = 0;
        end
           
        function state = stepSARSA(self, state)
            s = state.s;
            a = state.a;

            state.Rtot = state.Rtot + self.R(s);
            state.path = [state.path, s];

            new_s = samplePF(self.P(:, s, a));
            pi = self.eps_greedy(new_s);
            new_a = samplePF(pi);

            r = self.R(new_s);
            pe = r + self.gamma * self.Q(new_s, new_a) - self.Q(s, a);
            self.Q(s,a) = self.Q(s,a) + self.alpha * pe;
            
            if ismember(new_s, self.B)
                % Boundary state
                %
                state.Rtot = state.Rtot + self.R(new_s);
                state.path = [state.path, new_s];
                state.done = true;
            else
                % Internal state
                %
                state.s = new_s;
                state.a = new_a;
                state.r = r;
                state.pe = pe;
                state.pi = pi;
            end
            state.rs = [state.rs, r];
            state.pes = [state.pes, pe];
        end

        %
        % Run an episode and update Q-values using Q-learning
        %

        function [Rtot, path] = sampleQ(varargin)
            self = varargin{1};
            [Rtot, path] = self.sample_helper(@self.init_sampleQ, @self.stepQ, varargin{2:end});
        end

        function state = init_sampleQ(self, s)
            if ~exist('s', 'var')
                s = find(self.map == self.agent_symbol);
            end
            assert(numel(find(self.I == s)) == 1);

            state.Rtot = 0;
            state.path = [];
            state.rs = [];
            state.pes = [];
            state.s = s;
            pi = self.eps_greedy(s);
            state.pi = pi;
            state.a = samplePF(pi);
            state.done = false;
            state.method = 'Q';
            state.r = 0;
            state.pe = 0;
        end

        function state = stepQ(self, state)
            s = state.s;
            a = state.a;

            state.Rtot = state.Rtot + self.R(s);
            state.path = [state.path, s];

            new_s = samplePF(self.P(:,s,a));
            pi = self.eps_greedy(new_s);
            new_a = samplePF(pi);
       
            r = self.R(new_s);
            pe = r + self.gamma * max(self.Q(new_s, :)) - self.Q(s, a);
            self.Q(s,a) = self.Q(s,a) + self.alpha * pe;
            
            if ismember(new_s, self.B)
                % Boundary state
                %
                state.Rtot = state.Rtot + self.R(new_s);
                state.path = [state.path, new_s];
                state.done = true;
            else
                % Internal state
                %
                state.s = new_s;
                state.a = new_a;
                state.r = r;
                state.pe = pe;
                state.pi = pi;
            end
            state.rs = [state.rs, r];
            state.pes = [state.pes, pe];
        end

        %
        % Run an episode and update state-values and policy using actor-critic
        %

        function [Rtot, path] = sampleAC(varargin)
            self = varargin{1};
            [Rtot, path] = self.sample_helper(@self.init_sampleAC, @self.stepAC, varargin{2:end});
        end

        function state = init_sampleAC(self, s)
            if ~exist('s', 'var')
                s = find(self.map == self.agent_symbol);
            end
            assert(numel(find(self.I == s)) == 1);

            state.Rtot = 0;
            state.path = [];
            state.rs = [];
            state.pes = [];
            state.s = s;
            pi = self.softmax(s);
            state.pi = pi;
            state.a = samplePF(pi);
            state.done = false;
            state.method = 'AC';
            state.r = 0;
            state.pe = 0;

            self.E_V(:) = 0;
            self.E_Q(:) = 0;
        end

        function state = stepAC(self, state)
            s = state.s;
            a = state.a;

            state.Rtot = state.Rtot + self.R(s);
            state.path = [state.path, s];

            new_s = samplePF(self.P(:,s,a));
            pi = self.softmax(new_s);
            new_a = samplePF(pi);
       
            r = self.R(new_s);
            pe = r + self.gamma * self.V(new_s) - self.V(s);

            % TD(0) -- for sanity checks
            %self.V(s) = self.V(s) + self.alpha * pe;
            %self.H(s, a) = self.H(s, a) + self.beta * pe;

            % update state values
            self.E_V(s) = self.E_V(s) + 1;
            self.V = self.V + self.alpha * pe * self.E_V;
            self.E_V = self.E_V * self.gamma * self.lambda;

            % update policies
            self.E_Q(s, a) = self.E_Q(s, a) + 1;
            self.H = self.H + self.beta * pe * self.E_Q;
            self.E_Q = self.E_Q * self.gamma * self.lambda;
            
            if ismember(new_s, self.B)
                % Boundary state
                %
                state.Rtot = state.Rtot + self.R(new_s);
                state.path = [state.path, new_s];
                state.done = true;
            else
                % Internal state
                %
                state.s = new_s;
                state.a = new_a;
                state.r = r;
                state.pe = pe;
                state.pi = pi;
            end
            state.rs = [state.rs, r];
            state.pes = [state.pes, pe];
        end

        % Generic function that samples paths given a state initializer and a step function
        %
        function [Rtot, path] = sample_helper(self, init_fn, step_fn, s, do_print)
            if ~exist('s', 'var')
                state = init_fn();
            else
                state = init_fn(s);
            end
            if ~exist('do_print', 'var')
                do_print = false;
            end

            map = self.map;
            if do_print, disp(map); end
            while numel(state) > 1 || ~state.done
                if numel(state) == 1 % else MAXQ... TODO cleanup
                    [x, y] = self.I2pos(state.s);
                    old_s = state.s;
                    old_a = state.a;
                end
               
                state = step_fn(state); 

                if numel(state) == 1 % else MAXQ... TODO cleanup
                    if state.done
                        if do_print, fprintf('(%d, %d), %d --> END [%.2f%%]\n', x, y, old_a, self.P(state.s, old_s, old_a) * 100); end
                    else
                        [new_x, new_y] = self.I2pos(state.s);
                        map(x, y) = self.empty_symbol;
                        map(new_x, new_y) = self.agent_symbol;
                        if do_print, fprintf('(%d, %d), %d --> (%d, %d) [%.2f%%]\n', x, y, old_a, new_x, new_y, self.P(state.s, old_s, old_a) * 100); end
                        if do_print, disp(map); end
                    end
                end
            end
            if do_print, fprintf('Total reward: %d\n', state.Rtot); end

            Rtot = state.Rtot;
            path = state.path;
        end

        % Pick action a from state s using eps-greedy based on action-values
        % Returns the action PF
        % used for SARSA and Q-learning
        %
        function p = eps_greedy(self, s)
            p = eps_greedy(self.Q(s,:), MDP.eps);
        end

        % Pick action a from state s using softmax based on H policy parameters
        % Returns the action PF
        % used for actor-critic
        %
        function p = softmax(self, s)
            h = self.H(s, :);
            %{
            e = self.E_Q(s, :);
            scale = mean(abs(h(~isinf(h)))) / mean(e);
            if ~isinf(scale) && ~isnan(scale)
                h = h - e * scale;
            end
            %}
            p = exp(h);
            p = p / sum(p);
        end

        % Get PVFs as in Machado et al (2017)
        %
        function pvfs = get_pvfs(self)
            A = sum(self.P, 3);
            D = diag(sum(A, 2));
            L = D - A; % combinatorial graph Laplacian
            %L = (pinv(D) ^ 0.5) * (L - A) * (pinv(D) ^ 0.5); % normalized graph Laplacian
            [Q, Lambda] = eig(L);
            pvfs = Q;
        end

    end % end methods
end % end class
