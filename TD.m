% TD-learning (SARSA, Q-learning, and actor-critic) as in Sutton & Barto (2013)
% TODO dedupe with LMDP
%
classdef TD < handle

    properties (Constant = true)
        % General
        % 
        R_I = -1; % penalty for staying in one place
        alpha = 0.1; % learning rate
        gamma = 1; % discount rate
        eps = 0.1; % eps for eps-greedy
        beta = 0.1; % learning rate for policy (actor-critic)
        GPI_threshold = 0.1; % threshold for convergence of V(s) during policy evaluation

        % Maze
        %
        absorbing_symbols = '-0123456789$';
        agent_symbol = 'X';
        empty_symbol = '.';
        impassable_symbols = '#';
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

        I2B = []; % I2B(s) = corresponding B state for given I state s, or 0 if none
        B2I = []; % B2I(s) = corresponding I state for given B state s


        % Maze stuff
        %
        map = [];
    end
   
    methods

        % Initialize an MDP from a maze
        %
        function self = TD(map)
            self.map = map; % so we can use pos2I

            absorbing_inds = find(ismember(map, self.absorbing_symbols)); % goal squares = internal states with corresponding boundary states
            I_with_B = absorbing_inds;
            Ni = numel(map); % internal states = all squares, including walls (useful for (x,y) --> state)
            Nb = numel(I_with_B); % boundary states
            N = Ni + Nb;
            
            S = 1 : N; % set of states
            I = 1 : Ni; % set of internal states
            B = Ni + 1 : Ni + Nb; % set of boundary states
            A = 1:6;  % set of actions
            self.S = S;
            self.I = I;
            self.B = B;
            self.A = A;

            I2B = zeros(N, 1);
            B2I = zeros(N, 1);
            I2B(I_with_B) = B; % mapping from I states to corresponding B states
            B2I(I2B(I2B > 0)) = I_with_B; % mapping from B states to corresponding I states
            self.I2B = I2B;
            self.B2I = B2I;

            P = zeros(N, N, numel(A)); % transitions P(s'|s,a); defaults to 0
            Q = zeros(N, numel(A)); % Q-values Q(s, a)
            V = zeros(N, 1); % V-values V(s)
            H = zeros(N, numel(A)); % policy parameters
            R = nan(N, 1); % instantaneous reward f'n R(s)
          
            % action 1 = stand still
            % actions 2,3,4,5 = move to adjacent squares
            % action 6 = pick up reward (to make it compatible with LMDPs)
            %
            % adjacency list
            % each row = [dx, dy, non-normalized P(s'|s)]
            % => random walk, but also bias towards staying in 1 place
            %
            adj = [0, 0; ...
                -1, 0; ...
                0, -1; ...
                1, 0; ...
                0, 1];
            assert(size(adj, 1) + 1 == numel(A));

            % iterate over all internal states s
            %
            for x = 1:size(map, 1)
                for y = 1:size(map, 2)
                    s = self.pos2I(x, y);
                    fprintf('(%d, %d) --> %d = ''%c''\n', x, y, s, map(x, y));
                    assert(ismember(s, S));
                    assert(ismember(s, I));
                    
                    R(s) = self.R_I; % time is money for internal states

                    if ismember(map(x, y), self.impassable_symbols)
                        % Impassable state e.g. wall -> no neighbors
                        %
                        continue;
                    end
                    
                    % Iterate over all adjacent states s --> s' and update passive
                    % dynamics P
                    %
                    for a = 1:size(adj, 1)
                        new_x = x + adj(a, 1);
                        new_y = y + adj(a, 2);
                        if new_x <= 0 || new_x > size(map, 1) || new_y <= 0 || new_y > size(map, 2)
                            continue % outside the map
                        end
                        if ismember(map(new_x, new_y), self.impassable_symbols)
                            continue; % impassable neighbor state s'
                        end
                        
                        new_s = self.pos2I(new_x, new_y);
                        fprintf('      (%d, %d) --(%d)--> %d = ''%c''\n', new_x, new_y, a, new_s, map(new_x, new_y));
                        assert(ismember(new_s, S));
                        assert(ismember(new_s, I));
                            
                        % transition f'n P(new_s|s,a)
                        % will normalize later
                        %
                        P(new_s, s, a) = 1;
                    end
                 
                    % Check if there's a corresponding boundary state
                    %
                    if I2B(s)
                        % There's also a boundary state in this square
                        %
                        assert(ismember(s, absorbing_inds));
                        assert(ismember(map(x, y), self.absorbing_symbols));

                        b = I2B(s);
                        P(b, s, 6) = 1; % action 6 = pick up reward
                        
                        % Get the reward for the boundary state
                        %
                        switch map(x, y)
                            case '$'
                                R(b) = 10; % $$$ #KNOB
                                
                            case '-'
                                R(b) = -Inf; % :( #KNOB
                                
                            otherwise
                                R(b) = str2num(map(x, y)); % e.g. 9 = $9 #KNOB
                        end   
                    end

                    % Normalize transition probabilities
                    %
                    for a = A
                        if sum(P(:, s, a)) == 0
                            P(s, s, a) = 1; % impossible moves keep you stationary
                        end
                        P(:, s, a) = P(:, s, a) / sum(P(:, s, a)); % normalize P(.|s,a)
                    end
                end
            end
            
            assert(isequal(I, setdiff(S, B)));
            
            self.P = P;
            self.R = R;
            self.Q = Q;
            self.V = V;
            self.H = H;
        end

        % Solve using generalized policy iteration.
        % Notice that the resulting policy is deterministic
        %
        function solveGPI(self)
            N = numel(self.S);
            pi = randint(numel(self.A), [N 1]);

            policy_stable = false;
            while ~policy_stable
                % policy evaluation
                %
                delta = Inf;
                while delta > self.GPI_threshold
                    delta = 0;
                    for s = self.S
                        v = self.V(s);
                        a = pi(s);
                        self.V(s) = sum(self.P(:, s, a) .* (self.R(:) + self.gamma * self.V(:)));
                        delta = max(delta, v - self.V(s));
                    end
                end

                % policy improvement
                %
                policy_stable = true;
                for s = self.S
                    old_a = pi(s);
                    [~, pi(s)] = max();
                end
            end

            self.pi = pi;
        end

        % Sample paths from deterministic policy pi generated using generalized policy iteration
        %
        function [Rtot, path] = sampleGPI(self, s)
            f
        end

        % Run an episode and update Q-values using SARSA
        %
        function [Rtot, path] = sampleSARSA(self, s)
            if ~exist('s', 'var')
                s = find(self.map == self.agent_symbol);
            end
            assert(numel(find(self.I == s)) == 1);

            Rtot = 0;
            path = [];

            a = self.eps_greedy(s);
            
            map = self.map;
            disp(map);
            while true
                Rtot = Rtot + self.R(s);
                path = [path, s];

                [x, y] = self.I2pos(s);
                
                new_s = samplePF(self.P(:,s,a));
                new_a = self.eps_greedy(new_s);

                oldQ = self.Q(s,a); % for debugging
                r = self.R(new_s);
                pe = r + self.gamma * self.Q(new_s, new_a) - self.Q(s, a);
                self.Q(s,a) = self.Q(s,a) + self.alpha * pe;
                
                if ismember(new_s, self.B)
                    % Boundary state
                    %
                fprintf('(%d, %d), %d --> END [%.2f%%], old Q = %.2f, pe = %.2f, Q = %.2f\n', x, y, a, self.P(new_s, s, a) * 100, oldQ, pe, self.Q(s, a));

                    Rtot = Rtot + self.R(new_s);
                    path = [path, new_s];
                    break;
                end

                % Internal state
                %
                [new_x, new_y] = self.I2pos(new_s);
                
                map(x, y) = self.empty_symbol;
                map(new_x, new_y) = self.agent_symbol;
                
                fprintf('(%d, %d), %d --> (%d, %d), %d [%.2f%%], old Q = %.2f, pe = %.2f, Q = %.2f\n', x, y, a, new_x, new_y, new_a, self.P(new_s, s, a) * 100, oldQ, pe, self.Q(s, a));
                disp(map);
                
                s = new_s;
                a = new_a;
            end
            fprintf('Total reward: %d\n', Rtot);
        end

        % Run an episode and update Q-values using Q-learning
        % TODO dedupe with sampleSARSA
        %
        function [Rtot, path] = sampleQ(self, s)
            if ~exist('s', 'var')
                s = find(self.map == self.agent_symbol);
            end
            assert(numel(find(self.I == s)) == 1);

            Rtot = 0;
            path = [];

            map = self.map;
            disp(map);
            while true
                Rtot = Rtot + self.R(s);
                path = [path, s];

                [x, y] = self.I2pos(s);
                
                a = self.eps_greedy(s);
                new_s = samplePF(self.P(:,s,a));
            
                oldQ = self.Q(s,a); % for debugging
                pe = self.R(new_s) + self.gamma * max(self.Q(new_s, :)) - self.Q(s, a);
                self.Q(s,a) = self.Q(s,a) + self.alpha * pe;
                
                if ismember(new_s, self.B)
                    % Boundary state
                    %
                fprintf('(%d, %d), %d --> END [%.2f%%], old Q = %.2f, pe = %.2f, Q = %.2f\n', x, y, a, self.P(new_s, s, a) * 100, oldQ, pe, self.Q(s, a));

                    Rtot = Rtot + self.R(new_s);
                    path = [path, new_s];
                    break;
                end

                % Internal state
                %
                [new_x, new_y] = self.I2pos(new_s);
                
                map(x, y) = self.empty_symbol;
                map(new_x, new_y) = self.agent_symbol;
                
                fprintf('(%d, %d), %d --> (%d, %d) [%.2f%%], old Q = %.2f, pe = %.2f, Q = %.2f\n', x, y, a, new_x, new_y, self.P(new_s, s, a) * 100, oldQ, pe, self.Q(s, a));
                disp(map);
                
                s = new_s;
            end
            fprintf('Total reward: %d\n', Rtot);
        end

        % Run an episode and update V-values and policy using actor-critic
        % TODO dedupe with sampleSARSA
        %
        function [Rtot, path] = sampleAC(self, s)
            if ~exist('s', 'var')
                s = find(self.map == self.agent_symbol);
            end
            assert(numel(find(self.I == s)) == 1);

            Rtot = 0;
            path = [];

            map = self.map;
            disp(map);
            while true
                Rtot = Rtot + self.R(s);
                path = [path, s];

                [x, y] = self.I2pos(s);
                
                a = self.softmax(s);
                new_s = samplePF(self.P(:,s,a));
            
                oldV = self.V(s); % for debugging
                pe = self.R(new_s) + self.gamma * self.V(new_s) - self.V(s);
                self.V(s) = self.V(s) + self.alpha * pe;
                self.H(s, a) = self.H(s, a) + self.beta * pe;
                
                if ismember(new_s, self.B)
                    % Boundary state
                    %
                    fprintf('(%d, %d), %d --> END [%.2f%%], old V = %.2f, pe = %.2f, V = %.2f\n', x, y, a, self.P(new_s, s, a) * 100, oldV, pe, self.V(s));

                    Rtot = Rtot + self.R(new_s);
                    path = [path, new_s];
                    break;
                end

                % Internal state
                %
                [new_x, new_y] = self.I2pos(new_s);
                
                map(x, y) = self.empty_symbol;
                map(new_x, new_y) = self.agent_symbol;
                
                fprintf('(%d, %d), %d --> (%d, %d) [%.2f%%], old V = %.2f, pe = %.2f, V = %.2f\n', x, y, a, new_x, new_y, self.P(new_s, s, a) * 100, oldV, pe, self.V(s));
                disp(map);
                
                s = new_s;
            end
            fprintf('Total reward: %d\n', Rtot);
        end

        % Pick action a from state s using eps-greedy based on Q-values
        % used for SARSA and Q-learning
        %
        function a = eps_greedy(self, s)
            [~, a] = max(self.Q(s,:));
            if numel(unique(self.Q(s,:))) == 1
                % no max => choose at random
                %
                a = datasample(self.A, 1);
            else
                % return best action
                %
                if rand() < self.eps
                    % with small probability eps, return another action at random
                    %
                    a = datasample(setdiff(self.A, a), 1);
                end
            end
        end

        % Pick action a from state s using softmax based on H policy parameters
        % used for actor-critic
        %
        function a = softmax(self, s)
            p = exp(self.H(s, :));
            p = p / sum(p);
            a = samplePF(p);
        end

        % Convert from maze position to internal state
        %
        function s = pos2I(self, x, y)
            s = sub2ind(size(self.map), x, y);
            assert(ismember(s, self.I));
        end

        % Convert from internal state to maze position
        %
        function [x, y] = I2pos(self, s)
            assert(ismember(s, self.I));
            [x, y] = ind2sub(size(self.map), s);
        end

    end
end
