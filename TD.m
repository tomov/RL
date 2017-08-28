% TD-learning (SARSA, Q-learning, and actor-critic) as in Sutton & Barto (2013)
% For 'rooms' domain only
% TODO dedupe with LMDP
%
classdef TD < handle

    properties (Constant = true)
        % General
        % 
        R_I = 0; % penalty for staying in one place
        alpha = 0.5; % learning rate
        gamma = 0.9; % discount rate
        eps = 0.1; % eps for eps-greedy
        beta = 0.5; % learning rate for policy (actor-critic)
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

        % Maze stuff
        %
        map = [];

        % GUI
        %
        state_gui = []; % state for the GUI step-through
        map_gui = []; % figure for the GUI
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
            I2B(I_with_B) = B; % mapping from I states to corresponding B states

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
                    %fprintf('(%d, %d) --> %d = ''%c''\n', x, y, s, map(x, y));
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
                        %fprintf('      (%d, %d) --(%d)--> %d = ''%c''\n', new_x, new_y, a, new_s, map(new_x, new_y));
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
        % TODO FIXME if you have negative internal rewards,
        % it's never going to terminate your initial random policy doesn't access any terminal states => the V's will keep getting more and more negative forever...
        % ALSO you need to have a discount factor < 1, otherwise
        % it gets all screwed up -- first the closest-to-B state picks the right action but then once its neighbors have the same value as the goal state, it starts going to them instead...
        %
        function solveGPI(self)
            N = numel(self.S);
            pi = randi(numel(self.A), [N 1]);

            self.V(self.B) = self.R(self.B); % boundary V(s) = R(s)

            assert(min(self.R(self.I)) >= 0); % doesn't work with negative internal rewards -> V(s)'s keep going towards -infinity forever
            assert(self.gamma < 1); % doesn't work with gamma = 1 -> the states start going around in circles after a few iterations

            policy_stable = false;
            iter = 0;
            while ~policy_stable
                % policy evaluation
                %
                delta = Inf;
                while delta > self.GPI_threshold
                    delta = 0;
                    %fprintf('\n');
                    for s = self.I
                        v = self.V(s);
                        a = pi(s);
                        self.V(s) = sum(self.P(:, s, a) .* (self.R(s) + self.gamma * self.V(:)));
                        delta = max(delta, abs(v - self.V(s)));
                        %fprintf('%d: %.2f -> %.2f (action %d), delta %.2f\n', s, v, self.V(s), a, delta);
                    end
                end
                %disp(self.V');

                % policy improvement
                %
                policy_stable = true;
                for s = self.I
                    a = pi(s);
                    r = squeeze(sum(self.P(:, s, :) .* (self.R(s) + self.gamma * self.V(:)), 1));
                    %fprintf('  -- %d: %s\n', s, sprintf('%d ', r));
                    assert(numel(r) == numel(self.A));
                    [~, pi(s)] = max(r);
                    if pi(s) ~= a
                        policy_stable = false;
                    end
                end
                %disp(pi');
            end

            self.pi = pi;
        end

        %
        % Sample paths from deterministic policy pi generated using generalized policy iteration
        %

        function sampleGPI_gui(varargin)
            self = varargin{1};
            self.sample_gui_helper(@self.init_sampleGPI, @self.stepGPI, varargin{2:end});
        end

        function res = sampleGPI(varargin)
            self = varargin{1};
            res = self.sample_helper(@self.init_sampleGPI, @self.stepGPI, varargin{2:end});
        end

        function state = init_sampleGPI(self, s)
            if ~exist('s', 'var')
                s = find(self.map == self.agent_symbol);
            end
            assert(numel(find(self.I == s)) == 1);

            state.Rtot = 0;
            state.path = [];
            state.s = s;
            state.a = -1;
            state.done = false;
        end

        function state = stepGPI(self, state)
            state.Rtot = state.Rtot + self.R(state.s);
            state.path = [state.path, state.s];

            a = self.pi(state.s);
            new_s = samplePF(self.P(:, state.s, a));

            if ismember(new_s, self.B)
                % Boundary state
                %
                state.Rtot = state.Rtot + self.R(new_s);
                state.path = [state.path, new_s];
                state.done = true;
            else
                % Internal state
                %
                state.a = a;
                state.s = new_s;
            end
        end

        %
        % Run an episode and update Q-values using SARSA
        %

        function res = sampleSARSA(varargin)
            self = varargin{1};
            res = self.sample_helper(@self.init_sampleSARSA, @self.stepSARSA, varargin{2:end});
        end

        function state = init_sampleSARSA(self, s)
            if ~exist('s', 'var')
                s = find(self.map == self.agent_symbol);
            end
            assert(numel(find(self.I == s)) == 1);

            state.Rtot = 0;
            state.path = [];
            state.s = s;
            state.a = self.eps_greedy(s); % notice for SARSA, a is the action we took from state s, NOT the action that got us to state s (unlike for the others)
            state.done = false;
            state.r = 0;
            state.pe = 0;
        end
           
        function state = stepSARSA(self, state)
            s = state.s;
            a = state.a;

            state.Rtot = state.Rtot + self.R(s);
            state.path = [state.path, s];

            new_s = samplePF(self.P(:, s, a));
            new_a = self.eps_greedy(new_s);

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
            end
        end

        %
        % Run an episode and update Q-values using Q-learning
        %

        function res = sampleQ(varargin)
            self = varargin{1};
            res = self.sample_helper(@self.init_sampleQ, @self.stepQ, varargin{2:end});
        end

        function state = init_sampleQ(self, s)
            if ~exist('s', 'var')
                s = find(self.map == self.agent_symbol);
            end
            assert(numel(find(self.I == s)) == 1);

            state.Rtot = 0;
            state.path = [];
            state.s = s;
            state.a = -1;
            state.done = false;
            state.r = 0;
            state.pe = 0;
        end

        function state = stepQ(self, state)
            s = state.s;

            state.Rtot = state.Rtot + self.R(s);
            state.path = [state.path, s];

            a = self.eps_greedy(s);
            new_s = samplePF(self.P(:,s,a));
       
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
            end
        end

        %
        % Run an episode and update V-values and policy using actor-critic
        %

        function res = sampleAC(varargin)
            self = varargin{1};
            res = self.sample_helper(@self.init_sampleAC, @self.stepAC, varargin{2:end});
        end

        function state = init_sampleAC(self, s)
            if ~exist('s', 'var')
                s = find(self.map == self.agent_symbol);
            end
            assert(numel(find(self.I == s)) == 1);

            state.Rtot = 0;
            state.path = [];
            state.s = s;
            state.a = -1;
            state.done = false;
            state.r = 0;
            state.pe = 0;
        end

        function state = stepAC(self, state)
            s = state.s;

            state.Rtot = state.Rtot + self.R(s);
            state.path = [state.path, s];

            a = self.softmax(s);
            new_s = samplePF(self.P(:,s,a));
       
            r = self.R(new_s);
            pe = r + self.gamma * self.V(new_s) - self.V(s);
            self.V(s) = self.V(s) + self.alpha * pe;
            self.H(s, a) = self.H(s, a) + self.beta * pe;
            
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
            end
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
            while ~state.done
                [x, y] = self.I2pos(state.s);
                old_s = state.s;
                old_a = state.a;
               
                state = step_fn(state); 

                if state.done
                    %if do_print, fprintf('(%d, %d), %d --> END [%.2f%%]\n', x, y, old_a, self.P(state.s, old_s, old_a) * 100); end
                else
                    [new_x, new_y] = self.I2pos(state.s);
                    map(x, y) = self.empty_symbol;
                    map(new_x, new_y) = self.agent_symbol;
                    %if do_print, fprintf('(%d, %d), %d --> (%d, %d) [%.2f%%]\n', x, y, old_a, new_x, new_y, self.P(state.s, old_s, old_a) * 100); end
                    if do_print, disp(map); end
                end
            end
            if do_print, fprintf('Total reward: %d\n', state.Rtot); end

            Rtot = state.Rtot;
            path = state.path;
        end

        %
        % Generic function that samples paths using a nice GUI
        %

        function sample_gui_helper(self, init_fn, step_fn, s)
            if ~exist('s', 'var')
                self.state_gui = init_fn();
            else
                self.state_gui = init_fn(s);
            end

			self.map_gui = figure;
            self.plot_gui();

            step_callback = @(hObject, eventdata) self.step_gui_callback(step_fn, hObject, eventdata);

			step_button = uicontrol('Style', 'pushbutton', 'String', 'Step', ...
									 'Position', [10 10 70 20], ...
									 'Callback', step_callback);
        end

        function step_gui_callback(self, step_fn, hObject, eventdata, handles)
            if self.state_gui.done
                return
            end

            self.state_gui = step_fn(self.state_gui);
            self.plot_gui();
        end

        function plot_gui(self)
            figure(self.map_gui);

            % plot map and current value f'n
            %
            subplot(1, 2, 1);
            vi = self.V(self.I);
            imagesc(log(reshape(vi, size(self.map))));
            [x, y] = ind2sub(size(self.map), self.state_gui.s);
            text(y, x, 'X', 'FontSize', 10, 'FontWeight', 'bold');
            if self.state_gui.done
                xlabel(sprintf('FINISHED Total reward: %.2f', self.state_gui.Rtot));
            else
                xlabel(sprintf('Total reward: %.2f', self.state_gui.Rtot));
            end
            title('Value function V');
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
