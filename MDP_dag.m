% TD-learning (SARSA, Q-learning, and actor-critic) as in Sutton & Barto (2013)
% For 'rooms' domain only
% TODO dedupe with MDP
%
classdef MDP2 < handle

    properties (Constant = true)
        % General
        % 
        R_I = -1; % penalty for staying in one place
        alpha = 0.1; % learning rate
        gamma = 0.9; % discount rate
        eps = 0.1; % eps for eps-greedy
        beta = 0.1; % learning rate for policy (actor-critic)
        GPI_threshold = 0.1; % threshold for convergence of V(s) during policy evaluation

        % Maze
        %
        absorbing_symbols = '-0123456789$';
        agent_symbol = 'X';
        empty_symbol = '.';
        impassable_symbols = '#';
        % action 1 = stand still
        % actions 2,3,4,5 = move to adjacent squares
        % action 6 = pick up reward (to make it compatible with LMDPs)
        %
        % adjacency list
        % each row = [dx, dy, non-normalized P(s'|s)]
        % => random walk, but also bias towards staying in 1 place
        %
        adj = [ ...
            -1, 0; ...
            0, -1; ...
            1, 0; ...
            0, 1];
        A_names = {'up', 'left', 'down', 'right', 'eat'};
    end

    properties (Access = public)
        S = []; % all states
        S_names = {}; % state names

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

        % Maze stuff
        %
        map = [];
        
        % GUI
        %
        gui_state = []; % state for the GUI step-through
        gui_map = []; % figure for the GUI
        gui_timer = []; % timer for the GUI
    end
   
    methods

        % Initialize an MDP from a maze
        %
        function self = MDP2(lambda)
            if ~exist('lambda', 'var')
                lambda = 0;
            end
            self.lambda = lambda;
        end

        function init_from_DAG(self, next_keys, next_values, rewards)
            % find all unique states
            %
            S_names = {};
            for i = 1:numel(next_keys)
                key = next_keys{i};
                if ~ismember(key, S_names)
                    S_names = [S_names, {key}];
                end
                for val = next_values{i}
                    val = val{1};
                    if ~ismember(val, S_names)
                        S_names = [S_names, {val}];
                    end
                end
            end

            % set up state space S and action space A
            %
            N_S = numel(S_names);
            N_A = numel(next_values{1});
            S = 1:N_S;
            A = 1:N_A;

            % find boundary states B and internal states I
            %
            B = find(~ismember(S_names, next_keys));
            I = setdiff(S, B);

            % set up transition matrix P
            %
            P = zeros(N_S, N_S, N_A); % transitions P(s'|s,a); defaults to 0
            for i = 1:numel(next_keys)
                key = next_keys{i};
                s = find(strcmp(key, S_names));
                assert(~isempty(s));
                for a = A
                    val = next_values{i}{a};
                    new_s = find(strcmp(val, S_names));
                    assert(~isempty(new_s));
                    P(new_s, s, a) = 1;
                end
            end

            % set up reward structure
            %
            R = nan(N_S, 1); % instantaneous reward f'n R(s)
            assert(numel(keys(rewards)) == N_S);
            for key = keys(rewards)
                key = key{1};
                s = find(strcmp(key, S_names));
                assert(~isempty(s));
                val = rewards(key);
                R(s) = val;
            end

            % set up state and action values
            %
            Q = zeros(N_S, N_A); % action values Q(s, a)
            V = zeros(N_S, 1); % state values V(s)
            H = zeros(N_S, N_A); % policy parameters
            E_V = zeros(N_S, 1); % eligibility traces E(s) for state values
            E_Q = zeros(N_S, N_A); % eligibility traces E(s, a) for action values

            for s = S
                for a = A
                    if sum(P(:, s, a)) > 0 % allowed action
                        assert(sum(P(:, s, a)) == 1);
                    else % disallowed action
                        H(s, a) = -Inf;
                        Q(s, a) = -Inf;
                    end
                end
            end

            % update object
            %
            self.S = S;
            self.S_names = S_names;
            self.I = I;
            self.B = B;
            self.P = P;
            self.R = R;
            self.Q = Q;
            self.V = V;
            self.H = H;
            self.E_V = E_V;
            self.E_Q = E_Q;
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

        function sampleSARSA_gui(varargin)
            self = varargin{1};
            self.sample_gui_helper(@self.init_sampleSARSA, @self.stepSARSA, varargin{2:end});
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

        function sampleQ_gui(varargin)
            self = varargin{1};
            self.sample_gui_helper(@self.init_sampleQ, @self.stepQ, varargin{2:end});
        end

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
        % Run an episode and update V-values and policy using actor-critic
        %

        function sampleAC_gui(varargin)
            self = varargin{1};
            self.sample_gui_helper(@self.init_sampleAC, @self.stepAC, varargin{2:end});
        end

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

        %
        % Generic function that samples paths using a nice GUI
        %

        function sample_gui_helper(self, init_fn, step_fn, s)
            if ~exist('s', 'var')
                self.gui_state = init_fn();
            else
                self.gui_state = init_fn(s);
            end

			self.gui_map = figure;
            self.plot_gui();

            step_callback = @(hObject, eventdata) self.step_gui_callback(step_fn, hObject, eventdata);
            start_callback = @(hObject, eventdata) self.start_gui_callback(hObject, eventdata);
            reset_callback = @(hObject, eventdata) self.reset_gui_callback(init_fn, hObject, eventdata);
            stop_callback = @(hObject, eventdata) stop(self.gui_timer);
            sample_callback = @(hObject, eventdata) self.sample_gui_callback(step_fn, hObject, eventdata);

            self.gui_timer = timer('Period', 0.5, 'TimerFcn', step_callback, 'ExecutionMode', 'fixedRate', 'TasksToExecute', 1000000);
			uicontrol('Style', 'pushbutton', 'String', 'Start', ...
			  		 'Position', [10 50 + 90 40 20], ...
			  		 'Callback', start_callback);
			uicontrol('Style', 'pushbutton', 'String', 'Stop', ...
					 'Position', [10 50 + 70 40 20], ...
					 'Callback', stop_callback);
			uicontrol('Style', 'pushbutton', 'String', 'Reset', ...
					 'Position', [10 50 + 50 40 20], ...
					 'Callback', reset_callback);
			uicontrol('Style', 'pushbutton', 'String', 'Step', ...
			  		 'Position', [10 25 + 30 40 20], ...
			  		 'Callback', step_callback);
			uicontrol('Style', 'pushbutton', 'String', 'Skip', ...
					 'Position', [10 10 40 20], ...
					 'Callback', sample_callback);
        end

        % Single action
        %
        function step_gui_callback(self, step_fn, hObject, eventdata)
            if self.gui_state.done
                stop(self.gui_timer);
                return
            end

            [x, y] = self.I2pos(self.gui_state.s);
            old_s = self.gui_state.s;
            old_a = self.gui_state.a;

            self.gui_state = step_fn(self.gui_state);
            self.plot_gui();

            if self.gui_state.done
                fprintf('(%d, %d), %d --> END [%.2f%%]\n', x, y, old_a, self.P(self.gui_state.s, old_s, old_a) * 100);
            else
                [new_x, new_y] = self.I2pos(self.gui_state.s);
                map(x, y) = self.empty_symbol;
                map(new_x, new_y) = self.agent_symbol;
                fprintf('(%d, %d), %d --> (%d, %d) [%.2f%%]\n', x, y, old_a, new_x, new_y, self.P(self.gui_state.s, old_s, old_a) * 100);
            end
        end

        % Animate entire episode until the end
        %
        function start_gui_callback(self, hObject, eventdata)
            start(self.gui_timer);
        end

        % Reset the state
        %
        function reset_gui_callback(self, init_fn, hObject, eventdata)
            self.gui_state = init_fn(self.gui_state.path(1));
            self.plot_gui();
        end

        % Run entire episode until the end
        %
        function sample_gui_callback(self, step_fn, hObject, eventdata)
            while numel(self.gui_state) > 1 || ~self.gui_state.done
                self.gui_state = step_fn(self.gui_state);
            end
            self.plot_gui();
        end

        % Plot the GUI
        %
        function plot_gui(self)
            figure(self.gui_map);

            % plot map and rewards
            %
            subplot(2, 8, 1);
            m = self.map == '#';
            imagesc(reshape(m, size(self.map)));
            % goals
            goals = find(self.map == '$');
            for g = goals
                [x, y] = ind2sub(size(self.map), g);
                text(y, x, '$', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'green');
            end
            % agent
            [x, y] = ind2sub(size(self.map), self.gui_state.s);
            text(y, x, 'X', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'red');
            % label
            label = sprintf('Total reward: %.2f, steps: %d', self.gui_state.Rtot, numel(self.gui_state.path));
            if self.gui_state.done
                xlabel(['FINISHED!: ', label]);
            else
                xlabel(label);
            end
            ylabel(self.gui_state.method);
            title('map');

            % heat map of visited states
            %
            subplot(2, 8, 2);
            v = zeros(size(self.map));
            for s = self.gui_state.path
                if s <= numel(self.map) % i.e. s in I
                    v(s) = v(s) + 1;
                end
            end
            imagesc(reshape(v, size(self.map)));
            title('visited');

            % plot map and current state-value f'n V(s)
            %
            subplot(2, 8, 3);
            vi = self.V(self.I);
            imagesc(reshape(vi, size(self.map)));
            [x, y] = ind2sub(size(self.map), self.gui_state.s);
            text(y, x, 'X', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'red');
            title('state-value function V(.)');

            % current state-value eligibility trace f'n E(s)
            %
            subplot(2, 8, 4);
            ei = self.E_V(self.I);
            imagesc(reshape(ei, size(self.map)));
            [x, y] = ind2sub(size(self.map), self.gui_state.s);
            text(y, x, 'X', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'red');
            title('state-value eligibility traces E(.)');

            % current action-value eligibility trace f'n E(s)
            %
            subplot(2, 8, 5);
            ei = sum(self.E_Q(self.I, :), 2);
            imagesc(reshape(ei, size(self.map)));
            [x, y] = ind2sub(size(self.map), self.gui_state.s);
            text(y, x, 'X', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'red');
            title('action-value eligibility traces E(.,.)');


            % plot map and current action-value f'n max Q(s, a)
            %
            subplot(2, 8, 6);
            qi = self.Q(self.I, :);
            qi = max(qi, [], 2);
            imagesc(reshape(qi, size(self.map)));
            [x, y] = ind2sub(size(self.map), self.gui_state.s);
            text(y, x, 'X', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'red');
            title('action-value function max_a Q(.,a)');

            % plot map and transition probability across all possible actions, P(.|s)
            %
            subplot(2, 8, 7);
            pi = self.gui_state.pi';
            p = squeeze(self.P(self.I, self.gui_state.s, :));
            p = p * pi;
            imagesc(reshape(p, size(self.map)));
            [x, y] = ind2sub(size(self.map), self.gui_state.s);
            text(y, x, 'X', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'red');
            title('policy P(.|s)');

            % plot map and current transition probability given the selected action, P(.|s,a)
            %
            subplot(2, 8, 8);
            p = self.P(self.I, self.gui_state.s, self.gui_state.a);
            imagesc(reshape(p, size(self.map)));
            [x, y] = ind2sub(size(self.map), self.gui_state.s);
            text(y, x, 'X', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'red');
            title('transition probability for chosen action P(.|s,a)');

            % plot reward / PE history
            %
            rs = self.gui_state.rs;
            pes = self.gui_state.pes;
            %path = self.gui_state.path; <-- differs for SMDP
            hist = 10000;
            if numel(rs) > hist
                rs = rs(end-hist:end);
                pes = pes(end-hist:end);
                %path = path(end-hist:end);
            end
            subplot(2, 1, 2);
            plot(rs);
            hold on;
            plot(pes);
            hold off;
            legend('rewards', 'PEs');

        end

        % Pick action a from state s using eps-greedy based on Q-values
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
