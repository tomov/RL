% Two-layer options framework as in Sutton et al (1999) based on semi-MDPs.
% Works with 'rooms' domain only.
% Uses subgoals to define the options (as Sec 7)
%
classdef SMDP < handle

    properties (Constant = true)
        R_I = 0; % penalty for remaining stationary. TODO dedupe with MDP.R_I = 0 TODO why does -1 not work -- it keeps wanting to stay still

        % Maze
        %
        pseudoreward_symbol = '1';
        subtask_symbol = 'S';
    end

    properties (Access = public)
        O = []; % all options

        pi = {}; % pi{o}(:) = deterministic policy for option o: pi{o}(s) = what (lower-level) action to take at state s while executing option o. Roughly corresponds to Zi(o,:) which tells you the policy for subtask o in the MLMDP framework
        smdp = {}; % SMDP for each option
        mdp = []; % augmented MDP with the options as additional actions

        % Maze
        %
        map = [];
    end

    methods

        % Initialize an SMDP from a maze
        %
        function self = SMDP(map)
            self.map = map;

            %
            % Set up options and their policies based on subtask states S
            %

			subtask_inds = find(map == SMDP.subtask_symbol)';
            goal_inds = find(ismember(map, MDP.absorbing_symbols));

			map(subtask_inds) = MDP.empty_symbol; % erase subtask states
			map(goal_inds) = MDP.empty_symbol; % erase goal states

            O = 1:numel(subtask_inds); % 
            self.pi = cell(numel(O), 1); % set of policies for each option

            % For each subtask, find the optimal policy that gets you
            % from anywhere to its goal state. 
            %
			for s = subtask_inds
                % Set rewards of 0 (MDP.R_I) for all internal states
                % and reward of 1 at the subtask state
                %
                map(s) = SMDP.pseudoreward_symbol;

                T = MDP(map);
                T.solveGPI();
                
                o = find(subtask_inds == s);
                self.pi{o} = T.pi; % policy for option o corresponding to subtask with goal state s
                self.smdp{o} = T; % for debugging

                map(s) = MDP.empty_symbol;
			end

            %
            % Create a augmented MDP with the options
            %

            map = self.map;
            map(subtask_inds) = '.'; % erase subtask states

            T = MDP(map);

            O = numel(T.A) + 1 : numel(T.A) + numel(subtask_inds); % set of options = set of subtasks = set of subtask goal states; immediately follow the regular actions in indexing
            T.A = [T.A, O]; % augment actions with options

            % Augment transitions P(s'|s,a)
            %
            N = numel(T.S);
            T.Q = zeros(N, numel(T.A));
            for a = O % for each action that is an option
                o = find(O == a);
                s = subtask_inds(o);
                T.P(s, :, a) = 1; % from wherever we take the option, we end up at its goal state (b/c its policy is deterministic)
                T.P(s, s, a) = 0; % ...except from its goal state -- can't take the option there (makes no sense)
            end
            % sanity check them
            p = sum(T.P, 1);
            p = p(:);
            assert(sum(abs(p - 1) < 1e-8 | abs(p) < 1e-8) == numel(p));

            T.R(T.I) = SMDP.R_I; % penalty for staying still

            self.O = O;
            self.mdp = T;
        end 

        %
        % Run an episode and update Q-values using Q-learning
        %

        function sampleQ_gui(varargin)
            self = varargin{1};
            self.sample_gui_helper(@self.init_sampleQ, @self.stepQ, varargin{2:end}); % we're using the local one
        end

        function res = sampleQ(varargin)
            self = varargin{1};
            res = self.mdp.sample_helper(@self.init_sampleQ, @self.stepQ, varargin{2:end});
        end

        function state = init_sampleQ(self, s)
            if ~exist('s', 'var')
                s = find(self.mdp.map == self.mdp.agent_symbol);
            end
            assert(numel(find(self.mdp.I == s)) == 1);

            state.Rtot = 0;
            state.path = [];
            state.s = s;
            pi = self.mdp.eps_greedy(s);
            state.pi = pi;
            state.a = samplePF(pi);
            state.done = false;
            state.method = 'Q';
            state.r = 0;
            state.pe = 0;
            state.in_option = ismember(state.a, self.O); % for GUI
        end

        function state = stepQ(self, state)
            s = state.s;
            a = state.a;

            state.Rtot = state.Rtot + self.mdp.R(s);
            state.path = [state.path, s];

            % Sample new state and new action
            %
            new_s = samplePF(self.mdp.P(:,s,a));
            pi = self.mdp.eps_greedy(new_s);
            new_a = NaN;
            while isnan(new_a) || max(self.mdp.P(:, new_s, new_a)) == 0 % this means the action is not allowed; we could be more explicit about this TODO
                new_a = samplePF(pi);
            end
           
            % Compute PE
            %
            oldQ = self.mdp.Q(s,a); % for debugging
            if ismember(a, self.O)
                % option -> execute option policy until the end
                %
                o = find(self.O == a);
                [~, option_path] = self.smdp{o}.sampleGPI(s); % execute option policy
                option_path = option_path(2:end-1); % don't count s and fake B state
                k = numel(option_path);
                assert(k > 0); % b/c can't take an option from its goal state
                state.path = [state.path, option_path(1:end-1)];

                r = self.mdp.R(option_path) .* MDP.gamma .^ (0:k-1)'; % from Sec 3.2 of Sutton (1999)
                fprintf('       option! %d -> path = %s, r = %s\n', o, sprintf('%d ', option_path), sprintf('%.2f ', r));
                r = sum(r);
                pe = sum(r) + (MDP.gamma ^ k) * max(self.mdp.Q(new_s, :)) - self.mdp.Q(s, a);
            else
                % primitive action -> regular Q learning
                %
                r = self.mdp.R(new_s);
                pe = r + MDP.gamma * max(self.mdp.Q(new_s, :)) - self.mdp.Q(s, a);
            end

            % Update Q values
            %
            self.mdp.Q(s,a) = self.mdp.Q(s,a) + self.mdp.alpha * pe;

            % Check for boundary conditions
            %
            if ismember(new_s, self.mdp.B)
                % Boundary state
                %
                state.Rtot = state.Rtot + self.mdp.R(new_s);
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

            % for GUI
            %
            state.in_option = ismember(new_a, self.O);
        end

        %
        % Generic function that samples paths using a nice GUI
        %

        % TODO dedupe with MDP 
        function sample_gui_helper(self, init_fn, step_fn, s)
            if ~exist('s', 'var')
                self.mdp.gui_state = init_fn();
            else
                self.mdp.gui_state = init_fn(s);
            end

			self.mdp.gui_map = figure;
            self.mdp.plot_gui();

            step_callback = @(hObject, eventdata) self.mdp.step_gui_callback(step_fn, hObject, eventdata);
            recursive_step_callback = @(hObject, eventdata) self.step_gui_callback(step_fn, hObject, eventdata); % notice this is differnet -- it's the local stepping one so we can customize it for the options
            start_callback = @(hObject, eventdata) self.mdp.start_gui_callback(hObject, eventdata);
            reset_callback = @(hObject, eventdata) self.mdp.reset_gui_callback(init_fn, hObject, eventdata);
            stop_callback = @(hObject, eventdata) stop(self.mdp.gui_timer);
            sample_callback = @(hObject, eventdata) self.mdp.sample_gui_callback(step_fn, hObject, eventdata);

            self.mdp.gui_timer = timer('Period', 0.1, 'TimerFcn', step_callback, 'ExecutionMode', 'fixedRate', 'TasksToExecute', 1000000);
			uicontrol('Style', 'pushbutton', 'String', 'Start', ...
			  		 'Position', [10 90 70 20], ...
			  		 'Callback', start_callback);
			uicontrol('Style', 'pushbutton', 'String', 'Stop', ...
					 'Position', [10 70 70 20], ...
					 'Callback', stop_callback);
			uicontrol('Style', 'pushbutton', 'String', 'Reset', ...
					 'Position', [10 50 70 20], ...
					 'Callback', reset_callback);
			uicontrol('Style', 'pushbutton', 'String', 'Step', ...
			  		 'Position', [10 30 70 20], ...
			  		 'Callback', recursive_step_callback);
			uicontrol('Style', 'pushbutton', 'String', 'Skip', ...
					 'Position', [10 10 70 20], ...
					 'Callback', sample_callback);
        end

        % TODO Dedupe with MDP
        function step_gui_callback(self, step_fn, hObject, eventdata)
            if self.mdp.gui_state.done
                return
            end

            % See if we're about to execute an option
            % and visualize it nicely
            %
            s = self.mdp.gui_state.s; 
            a = self.mdp.gui_state.a;
            if self.mdp.gui_state.in_option
                assert(ismember(a, self.O));
                self.mdp.gui_state.in_option = false; % next time just move on
                o = find(self.O == a);
                fprintf('  ...executing action %d = option %d, from state %d', a, o, s);
                self.smdp{o}.sampleGPI_gui(s);
            else
                self.mdp.step_gui_callback(step_fn, hObject, eventdata);
            end
        end
    end
end
