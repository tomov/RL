% Two-layer options framework as in Sutton et al (1999) based on semi-MDPs.
% Can also be used with the hierarchical semi-Markov Q-learning (HSMQ) of Dietterich (2000) by passing is_hsm = true.
% Works with 'rooms' domain only.
% Uses subgoals to define the options (as Sec 7)
%
classdef SMDP < handle

    properties (Constant = true)
        R_I = -1; % penalty for remaining stationary. TODO dedupe with MDP.R_I 

        % Maze
        %
        pseudoreward_symbol = '1';
        subtask_symbol = 'S';

        % DAG
        %
        R_pseudoreward = 1;
    end

    properties (Access = public)
        O = []; % all options

        pi = {}; % pi{o}(:) = deterministic policy for option o: pi{o}(s) = what (lower-level) action to take at state s while executing option o. Roughly corresponds to Zi(o,:) which tells you the policy for subtask o in the MLMDP framework
        smdp = {}; % SMDP for each option
        mdp = []; % augmented MDP with the options as additional actions

        % whether to solve with HSMQ.
        % If true, solve subtask SMDP's online along with the whole MDP (HSMQ algorithm, Dietterich 2000).
        % If false, pre-solve SMDP's and only solve the MDP online (Options framework, Sutton 1999).
        %
        is_hsm = false;

        % Maze
        %
        map = [];

        % DAG
        %
        next_keys = []; % for convenience
        next_values = [];
        DAG = []; % actual DAG
        DAG_trans = []; % transitive closure of the DAG to check for connectivity

        is_maze = false;
        is_dag = false;
    end

    methods

        function self = SMDP(is_hsm)
            if ~exist('is_hsm', 'var')
                is_hsm = false; % by default, we pre-solve the SMDP's i.e. we don't do HSMQ
            end
            self.is_hsm = is_hsm;
        end

        % Initialize an SMDP from a maze
        %
        function init_from_maze(self, map)
            self.is_maze = true;
            self.is_dag = false;
            self.map = map;

            %
            % Set up options and their policies based on subtask states S
            %

			subtask_inds = find(map == SMDP.subtask_symbol)';
            goal_inds = find(ismember(map, MDP_maze.absorbing_symbols));

			map(subtask_inds) = MDP_maze.empty_symbol; % erase subtask states
			map(goal_inds) = MDP_maze.empty_symbol; % erase goal states

            O = 1:numel(subtask_inds); 
            self.pi = cell(numel(O), 1); % set of policies for each option

            % For each subtask, find the optimal policy that gets you
            % from anywhere to its goal state. 
            %
			for s = subtask_inds
                % Set rewards of 0 (MDP.R_I) for all internal states
                % and reward of 1 at the subtask state
                %
                map(s) = SMDP.pseudoreward_symbol;

                smdp = MDP_maze(map);
                if ~self.is_hsm
                    smdp.solveGPI(); % in regular Options framework, we assume the policies of the subtasks are given
                end
                
                o = find(subtask_inds == s);
                self.pi{o} = smdp.pi; % policy for option o corresponding to subtask with goal state s
                self.smdp{o} = smdp;

                map(s) = MDP_maze.empty_symbol;
			end

            %
            % Create a augmented MDP with the options
            %

            map = self.map;
            map(subtask_inds) = '.'; % erase subtask states

            mdp = MDP_maze(map);

            O = numel(mdp.A) + 1 : numel(mdp.A) + numel(subtask_inds); % set of options = set of subtasks = set of subtask goal states; immediately follow the regular actions in indexing
            mdp.A = [mdp.A, O]; % augment actions with options

            % Augment transitions P(s'|s,a)
            %
            N = numel(mdp.S);
            mdp.Q = zeros(N, numel(mdp.A));
            mdp.E_Q = zeros(N, numel(mdp.A));
            mdp.H = zeros(N, numel(mdp.A));
            for a = O % for each action that is an option
                o = find(O == a);
                s = subtask_inds(o);
                mdp.P(s, :, a) = 1; % from wherever we take the option, we end up at its goal state (b/c its policy is deterministic)
                mdp.P(s, s, a) = 0; % ...except from its goal state -- can't take the option there (makes no sense)
            end

            mdp.normalize_P();

            mdp.R(mdp.I) = SMDP.R_I; % penalty for staying still

            self.O = O;
            self.mdp = mdp;
        end 

        % Initialize an MDP from a DAG
        % TODO dedupe with init_from_maze
        %
        function init_from_dag(self, next_keys, next_values, rewards, subtask_states)
            self.is_dag = true;
            self.is_maze = false;
            self.next_keys = next_keys;
            self.next_values = next_values;

            % Create MDP that we'll later augment with options
            %
            mdp = MDP_dag(next_keys, next_values, rewards);
            self.mdp = mdp;

            % get DAG
            %
            adj = sum(self.mdp.P, 3) > 0;
            self.DAG = digraph(adj', self.mdp.S_names);
            self.DAG_trans = transclosure(self.DAG);

            % Set up options and their policies based on subtask states
            %
            self.O = [];
            for subtask_state = subtask_states
                self.add_option(subtask_state);
            end
        end

        % add an option to an already existing SMDP
        %
        function add_option(self, subtask_state)
            assert(self.is_dag);

            s = self.mdp.get_state_by_name(subtask_state);

            % For each subtask, find the optimal policy that gets you
            % from anywhere to its goal state. 
            %
            prs = SMDP.R_I * ones(numel(self.mdp.S), 1);
            prs(s) = SMDP.R_pseudoreward;
            pseudorewards = containers.Map(self.mdp.S_names, prs);

            smdp = MDP_dag(self.next_keys, self.next_values, pseudorewards);
            assert(~ismember(s, smdp.B));
            smdp.B = [smdp.B, s]; % add s to terminal states
            smdp.I = setdiff(smdp.S, smdp.B);
            if ~self.is_hsm
                smdp.solveGPI(); % in regular Options framework, we assume the policies of the subtasks are given
            end

            o = numel(self.O) + 1;
            self.pi{o} = smdp.pi; % policy for option o corresponding to subtask with goal state s
            self.smdp{o} = smdp;

            %
            % Augment the main MDP with the option
            %

            a = numel(self.mdp.A) + 1;
            self.O = [self.O, a];
            self.mdp.A = [self.mdp.A, a]; % augment MDP actions with new option

            % Augment transitions P(s'|s,a)
            %
            N = numel(self.mdp.S);
            self.mdp.Q = [self.mdp.Q, zeros(N,1)];
            self.mdp.E_Q = [self.mdp.E_Q, zeros(N,1)];
            self.mdp.H = [self.mdp.H, zeros(N,1)];

            % find states that can get to 
            %
            for old_s = 1:numel(self.mdp.S)
                if findedge(self.DAG_trans, self.mdp.S_names{old_s}, self.mdp.S_names{s});
                    self.mdp.P(s, old_s, a) = 1;
                    fprintf('     %d (%s) --> %d (%s)\n', old_s, self.mdp.S_names{old_s}, s, self.mdp.S_names{s});
                else
                    self.mdp.P(s, old_s, a) = 0;
                end
            end
            self.mdp.P(s, s, a) = 0; % policy not available in goal state
            self.mdp.normalize_P();
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
            assert(numel(find(self.mdp.I == s)) == 1);

            state.Rtot = 0;
            state.path = [];
            state.rs = [];
            state.pes = [];
            state.s = s;
            pi = self.mdp.eps_greedy(s);
            state.pi = pi;
            state.a = samplePF(pi);
            state.done = false;
            state.method = 'Options-Q';
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
                %
                % option -> execute option policy until the end
                %
                o = find(self.O == a);

                % sample a path from the policy of the option
                %
                if self.is_hsm
                    % HSMQ algorithm (Dietterich 2000) -> subtask policies are learnt online
                    %
                    [~, option_path] = self.smdp{o}.sampleQ(s);
                else
                    % Options framework (Sutton 1999) -> subtask policies are given (precomputed)
                    %
                    [~, option_path] = self.smdp{o}.sampleGPI(s);
                end

                option_path = option_path(2:end-1); % don't count s (starting) and fake B (terminal) state
                k = numel(option_path);
                assert(k > 0); % b/c can't take an option from its goal state
                state.path = [state.path, option_path(1:end-1)];

                rs = self.mdp.R(option_path) .* MDP.gamma .^ (0:k-1)'; % from Sec 3.2 of Sutton (1999)
                fprintf('       option! %d -> path = %s, rs = %s\n', o, sprintf('%d ', option_path), sprintf('%.2f ', rs));
                r = sum(rs);
                pe = r + (MDP.gamma ^ k) * max(self.mdp.Q(new_s, :)) - self.mdp.Q(s, a);

                state.Rtot = state.Rtot + sum(self.mdp.R(option_path(1:end-1)));
            else
                %
                % primitive action -> regular Q learning
                %
                r = self.mdp.R(new_s);
                fprintf('     nonoption %d -> go to %d, r = %.2f\n', a, new_s, r);
                pe = r + MDP.gamma * max(self.mdp.Q(new_s, :)) - self.mdp.Q(s, a);
            end

            % Update Q values
            %
            self.mdp.Q(s,a) = self.mdp.Q(s,a) + MDP.alpha * pe;

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
            state.rs = [state.rs, r];
            state.pes = [state.pes, pe];

            % for GUI
            %
            state.in_option = ismember(new_a, self.O);
        end

        %
        % Run an episode and update state-values and policy using actor-critic
        %

        function sampleAC_gui(varargin)
            self = varargin{1};
            self.sample_gui_helper(@self.init_sampleAC, @self.stepAC, varargin{2:end}); % we're using the local one
        end

        function [Rtot, path] = sampleAC(varargin)
            self = varargin{1};
            [Rtot, path] = self.mdp.sample_helper(@self.init_sampleAC, @self.stepAC, varargin{2:end});
        end

        function state = init_sampleAC(self, s)
            assert(numel(find(self.mdp.I == s)) == 1);

            state.Rtot = 0;
            state.path = [];
            state.rs = [];
            state.pes = [];
            state.s = s;
            pi = self.mdp.softmax(s);
            state.pi = pi;
            state.a = samplePF(pi);
            state.done = false;
            state.method = 'Options-AC';
            state.r = 0;
            state.pe = 0;
            state.in_option = ismember(state.a, self.O); % for GUI

            self.mdp.E_V(:) = 0;
            self.mdp.E_Q(:) = 0;
        end

        function state = stepAC(self, state)
            s = state.s;
            a = state.a;

            state.Rtot = state.Rtot + self.mdp.R(s);
            state.path = [state.path, s];

            new_s = samplePF(self.mdp.P(:,s,a));
            new_a = NaN;
            if ~ismember(new_s, self.mdp.B)
                pi = self.mdp.softmax(new_s);
                new_a = samplePF(pi);
            end

            % Sample new state and new action
            %
            new_s = samplePF(self.mdp.P(:,s,a));
            pi = self.mdp.softmax(new_s);

            % Compute PE
            %
            oldQ = self.mdp.Q(s,a); % for debugging
            if ismember(a, self.O)
                %
                % option -> execute option policy until the end
                %
                o = find(self.O == a);

                % sample a path from the policy of the option
                %
                if self.is_hsm
                    % HSMQ algorithm (Dietterich 2000) -> subtask policies are learnt online
                    %
                    [~, option_path] = self.smdp{o}.sampleAC(s);
                else
                    % Options framework (Sutton 1999) -> subtask policies are given (precomputed)
                    %
                    [~, option_path] = self.smdp{o}.sampleGPI(s);
                end

                if self.is_maze
                    option_path = option_path(2:end-1); % don't count s (starting) and fake B (terminal) state
                else
                    assert(self.is_dag);
                    option_path = option_path(2:end); % don't count s (starting) state
                end
                k = numel(option_path);
                assert(k > 0); % b/c can't take an option from its goal state
                state.path = [state.path, option_path(1:end-1)];

                rs = self.mdp.R(option_path) .* MDP.gamma .^ (0:k-1)'; % from Sec 3.2 of Sutton (1999)
                if self.is_maze
                    fprintf('       option! %d -> path = %s, rs = %s\n', o, sprintf('%d ', option_path), sprintf('%.2f ', rs));
                else
                    assert(self.is_dag);
                    fprintf('       option! %d -> path = %s, rs = %s\n', o, sprintf('%s ', self.mdp.S_names{option_path}), sprintf('%.2f ', rs));
                end
                r = sum(rs);
                pe = r + (MDP.gamma ^ k) * self.mdp.V(new_s) - self.mdp.V(s);

                state.Rtot = state.Rtot + sum(self.mdp.R(option_path(1:end-1)));
            else
                %
                % primitive action -> regular AC
                %
                r = self.mdp.R(new_s);
                fprintf('     nonoption %d -> go to %d, r = %.2f\n', a, new_s, r);
                pe = r + MDP.gamma * self.mdp.V(new_s) - self.mdp.V(s);
            end
       
            % TD(0) -- for sanity checks
            %self.V(s) = self.V(s) + self.alpha * pe;
            %self.H(s, a) = self.H(s, a) + self.beta * pe;

            % update state values
            self.mdp.E_V(s) = self.mdp.E_V(s) + 1;
            self.mdp.V = self.mdp.V + MDP.alpha * pe * self.mdp.E_V;
            self.mdp.E_V = self.mdp.E_V * MDP.gamma * self.mdp.lambda;

            % update policies
            self.mdp.E_Q(s, a) = self.mdp.E_Q(s, a) + 1;
            self.mdp.H = self.mdp.H + MDP.beta * pe * self.mdp.E_Q;
            self.mdp.E_Q = self.mdp.E_Q * MDP.gamma * self.mdp.lambda;
            
            if ismember(new_s, self.mdp.B)
                % Boundary state
                %
                state.Rtot = state.Rtot + self.mdp.R(new_s);
                state.path = [state.path, new_s];
                state.done = true;
            else
                % Internal state
                %
                state.a = new_a;
                state.r = r;
                state.pe = pe;
                state.pi = pi;
            end

            state.s = new_s;
            state.rs = [state.rs, r];
            state.pes = [state.pes, pe];

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
			  		 'Callback', recursive_step_callback);
			uicontrol('Style', 'pushbutton', 'String', 'Skip', ...
					 'Position', [10 10 40 20], ...
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
                fprintf('  ...executing action %d = option %d, from state %d\n', a, o, s);
                if self.is_hsm
                    % WARNING: this will update the Q values of the option, i.e. if you step through this
                    % for every option, you will effectively be running two iterations of Q-learning.
                    % This is not necessarily a bad thing in the GUI as we want things to converge faster anyway,
                    % it will just be slightly different than the non-GUI case
                    % 
                    if endsWith(self.mdp.gui_state.method, 'Q')
                        self.smdp{o}.sampleQ_gui(s);
                    else
                        self.smdp{o}.sampleAC_gui(s);
                    end
                else
                    self.smdp{o}.sampleGPI_gui(s);
                end
            else
                self.mdp.step_gui_callback(step_fn, hObject, eventdata);
            end
        end

        function plot_subtasks(self)
            figure;
            for o = 1:numel(self.smdp)
                subplot(1, numel(self.smdp), o);
                self.plot_helper(self.smdp.R, ['Subtask ', num2str(o)], '', 'R(s)');

                subplot(2, numel(self.smdp), o);
                self.plot_helper(self.smdp.V, '', '', 'V(s)');
            end
        end

        %
        % More helper f'ns
        %

    end
end
