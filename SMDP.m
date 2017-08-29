% Two-layer options framework as in Sutton et al (1999) based on semi-MDPs.
% Can also be used with the hierarchical semi-Markov (HSM) framework of Dietterich (2000) by passing is_hsm = true.
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
    end

    properties (Access = public)
        O = []; % all options

        pi = {}; % pi{o}(:) = deterministic policy for option o: pi{o}(s) = what (lower-level) action to take at state s while executing option o. Roughly corresponds to Zi(o,:) which tells you the policy for subtask o in the MLMDP framework
        smdp = {}; % SMDP for each option
        mdp = []; % augmented MDP with the options as additional actions

        % whether to solve as a HSM.
        % If true, solve subtask SMDP's online along with the whole MDP (HSM framework, Dietterich 2000).
        % If false, pre-solve SMDP's and only solve the MDP online (Options framework, Sutton 1999).
        %
        is_hsm = false;

        % Maze
        %
        map = [];
    end

    methods

        % Initialize an SMDP from a maze
        %
        function self = SMDP(map, is_hsm)
            self.map = map;
            if ~exist('is_hsm', 'var')
                is_hsm = false; % by default, we pre-solve the SMDP's i.e. we don't do HSM
            end

            %
            % Set up options and their policies based on subtask states S
            %

			subtask_inds = find(map == SMDP.subtask_symbol)';
            goal_inds = find(ismember(map, MDP.absorbing_symbols));

			map(subtask_inds) = MDP.empty_symbol; % erase subtask states
			map(goal_inds) = MDP.empty_symbol; % erase goal states

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

                smdp = MDP(map);
                if ~is_hsm
                    smdp.solveGPI(); % in regular Options framework, we assume the policies of the subtasks are given
                end
                
                o = find(subtask_inds == s);
                self.pi{o} = smdp.pi; % policy for option o corresponding to subtask with goal state s
                self.smdp{o} = smdp;

                map(s) = MDP.empty_symbol;
			end
            self.is_hsm = is_hsm;

            %
            % Create a augmented MDP with the options
            %

            map = self.map;
            map(subtask_inds) = '.'; % erase subtask states

            mdp = MDP(map);

            O = numel(mdp.A) + 1 : numel(mdp.A) + numel(subtask_inds); % set of options = set of subtasks = set of subtask goal states; immediately follow the regular actions in indexing
            mdp.A = [mdp.A, O]; % augment actions with options

            % Augment transitions P(s'|s,a)
            %
            N = numel(mdp.S);
            mdp.Q = zeros(N, numel(mdp.A));
            for a = O % for each action that is an option
                o = find(O == a);
                s = subtask_inds(o);
                mdp.P(s, :, a) = 1; % from wherever we take the option, we end up at its goal state (b/c its policy is deterministic)
                mdp.P(s, s, a) = 0; % ...except from its goal state -- can't take the option there (makes no sense)
            end
            % sanity check them
            p = sum(mdp.P, 1);
            p = p(:);
            assert(sum(abs(p - 1) < 1e-8 | abs(p) < 1e-8) == numel(p));

            mdp.R(mdp.I) = SMDP.R_I; % penalty for staying still

            self.O = O;
            self.mdp = mdp;
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
            state.rs = [];
            state.pes = [];
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
                %
                % option -> execute option policy until the end
                %
                o = find(self.O == a);

                % sample a path from the policy of the option
                %
                if self.is_hsm
                    % HSM framework (Dietterich 2000) -> subtask policies are learnt online
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
                    self.smdp{o}.sampleQ_gui(s);
                else
                    self.smdp{o}.sampleGPI_gui(s);
                end
            else
                self.mdp.step_gui_callback(step_fn, hObject, eventdata);
            end
        end
    end
end
