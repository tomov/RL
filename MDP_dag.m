% TD-learning for DAG
%
classdef MDP_dag < MDP

    properties (Constant = true)
    end

    properties (Access = public)
        S_names = {}; % state names

        % GUI
        %
        gui_state = []; % state for the GUI step-through
        gui_map = []; % figure for the GUI
        gui_timer = []; % timer for the GUI
    end
   
    methods

        % Initialize an MDP from a DAG
        %
        function self = MDP_dag(next_keys, next_values, rewards, lambda)
            if ~exist('lambda', 'var')
                lambda = 0;
            end
            self = self@MDP(lambda);

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
            self.A = A;
            self.P = P;
            self.R = R;
            self.Q = Q;
            self.V = V;
            self.H = H;
            self.E_V = E_V;
            self.E_Q = E_Q;
        end

        %
        % Generic function that samples paths using a nice GUI
        %

        function sample_gui_helper(self, init_fn, step_fn, s)
            if ischar(s)
                s = find(strcmp(self.S_names, s));
            end
            self.gui_state = init_fn(s);

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

            self.gui_state = step_fn(self.gui_state);
            self.plot_gui();
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
        function plot_helper(self, what, tit, xlab, ylab)
            imagesc(what);
            xlabel(xlab);
            ylabel(ylab);
            title(tit);
            set(gca, 'YTick', 1:numel(what), 'YTickLabel', self.S_names);
            text(1, self.gui_state.s, 'X', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'red');
        end

        function plot_gui(self)
            figure(self.gui_map);

            score = sprintf('Total reward: %.2f, steps: %d', self.gui_state.Rtot, numel(self.gui_state.path));
            if self.gui_state.done
                score = ['FINISHED!: ', score];
            end

            subplot(2, 8, 1);
            self.plot_helper(self.R, 'R', score, self.gui_state.method);

            subplot(2, 8, 2);
            v = zeros(numel(self.S), 1);
            for s = self.gui_state.path
                v(s) = v(s) + 1;
            end
            self.plot_helper(v, 'visited', '', '');

            subplot(2, 8, 3);
            self.plot_helper(self.V, 'V(s)', '', '');

            subplot(2, 8, 4);
            self.plot_helper(self.E_V, 'E_V(s)', '', '');

            subplot(2, 8, 5);
            self.plot_helper(sum(self.E_Q, 2), 'sum E_Q(s,.)', '', '');

            % plot map and current action-value f'n max Q(s, a)
            %
            subplot(2, 8, 6);
            self.plot_helper(max(self.Q, [], 2), 'max Q(s,.)', '', '');

            % plot map and transition probability across all possible actions, P(.|s)
            %
            subplot(2, 8, 7);
            pi = self.gui_state.pi';
            p = squeeze(self.P(:, self.gui_state.s, :));
            p = p * pi;
            self.plot_helper(p, 'policy \pi = P(s''|s)', '', '');

            % plot map and current transition probability given the selected action, P(.|s,a)
            %
            subplot(2, 8, 8);
            p = self.P(:, self.gui_state.s, self.gui_state.a);
            self.plot_helper(p, 'P(s''|s,a)', '', '');

            % plot reward / PE history
            %
            rs = [0 self.gui_state.rs];
            pes = [0 self.gui_state.pes];
            path = self.gui_state.path;
            subplot(2, 1, 2);
            plot(rs);
            hold on;
            plot(pes);
            hold off;
            set(gca, 'XTick', 1:numel(rs));
            xticklabels(self.S_names(path));
            legend('rewards', 'PEs');

        end

    end
end
