% TD-learning for 'rooms' domain only
%
classdef MDP_maze < MDP

    properties (Constant = true)
        R_I = -1; % penalty for staying in one place

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
        map = [];

        gui_state = []; % state for the GUI step-through
        gui_map = []; % figure for the GUI
        gui_timer = []; % timer for the GUI
    end
   
    methods

        % Initialize an MDP from a maze
        %
        function self = MDP_maze(map, lambda)
            if ~exist('lambda', 'var')
                lambda = 0;
            end
            self = self@MDP(lambda);
            self.map = map;

            absorbing_inds = find(ismember(map, self.absorbing_symbols)); % goal squares = internal states with corresponding boundary states
            I_with_B = absorbing_inds;
            Ni = numel(map); % internal states = all squares, including walls (useful for (x,y) --> state)
            Nb = numel(I_with_B); % boundary states
            N = Ni + Nb;
            
            S = 1 : N; % set of states
            I = 1 : Ni; % set of internal states
            B = Ni + 1 : Ni + Nb; % set of boundary states
            A = 1:numel(MDP_maze.A_names);  % set of actions
            self.S = S;
            self.I = I;
            self.B = B;
            self.A = A;

            I2B = zeros(N, 1);
            I2B(I_with_B) = B; % mapping from I states to corresponding B states

            P = zeros(N, N, numel(A)); % transitions P(s'|s,a); defaults to 0
            Q = zeros(N, numel(A)); % action values Q(s, a)
            V = zeros(N, 1); % state values V(s)
            H = zeros(N, numel(A)); % policy parameters
            R = nan(N, 1); % instantaneous reward f'n R(s)
            E_V = zeros(N, 1); % eligibility traces E(s) for state values
            E_Q = zeros(N, numel(A)); % eligibility traces E(s, a) for action values
          
            assert(size(MDP_maze.adj, 1) + 1 == numel(A));
            % iterate over all internal states s
            %
            for x = 1:size(map, 1)
                for y = 1:size(map, 2)
                    s = self.pos2I(x, y);
                    %fprintf('(%d, %d) --> %d = ''%c''\n', x, y, s, map(x, y));
                    assert(ismember(s, S));
                    assert(ismember(s, I));
                    
                    R(s) = MDP_maze.R_I; % time is money for internal states

                    if ismember(map(x, y), self.impassable_symbols)
                        % Impassable state e.g. wall -> no neighbors
                        %
                        continue;
                    end
                    
                    % Iterate over all adjacent states s --> s' and update passive
                    % dynamics P
                    %
                    for a = 1:size(MDP_maze.adj, 1)
                        new_x = x + MDP_maze.adj(a, 1);
                        new_y = y + MDP_maze.adj(a, 2);
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
                        P(b, s, end) = 1; % last action = pick up reward
                        
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

                    % Normalize transition probabilities and mark some actions as illegal
                    %
                    for a = A
                        %if sum(P(:, s, a)) == 0
                        %    P(s, s, a) = 1; % impossible moves keep you stationary
                        %end
                        if sum(P(:, s, a)) > 0 % allowed action
                            P(:, s, a) = P(:, s, a) / sum(P(:, s, a)); % normalize P(.|s,a)
                        else % disallowed action
                            H(s, a) = -Inf;
                            Q(s, a) = -Inf;
                        end
                    end
                end
            end
            
            assert(isequal(I, setdiff(S, B)));
            
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
            if ~exist('s', 'var')
                s = find(self.map == 'X');
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

            if numel(self.gui_state) == 1 % else MAXQ... TODO cleanup
                [x, y] = self.I2pos(self.gui_state.s);
                old_s = self.gui_state.s;
                old_a = self.gui_state.a;
            end

            self.gui_state = step_fn(self.gui_state);
            self.plot_gui();

            if numel(self.gui_state) == 1 % else MAXQ... TODO cleanup
                if self.gui_state.done
                    fprintf('(%d, %d), %d --> END [%.2f%%]\n', x, y, old_a, self.P(self.gui_state.s, old_s, old_a) * 100);
                else
                    [new_x, new_y] = self.I2pos(self.gui_state.s);
                    map(x, y) = self.empty_symbol;
                    map(new_x, new_y) = self.agent_symbol;
                    fprintf('(%d, %d), %d --> (%d, %d) [%.2f%%]\n', x, y, old_a, new_x, new_y, self.P(self.gui_state.s, old_s, old_a) * 100);
                end
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
        function plot_helper(self, what, tit, xlab, ylab)
            imagesc(reshape(what, size(self.map)));
            xlabel(xlab);
            ylabel(ylab);
            title(tit);
            s = self.gui_state.s;
            if ismember(s, self.B)
                s = self.gui_state.path(end-1);
            end
            [x, y] = ind2sub(size(self.map), s);
            text(y, x, 'X', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'red');
        end

        function plot_gui(self)
            figure(self.gui_map);

            score = sprintf('Total reward: %.2f, steps: %d', self.gui_state.Rtot, numel(self.gui_state.path));
            if self.gui_state.done
                score = ['FINISHED!: ', score];
            end

            % plot map and rewards
            %
            subplot(2, 8, 1);
            self.plot_helper(self.map == '#', 'map', score, self.gui_state.method);
            % goals
            goals = find(self.map == '$');
            for g = goals
                [x, y] = ind2sub(size(self.map), g);
                text(y, x, '$', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'green');
            end

            % heat map of visited states
            %
            subplot(2, 8, 2);
            v = zeros(size(self.map));
            for s = self.gui_state.path
                if s <= numel(self.map) % i.e. s in I
                    v(s) = v(s) + 1;
                end
            end
            self.plot_helper(v, 'visited', '', '');

            % plot map and current state-value f'n V(s)
            %
            subplot(2, 8, 3);
            self.plot_helper(self.V(self.I), 'V(s)', '', '');

            % current state-value eligibility trace f'n E(s)
            %
            subplot(2, 8, 4);
            self.plot_helper(self.E_V(self.I), 'E_V(s)', '', '');

            % current action-value eligibility trace f'n E(s)
            %
            subplot(2, 8, 5);
            self.plot_helper(sum(self.E_Q(self.I, :), 2), 'sum E_Q(s,.)', '', '');

            % plot map and current action-value f'n max Q(s, a)
            %
            subplot(2, 8, 6);
            self.plot_helper(max(self.Q(self.I, :), [], 2), 'max Q(s,.)', '', '');

            % plot map and transition probability across all possible actions, P(.|s)
            %
            subplot(2, 8, 7);
            pi = self.gui_state.pi';
            p = squeeze(self.P(self.I, self.gui_state.s, :));
            p = p * pi;
            self.plot_helper(p, 'policy \pi = P(s''|s)', '', '');

            % plot map and current transition probability given the selected action, P(.|s,a)
            %
            subplot(2, 8, 8);
            p = self.P(self.I, self.gui_state.s, self.gui_state.a);
            self.plot_helper(p, 'P(s''|s,a)', '', '');

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
