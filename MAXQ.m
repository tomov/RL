% MAXQ algorithm from Dietterich (2000)
% Works with 'rooms' domain only. Similar to the maze in Figure 6 of his paper
%
classdef MAXQ < handle

    properties (Constant = true)
        R_I = -1; % penalty for remaining stationary. TODO dedupe with MDP.R_I 

        % Maze
        %
        subtask_symbols = 'ABCDEFGHIJKLMNOP';
    end

    properties (Access = public)
        Max_nodes = {}; % Max nodes
        Q_nodes = {}; % Q nodes

        root = []; % root node of the MAXQ graph

        mdp = []; % low-level MDP

        % Maze
        %
        map = [];

        % GUI
        %
        gui_start_s = [];
    end

    methods

        % Initialize an MAXQ tree from a maze
        %
        function self = MAXQ(map)
            self.map = map;

            % Create a base MDP for convenience
            %
            all_subtask_inds = find(ismember(map, MAXQ.subtask_symbols));
            map(all_subtask_inds) = MDP.empty_symbol; % remove all subtasks
            mdp = MDP(map);
            self.mdp = mdp;

            % Lowest layer of the hierarchy: max nodes == primitive actions
            %
            for a = mdp.A
                max_node = struct;
                max_node.is_primitive = true;
                max_node.a = a;
                max_node.p = a;
                max_node.layer = 0;
                max_node.name = ['MaxAction ', num2str(a)];
                max_node.V = zeros(numel(mdp.S), 1); % Vi(s) from paper

                self.Max_nodes = [self.Max_nodes, max_node];
            end

            % Second layer of hierarchy: max nodes = subtasks
            %
            map = self.map;
            for c = unique(map(all_subtask_inds))' % for each subtask
                subtask_inds = find(map == c);
               
                max_node = struct;
                max_node.is_primitive = false;
                max_node.layer = 1;
                max_node.p = numel(self.Max_nodes) + 1;
                max_node.name = ['MaxSubtask ', c];
                max_node.I = subtask_inds'; % internal "active" states (S in the paper) = all states labeled for the subtask
                max_node.B = setdiff(mdp.S, max_node.I); % boundary "terminated" states (T in the paper)
                max_node.V = zeros(numel(mdp.S), 1); % Vi(s) from paper
      
                % create children q-nodes == the primitive actions.
                % they are parents of the primitve action max nodes
                max_node.children = [];
                for i = mdp.A
                    q_node = struct;
                    q_node.a = i; % child = primitive action i
                    q_node.layer = 0.5;
                    q_node.p = max_node.p;
                    q_node.idx = numel(self.Q_nodes) + 1;
                    q_node.name = ['QAction ', num2str(i)];
                    q_node.C = zeros(numel(mdp.S), 1); % Ci(s,a) from paper. Notice that i and a are specified by the q_node
                    q_node.Q = zeros(numel(mdp.S), 1); % Qi(s,a) from paper

                    self.Q_nodes = [self.Q_nodes, q_node];
                    max_node.children = [max_node.children, numel(self.Q_nodes)];
                end

                self.Max_nodes = [self.Max_nodes, max_node];
            end

            % Last layer = root node
            %
            max_node = struct;
            max_node.is_primitive = false;
            max_node.layer = 2;
            max_node.p = numel(self.Max_nodes) + 1;
            max_node.name = ['MaxRoot'];
            max_node.I = mdp.I;
            max_node.B = mdp.B;
            max_node.V = zeros(numel(mdp.S), 1); % Vi(s) from paper

            % children of root = q-nodes -> the subtasks
            max_node.children = [];
            for i = 1:numel(self.Max_nodes)
                m = self.Max_nodes{i};
                if m.layer == 1
                    assert(~m.is_primitive);
                    q_node = struct;
                    q_node.a = i;
                    q_node.layer = 1.5;
                    q_node.p = max_node.p;
                    q_node.idx = numel(self.Q_nodes) + 1;
                    q_node.name = ['QSubtask ', m.name(end)];
                    q_node.C = zeros(numel(mdp.S), 1); % Ci(s,a) from paper. Notice that i and a are specified by the q_node
                    q_node.Q = zeros(numel(mdp.S), 1); % Qi(s,a) from paper

                    self.Q_nodes = [self.Q_nodes, q_node];
                    max_node.children = [max_node.children, numel(self.Q_nodes)];
                end
            end

            self.Max_nodes = [self.Max_nodes, max_node];
            self.root = numel(self.Max_nodes);
        end 

        %
        % Run an episode and update V-values and Q-values using MAXQ-0-learning
        %

        function sample0_gui(varargin)
            self = varargin{1};
            self.gui_start_s = varargin{2};
            self.sample_gui_helper(@self.init_sample0, @self.step0, varargin{2:end}); % we're using the local one
        end

        function res = sample0(varargin)
            self = varargin{1};
            res = self.mdp.sample_helper(@self.init_sample0, @self.step0, varargin{2:end});
        end

        function stack = init_sample0(self, s)
            assert(numel(find(self.mdp.I == s)) == 1);
            state = self.init_state0(s, self.root);
            stack = [state]; % we're unrolling the recursion baby
        end

        function state = init_state0(self, s, p)
            state.Rtot = 0;
            state.path = [];
            state.s = s;
            state.p = p; % subtask / subroutine = max node = 'action'
            state.done = false; % have we reached a terminal state?
            state.method = 'MAXQ-0';

            state.second_half = false; % whether we're coming back from the recursion
            state.r = 0; % as returned from the lower layer
            state.new_s = 0; % as returned from the lower layer
            state.a = -1; % chosen action = max node
            state.pa = -1; % q node = parent of chosen action
        end
       
        % Iterative MaxQ-0
        % Check out the recursive version for clarity
        %
        function stack = step0(self, stack)
            spaces = repmat(' ', 1, numel(stack) * 5);

            state = stack(end);
            s = state.s;
            p = state.p;

            %{
            if ~self.Max_nodes{p}.is_primitive && (state.done || ismember(s, self.Max_nodes{p}.B))
                % Someone called us with a boundary state -> return immediately
                %
                state
                assert(~state.second_half);
                if numel(stack) > 1
                    % pass results back up the stack, if we're not root
                    %
                    stack(end-1).r = stack(end).Rtot;
                    stack(end-1).new_s = stack(end).new_s;
                    stack(end-1).second_half = true; % tell caller to move on
                    stack = stack(1:end-1);
                else
                    % we're finished -> nothing to pop
                    %
                    stack(1).done = true;
                end
            end
            %}

            max_node = self.Max_nodes{p};
            q_children = max_node.children;
            q_nodes = [self.Q_nodes{q_children}];
            max_children = [q_nodes.a]; % max children of the current max node (subroutine) p

            if ~state.second_half
                % Do the regular loop from maxQ0
                % up to calling maxQ0 recursively, or executing a primitive action
                %

                % Choose action / subroutine a
                %
                pi = self.glie(s, p);
                j = samplePF(pi);
                pa = q_children(j);
                a = max_children(j);
                stack(end).a = a;
                stack(end).pa = pa;

                fprintf('\n\n\n%s At maxQ0(%d, %d), executing a = %d (pa = %d, pi = %s, q children = %s, max children = %s)\n', spaces, s, p, a, pa, sprintf('%.2f ', pi), sprintf('%d ', q_children), sprintf('%d ', max_children));
                disp(max_node);

                % Execute a
                %
                if self.Max_nodes{a}.is_primitive
                    % If a is primitive, get reward r and new state s' from low-level MDP
                    %
                    new_s = samplePF(self.mdp.P(:,s,a));
                    r = self.mdp.R(new_s);

                    % Update Vi(s) for primitive node here (b/c we don't call maxQ0 on it)
                    %
                    pe = r - self.Max_nodes{a}.V(s); % Eq 11: PE = R(s') - Vi(s)
                    self.Max_nodes{a}.V(s) = self.Max_nodes{a}.V(s) + MDP.alpha * pe; % Eq 11: Vi(s) = Vi(s) + alpha * PE
                    
                    fprintf('%s       Primitive action -> new_s = %d, r = %.2f, Rtot = %.2f, pe = %.2f, V(a,s) = %.2f\n', spaces, new_s, r, stack(end).Rtot, pe, self.Max_nodes{a}.V(s));

                    stack(end).new_s = new_s;
                    stack(end).r = r;
                    stack(end).second_half = true; % tell caller to move on
                    stack(end).done = true; % primitive actions terminate immediately
                else
                    % a is a subroutine -> find reward r and new state s' recursively from children
                    %
                    fprintf('%s Complex action -> calling maxQ0(%d, %d)\n', spaces, s, a);

                    state = self.init_state0(s, a);
                    stack = [stack, state];
                    %[r, new_s] = self.maxQ0(s, a);
                end

            else
                % Do the second half of the recursion
                %

                % get locals from stack
                %
                a = stack(end).a;
                pa = stack(end).pa;
                new_s = stack(end).new_s;

               % assert(~ismember(s, self.Max_nodes{p}.B)); % cannot update a boundary state

                fprintf('%s ...back to maxQ0(%d, %d) after calling maxQ0(%d, %d): new_s = %d, r = %.2f\n', spaces, s, p, s, a, new_s, stack(end).r);

                % Update total reward
                %
                stack(end).Rtot = stack(end).Rtot + stack(end).r;
                stack(end).path = [stack(end).path, new_s];

                % Update Ci(s, a)
                %
                pe = self.Max_nodes{p}.V(new_s) - self.Q_nodes{pa}.C(s); % Eq 10: PE = Vi(s') - Ci(s,a)
                self.Q_nodes{pa}.C(s) = self.Q_nodes{pa}.C(s) + MDP.alpha * pe; % Eq 10: Ci(s,a) = Ci(s,a) + alpha * PE
                self.Q_nodes{pa}.Q(s) = self.Q_nodes{pa}.C(s) + self.Max_nodes{a}.V(s); % Eq 7: Qi(s,a) = Ci(s,a) + Va(s)

                Q = [];
                for j = 1:numel(max_children)
                    Q = [Q, self.Q_nodes{q_children(j)}.Q(s)]; % Qi(s,a)
                end
                self.Max_nodes{p}.V(s) = max(Q); % Eq 8: Vi(s) = max Qi(s,:)

                fprintf('%s Qs = [%s], pe = %.2f, C(p,s,a) = %.2f, V(p,s) = %.2f\n', spaces, sprintf('%.2f ', Q), pe, self.Q_nodes{pa}.C(s), self.Max_nodes{p}.V(s));

                if stack(end).done || ismember(new_s, self.Max_nodes{p}.B)
                    % Boundary state
                    %
                    fprintf('%s DONE!\n', spaces);

                    if numel(stack) > 1
                        % pass results back up the stack, if we're not root
                        %
                        stack(end-1).r = stack(end).Rtot;
                        stack(end-1).new_s = stack(end).new_s;
                        stack(end-1).second_half = true; % tell caller to move on
                        stack = stack(1:end-1);
                    else
                        % we're finished -> nothing to pop
                        %
                        stack(1).done = true;
                    end
                else
                    % Internal state -> we keep going
                    %
                    stack(end).s = new_s;
                    stack(end).second_half = false;
                end
            end
        end

        % Recursive MaxQ-0
        %
        function [Rtot, new_s] = maxQ0(self, s, p)
            if ~exist('p', 'var')
                p = self.root; % start from the top
            end
            spaces = repmat(' ', 1, (2 - self.Max_nodes{p}.layer) * 5);
            
            Rtot = 0;
            new_s = s;
            done = false;

            if ismember(s, self.Max_nodes{p}.B) % edge case
                return
            end

            while ~done
                max_node = self.Max_nodes{p};
                q_children = max_node.children;
                q_nodes = [self.Q_nodes{q_children}];
                max_children = [q_nodes.a]; % max children of the current max node (subroutine) p

                % Choose action / subroutine a
                %
                pi = self.glie(s, p);
                j = samplePF(pi);
                pa = q_children(j);
                a = max_children(j);

                fprintf('\n\n\n%s At maxQ0(%d, %d), executing a = %d (pa = %d, pi = %s, q children = %s, max children = %s)\n', spaces, s, p, a, pa, sprintf('%.2f ', pi), sprintf('%d ', q_children), sprintf('%d ', max_children));
                disp(max_node);

                % Execute a
                %
                if self.Max_nodes{a}.is_primitive
                    % If a is primitive, get reward r and new state s' from low-level MDP
                    %
                    new_s = samplePF(self.mdp.P(:,s,a));
                    r = self.mdp.R(new_s);

                    % Update Vi(s) for primitive node here (b/c we don't call maxQ0 on it)
                    %
                    pe = r - self.Max_nodes{a}.V(s); % Eq 11: PE = R(s') - Vi(s)
                    self.Max_nodes{a}.V(s) = self.Max_nodes{a}.V(s) + MDP.alpha * pe; % Eq 11: Vi(s) = Vi(s) + alpha * PE
                    
                    fprintf('%s       Primitive action -> new_s = %d, r = %.2f, Rtot = %.2f, pe = %.2f, V(a,s) = %.2f\n', spaces, new_s, r, Rtot, pe, self.Max_nodes{a}.V(s));

                else
                    % a is a subroutine -> find reward r and new state s' recursively from children
                    %
                    fprintf('%s Complex action -> calling maxQ0(%d, %d)\n', spaces, s, a);

                    [r, new_s] = self.maxQ0(s, a);

                    fprintf('%s ...back to maxQ0(%d, %d) after calling maxQ0(%d, %d): new_s = %d, r = %.2f\n', spaces, s, p, s, a, new_s, r);
                end

                assert(~ismember(s, self.Max_nodes{p}.B)); % cannot update a boundary state

                % Update total reward
                %
                Rtot = Rtot + r;

                % Update Ci(s, a)
                %
                pe = self.Max_nodes{p}.V(new_s) - self.Q_nodes{pa}.C(s); % Eq 10: PE = Vi(s') - Ci(s,a)
                self.Q_nodes{pa}.C(s) = self.Q_nodes{pa}.C(s) + MDP.alpha * pe; % Eq 10: Ci(s,a) = Ci(s,a) + alpha * PE
                self.Q_nodes{pa}.Q(s) = self.Q_nodes{pa}.C(s) + self.Max_nodes{a}.V(s); % Eq 7: Qi(s,a) = Ci(s,a) + Va(s)

                Q = [];
                for j = 1:numel(max_children)
                    Q = [Q, self.Q_nodes{q_children(j)}.Q(s)]; % Qi(s,a)
                end
                self.Max_nodes{p}.V(s) = max(Q); % Eq 8: Vi(s) = max Qi(s,:)

                fprintf('%s Qs = [%s], pe = %.2f, C(p,s,a) = %.2f, V(p,s) = %.2f\n', spaces, sprintf('%.2f ', Q), pe, self.Q_nodes{pa}.C(s), self.Max_nodes{p}.V(s));

                if ismember(new_s, self.Max_nodes{p}.B)
                    % Boundary state
                    %
                    fprintf('%s DONE!\n', spaces);
                    done = true;
                end

                s = new_s;
            end
        end

        function p = glie(self, s, p)
            % Actions = child q-nodes of the max_node
            % TODO implement actual greedy in the limit with infinite exploration
            %
            max_node = self.Max_nodes{p};
            Q = [];
            for i = 1:numel(max_node.children)
                q_node = self.Q_nodes{max_node.children(i)};
                Q = [Q, q_node.Q(s)]; % get all the Q-values
            end
            p = eps_greedy(Q, MDP.eps);
            fprintf('   glie for %s: %s (p = %s)\n', max_node.name, sprintf('%.2f ', Q), sprintf('%.2f ', p));
        end


        %
        % Generic function that samples paths using a nice GUI
        %

        % TODO dedupe with MDP 
        function sample_gui_helper(self, init_fn, step_fn, s)
            self.mdp.gui_state = init_fn(s);
			self.mdp.gui_map = figure;
            self.plot_gui();

            step_callback = @(hObject, eventdata) self.step_gui_callback(step_fn, hObject, eventdata);
            start_callback = @(hObject, eventdata) self.mdp.start_gui_callback(hObject, eventdata);
            reset_callback = @(hObject, eventdata) self.reset_gui_callback(init_fn, hObject, eventdata);
            stop_callback = @(hObject, eventdata) stop(self.mdp.gui_timer);
            sample_callback = @(hObject, eventdata) self.sample_gui_callback(step_fn, hObject, eventdata);

            self.mdp.gui_timer = timer('Period', 0.5, 'TimerFcn', step_callback, 'ExecutionMode', 'fixedRate', 'TasksToExecute', 1000000);
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

        function step_gui_callback(self, step_fn, hObject, eventdata)
            if numel(self.mdp.gui_state) == 1 && self.mdp.gui_state.done
                stop(self.mdp.gui_timer);
                return
            end

            self.mdp.gui_state = step_fn(self.mdp.gui_state);
            self.plot_gui();
        end

        function reset_gui_callback(self, init_fn, hObject, eventdata)
            self.mdp.gui_state = init_fn(self.gui_start_s);
            self.plot_gui();
        end

        function sample_gui_callback(self, step_fn, hObject, eventdata)
            while numel(self.mdp.gui_state) > 1 || ~self.mdp.gui_state.done
                self.mdp.gui_state = step_fn(self.mdp.gui_state);
            end
            self.plot_gui();
        end

        function plot_gui(self)
            figure(self.mdp.gui_map);
            stack = self.mdp.gui_state;
            state = stack(end);
            [x, y] = ind2sub(size(self.map), state.s);

            disp(state);

            n = 5;
            l = 0;
            for layer = 2:-1:0
                max_nodes = {};
                for i = 1:numel(self.Max_nodes)
                    if self.Max_nodes{i}.layer == layer
                        max_nodes = [max_nodes, self.Max_nodes{i}];
                    end
                end

                m = max(8, numel(max_nodes));
                if m == 8 && numel(max_nodes) == 1
                    m = 7; % hacks to make it pretty
                end
                offs = max(0, floor((8 - numel(max_nodes)) / 2));
                l = l + 1;
                for i = 1:numel(max_nodes)
                    max_node = max_nodes{i};
                    subplot(n, m, i + (l-1)*m + offs);
                    vi = max_node.V(self.mdp.I);
                    imagesc(reshape(vi, size(self.map)));
                    if max_node.p == state.a
                        text(y, x, 'X', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'red');
                    elseif max_node.p == state.p
                        text(y, x, 'X', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'white');
                    elseif ismember(max_node.p, [stack.p])
                        text(y, x, 'X', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'black');
                    end
                    if i == 1, ylabel('V(s)'); end
                    title(sprintf('%s (%d)', max_node.name, max_node.p));
                    axis off;
                end

                if layer == 0
                    break;
                end
                layer = layer - 0.5; % the Q nodes

                q_nodes = {};
                for i = 1:numel(self.Q_nodes)
                    if abs(self.Q_nodes{i}.layer - layer) < 1e-8
                        q_nodes = [q_nodes, self.Q_nodes{i}];
                    end
                end

                m = max(8, numel(q_nodes));
                offs = max(0, floor((8 - numel(q_nodes)) / 2));
                l = l + 1;
                for i = 1:numel(q_nodes)
                    q_node = q_nodes{i};
                    subplot(n, m, i + (l-1)*m + offs);
                    qi = q_node.Q(self.mdp.I);
                    imagesc(reshape(qi, size(self.map)));
                    if q_node.a == state.a && q_node.p == state.p
                        text(y, x, 'X', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'red');
                    elseif find(q_node.a == [stack.a]) == find(q_node.p == [stack.p])
                        text(y, x, 'X', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'black');
                    end
                    if i == 1, ylabel('Q(s,a)'); end
                    title(sprintf('%s (%d)', q_node.name, q_node.idx));
                    axis off;
                end
            end

            label = sprintf('Total reward: %.2f, steps: %d', state.Rtot, numel(state.path));
            if state.done
                xlabel(['FINISHED!: ', label]);
            else
                xlabel(label);
            end
        end

    end
end
