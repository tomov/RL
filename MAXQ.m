% MAXQ algorithm from Dietterich (2000)
% Works with 'rooms' domain only. Similar to the maze in Figure 6 of his paper
%
classdef MAXQ < handle

    properties (Constant = true)
        R_I = -1; % penalty for remaining stationary. TODO dedupe with MDP.R_I 

        % Maze
        %
        goal_symbol = '$';
        subtask_symbols = 'ABCDEFGHIJKLMNOP$';
    end

    properties (Access = public)
        max_nodes = {}; % Max nodes
        q_nodes = {}; % Q nodes

        root = []; % root node of the MAXQ graph

        mdp = []; % low-level MDP

        % Maze
        %
        map = [];

        % GUI
        %
        gui_start_s = [];
        gui_fig2 = []; % figure 2
    end

    methods

        % Initialize an MAXQ tree from a maze
        %
        function self = MAXQ(map)
            self.map = map;

            % Create a base MDP for convenience
            %
            goal_inds = find(map == MAXQ.goal_symbol);
            all_subtask_inds = find(ismember(map, MAXQ.subtask_symbols));
            map(all_subtask_inds) = MDP.empty_symbol; % remove all subtasks
            map(goal_inds) = MAXQ.goal_symbol; % the goal is also interpreted as a subtask; TODO fixme
            mdp = MDP(map);
            self.mdp = mdp;
            assert(numel(mdp.B) > 0); % make sure there's B states

            % Lowest layer of the hierarchy: max nodes == primitive actions
            %
            for a = mdp.A
                max_node = self.init_max_node(a, true, 0, ['MaxAction ', num2str(a), ' (', MDP.A_names{a}, ')'], self.mdp.I);

                self.max_nodes = [self.max_nodes, max_node];
            end

            % Second layer of hierarchy: max nodes = subtasks
            %
            map = self.map;
            for c = unique(map(all_subtask_inds))' % for each subtask
                subtask_inds = find(map == c);

                max_node = self.init_max_node(numel(self.max_nodes) + 1, false, 1, ['MaxSubtask ', c], subtask_inds')
      
                % create children q-nodes == the primitive actions.
                % they are parents of the primitve action max nodes
                max_node.children = [];
                for i = mdp.A
                    q_node = self.init_q_node(max_node.i, i, 0.5, ['QAction ', num2str(i)]);
                    self.q_nodes = [self.q_nodes, q_node];
                    max_node.children = [max_node.children, numel(self.q_nodes)];
                end

                self.max_nodes = [self.max_nodes, max_node];
            end

            % Last layer = root node
            %
            max_node = self.init_max_node(numel(self.max_nodes) + 1, false, 2, 'MaxRoot', mdp.I)

            % children of root = q-nodes -> the subtasks
            max_node.children = [];
            for i = 1:numel(self.max_nodes)
                m = self.max_nodes{i};
                if m.layer == 1
                    assert(~m.is_primitive);

                    q_node = self.init_q_node(max_node.i, i, 1.5, ['QSubtask ', m.name(end)]);

                    self.q_nodes = [self.q_nodes, q_node];
                    max_node.children = [max_node.children, numel(self.q_nodes)];
                end
            end

            self.max_nodes = [self.max_nodes, max_node];
            self.root = numel(self.max_nodes);
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

        function state = init_state0(self, s, i)
            state.Rtot = 0; % returned to higher layer as r
            state.path = [];
            state.s = s;
            state.i = i; % subtask / subroutine = max node = 'action'
            state.done = false; % have we reached a terminal state?
            state.method = 'MAXQ-0';

            state.second_half = false; % whether we're coming back from the recursion
            state.r = NaN; % as returned from the lower layer
            state.new_s = NaN; % as returned from the lower layer; returned to higher layer
            state.n = NaN; % as returned from lower layer; returned to higher layer as count
            state.a = NaN; % local: chosen action = max node
            state.ia = NaN; % local: q node = parent of chosen action
            state.count = 0; % local: number of steps

            state.rs = [];
            state.pes = [];
        end

        function max_node = init_max_node(self, i, is_primitive, layer, name, I)
            max_node = struct;
            max_node.is_primitive = is_primitive;
            max_node.layer = layer;
            max_node.i = i;
            max_node.name = name;
            max_node.I = I; % internal "active" states (S in the paper) = all states labeled for the subtask
            max_node.B = setdiff(self.mdp.S, max_node.I); % boundary "terminated" states (T in the paper)
            max_node.V = zeros(numel(self.mdp.S), 1); % Vi(s) from paper = expected discounted reward (in subtask i only!) for starting in state s and following policy for i
            max_node.pseudoR = zeros(numel(self.mdp.S), 1); % Ri~(s) = pseudorewards
        end

        function q_node = init_q_node(self, i, a, layer, name)
            q_node = struct;
            q_node.a = a; % child = primitive action i
            q_node.layer = layer;
            q_node.i = i;
            q_node.idx = numel(self.q_nodes) + 1;
            q_node.name = name;
            % note: (in subtask i only!) == in the state space defined by the internal states allowed in subtask i, that is all s in max_node.I
            q_node.C = zeros(numel(self.mdp.S), 1); % Ci(s,a) from paper = expected discounted reward (in subtask i only!) *after* completing action/subtask a in state s and then following the policy for i (i.e. excluding the intermediary rewards from the states/actions of executing a). Notice that i and a are specified by the q_node
            q_node.Q = zeros(numel(self.mdp.S), 1); % Qi(s,a) from paper = expected discounted reward (in subtask i only!) for completing action/subtask a in state s and then following the policy for i (i.e. including the intermediary rewards from the states/actions of executing a).
            q_node.pseudoC = zeros(numel(self.mdp.S), 1); % Ci~(s,a) = same as Ci but including pseudo-rewards for subtask i
            q_node.pseudoQ = zeros(numel(self.mdp.S), 1); % Qi~(s,a) = Ci~(s, a) + Va(s)
        end
       
        % Iterative MaxQ-0
        % no psueudo-rewards; task = internal states & boundary states
        % Check out the recursive version for clarity, although it's a bit broken
        %
        function stack = step0(self, stack)
            spaces = repmat(' ', 1, numel(stack) * 5);

            state = stack(end);
            s = state.s;
            i = state.i;
            count = state.count;

            pseudorewards = true; % true = MAXQ-Q (pseudorewards), false = MAXQ-0 TODO parametrize (this but also parametrize the subtasks)

            max_node = self.max_nodes{i};
            q_children = max_node.children;
            q_nodes = [self.q_nodes{q_children}];
            max_children = [q_nodes.a]; % max children of the current max node (subroutine) p

            if ~state.second_half
                % Do the regular loop from maxQ0
                % up to calling maxQ0 recursively, or executing a primitive action
                %

                fprintf('\n\n\n');

                % Choose action / subroutine a
                %
                pi = self.glie(s, i, false);
                j = samplePF(pi);
                ia = q_children(j);
                a = max_children(j);
                stack(end).a = a;
                stack(end).ia = ia;
                stack(end).second_half = true; % caller must move on next time

                fprintf('%s At maxQ0(i = %d, s = %d), executing a = %d (ia = %d, pi = %s, q children = %s, max children = %s)\n', spaces, i, s, a, ia, sprintf('%.2f ', pi), sprintf('%d ', q_children), sprintf('%d ', max_children));
                disp(max_node);

                % Execute a
                %
                if self.max_nodes{a}.is_primitive
                    % If a is primitive, get reward r and new state s' from low-level MDP
                    %
                    new_s = samplePF(self.mdp.P(:,s,a));
                    r = self.mdp.R(new_s);

                    % Update Vi(s) for primitive node here (b/c we don't call maxQ0 on it)
                    %
                    pe = r - self.max_nodes{a}.V(s); % Eq 11: PE = R(s') - Vi(s)
                    self.max_nodes{a}.V(s) = self.max_nodes{a}.V(s) + MDP.alpha * pe; % Eq 11: Vi(s) = Vi(s) + alpha * PE
                    
                    fprintf('%s       Primitive action -> new_s = %d, r = %.2f, pe = %.2f, V(a,s) = %.2f\n', spaces, new_s, r, pe, self.max_nodes{a}.V(s));

                    stack(end).new_s = new_s; % set up locals for second half of recursion
                    stack(end).r = r;
                    stack(end).n = 1;
                    stack(end).rs = [stack(end).rs, r]; % logging 
                    stack(end).pes = [stack(end).pes, pe];

                elseif ismember(s, self.max_nodes{a}.B)
                    % Illegal move to a subtask for which s is a boundary state
                    % => penalize so we don't do it again
                    % TODO is this okay? or even necessary?
                    %
                    fprintf('%s       ----------- Illegal action a = %d for s = %d\n', spaces, a, s);

                    stack(end).second_half = false; % try again...

                else
                    % a is a subroutine -> find reward r and new state s' recursively from children
                    %
                    fprintf('%s Complex action -> calling maxQ0(a = %d, s = %d)\n', spaces, a, i);

                    state = self.init_state0(s, a);
                    stack = [stack, state]; % recurse -> put next call to maxQ0(s,a) on top of stack
                end

            else
                % Do the second half of the recursion
                %

                % get locals / returned vars from stack
                %
                a = stack(end).a;
                ia = stack(end).ia;
                new_s = stack(end).new_s;
                n = stack(end).n;

                assert(~ismember(s, self.max_nodes{i}.B)); % cannot update a boundary state

                fprintf('\n%s ...back to maxQ0(i = %d, s = %d) after calling maxQ0(a = %d, s = %d): new_s = %d, r = %.2f, n = %d\n', spaces, i, s, a, s, new_s, stack(end).r, n);

                % Update locals 
                %
                stack(end).Rtot = stack(end).Rtot + stack(end).r;
                stack(end).path = [stack(end).path, new_s];
                stack(end).count = stack(end).count + n;

                % Update Ci(s, a)
                %
                if ~pseudorewards
                    % Regular ol' MAXQ-0
                    %
                    % Decompose value function:
                    % Vi(s) = (in subtask i) expected discounted reward starting at state s (and following policy for i)
                    % ...imagine policy for i says take action/subtask a, i.e. pi_i(s) = a
                    % => Vi(s) = (expected discounted reward from executiong action/subtask a) + (expected discounted reward from continuing with subtask i after that)
                    % ...imagine action/subtask a takes you to state s'
                    % => Vi(s) = Va(s) + Vi(s') * (discount rate)^(# steps in action/subtask a)
                    %          = Va(s) + Ci(s, a)
                    %          = Qi(s, a)
                    % ...for optimal policy,
                    % => Vi(s) = max_a Qi(s,a)
                    % => update Ci(s, a) using TD learning to converge to Vi(s') i.e. as if Vi(s') is its reward (it's the actual reward for primitive actions)
                    %
                    % notice that the discounting for rewards after s' is taken into account in C
                    %
                    r = (MDP.gamma ^ n) * self.max_nodes{i}.V(new_s);
                    pe = r - self.q_nodes{ia}.C(s); % Eq 10: PE = Vi(s') - Ci(s,a)
                    self.q_nodes{ia}.C(s) = self.q_nodes{ia}.C(s) + MDP.alpha * pe; % Eq 10: Ci(s,a) = Ci(s,a) + alpha * PE
                    self.q_nodes{ia}.Q(s) = self.q_nodes{ia}.C(s) + self.max_nodes{a}.V(s); % Eq 7: Qi(s,a) = Ci(s,a) + Va(s)
                else
                    % MAXQ-Q: use pseudorewards and update in SARSA-like fashion
                    % (s, a) --> (s', a*) i.e. (new_s, new_a)
                    % TODO implement all-states updating
                    %

                    % first pick a* = best action following s' in subtask i (line 13 of Table 4: MAXQ-Q)
                    %
                    [~, pseudoQ] = self.glie(new_s, i, true);
                    [~, j] = max(pseudoQ);
                    new_ia = q_children(j);
                    new_a = max_children(j);
                    fprintf('                   MAXQ-Q: pseudoQ = %s, j = %d, new_ia = %d, new_a = %d, new_s = %d\n', sprintf('%.2f ', pseudoQ), j, new_ia, new_a, new_s);

                    % Update (pseudoreward) C~ and Q~
                    % pseudo reward = pseudo reward at s' (in subtask i) + discounted reward for subtask a* at s' + discounted reward after subtask a* at s' (in subtask i)
                    %
                    %pseudo_r = (MDP.gamma ^ n) * (self.max_nodes{i}.pseudoR(new_s) + self.q_nodes{new_ia}.pseudoC(new_s) + self.max_nodes{new_a}.V(new_s)); % reward = Ri~(s') + Va*(s') + Ci~(s', a*) ?= Ri~(s') + Qi~(s', a*)
                    pseudo_r = (MDP.gamma ^ n) * (self.max_nodes{i}.pseudoR(new_s) + self.q_nodes{new_ia}.pseudoQ(new_s)); % reward = Ri~(s') + Va*(s') + Ci~(s', a*) ?= Ri~(s') + Qi~(s', a*)
                    pseudo_pe = pseudo_r - self.q_nodes{ia}.pseudoC(s);
                    self.q_nodes{ia}.pseudoC(s) = self.q_nodes{ia}.pseudoC(s) + MDP.alpha * pseudo_pe;  % Ci~(s, a) = Ci~(s, a) + alpha * PE
                    self.q_nodes{ia}.pseudoQ(s) = self.q_nodes{ia}.pseudoC(s) + self.max_nodes{a}.V(s); % Qi~(s, a) = Ci~(s, a) + Va(s)
                    fprintf('                   MAXQ-Q: pseudo_r = %.2f (%.2f + %.2f (= %.2f + %.2f)), pseudo_pe = %.2f, pseudo_C = %.2f, pseudo_Q = %.2f\n', pseudo_r, self.max_nodes{i}.pseudoR(new_s), self.q_nodes{new_ia}.pseudoQ(new_s), self.q_nodes{new_ia}.pseudoC(new_s), self.max_nodes{new_a}.V(new_s), pseudo_pe, self.q_nodes{ia}.pseudoC(s), self.q_nodes{ia}.pseudoQ(s));

                    % Update actual C and Q
                    % => same but no pseudorewards, and using C instead of C~
                    % notice it's almost the same as the MAXQ-0 case, however we're using the Q-value instead of the V-value
                    %
                    r = (MDP.gamma ^ n) * self.q_nodes{new_ia}.Q(new_s); % reward = Ci(s', a*) + Va*(s') ?= Qi(s',a*) ??
                    pe = r - self.q_nodes{ia}.C(s); % PE = Vi(s') - Ci(s,a)
                    self.q_nodes{ia}.C(s) = self.q_nodes{ia}.C(s) + MDP.alpha * pe; % Ci(s,a) = Ci(s,a) + alpha * PE
                    self.q_nodes{ia}.Q(s) = self.q_nodes{ia}.C(s) + self.max_nodes{a}.V(s); % Qi(s,a) = Ci(s,a) + Va(s)
                    fprintf('                   MAXQ-Q: r = %.2f, pe = %.2f, C = %.2f, Q = %.2f\n', r, pe, self.q_nodes{ia}.C(s), self.q_nodes{ia}.Q(s));
                end

                % Update V-value
                %
                [~, Q] = self.glie(s, i, false);
                self.max_nodes{i}.V(s) = max(Q); % Eq 8: Vi(s) = max Qi(s,:)

                fprintf('%s Qs = [%s], pe = %.2f, C = %.2f, V = %.2f, Rtot = %.2f, count = %d\n', spaces, sprintf('%.2f ', Q), pe, self.q_nodes{ia}.C(s), self.max_nodes{i}.V(s), stack(end).Rtot, stack(end).count);

                % Logging
                %
                stack(end).rs = [stack(end).rs, r];
                stack(end).pes = [stack(end).pes, pe];

                % Check for boundary conditions
                %
                if stack(end).done || ismember(new_s, self.max_nodes{i}.B)
                    % Boundary state
                    %
                    fprintf('%s DONE!\n', spaces);

                    if numel(stack) > 1
                        % pass results back up the stack, if we're not root
                        %
                        stack(end-1).r = stack(end).Rtot;
                        stack(end-1).new_s = stack(end).new_s;
                        stack(end-1).n = stack(end).count;
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
                    stack(end).second_half = false; % repeat the for-loop
                end
            end
        end


%{

        % Recursive MaxQ-0
        % TODO it's broken
        %
        function [Rtot, new_s] = maxQ0(self, s, i)
            if ~exist('p', 'var')
                i = self.root; % start from the top
            end
            spaces = repmat(' ', 1, (2 - self.max_nodes{i}.layer) * 5);
            
            Rtot = 0;
            done = false;

            while ~done
                max_node = self.max_nodes{i};
                q_children = max_node.children;
                q_nodes = [self.q_nodes{q_children}];
                max_children = [q_nodes.a]; % max children of the current max node (subroutine) p

                % Choose action / subroutine a
                %
                pi = self.glie(s, i);
                j = samplePF(pi);
                ia = q_children(j);
                a = max_children(j);

                fprintf('\n\n\n%s At maxQ0(%d, %d), executing a = %d (ia = %d, pi = %s, q children = %s, max children = %s)\n', spaces, s, i, a, ia, sprintf('%.2f ', pi), sprintf('%d ', q_children), sprintf('%d ', max_children));
                disp(max_node);

                % Execute a
                %
                if self.max_nodes{a}.is_primitive
                    % If a is primitive, get reward r and new state s' from low-level MDP
                    %
                    new_s = samplePF(self.mdp.P(:,s,a));
                    r = self.mdp.R(new_s);

                    % Update Vi(s) for primitive node here (b/c we don't call maxQ0 on it)
                    %
                    pe = r - self.max_nodes{a}.V(s); % Eq 11: PE = R(s') - Vi(s)
                    self.max_nodes{a}.V(s) = self.max_nodes{a}.V(s) + MDP.alpha * pe; % Eq 11: Vi(s) = Vi(s) + alpha * PE
                    
                    fprintf('%s       Primitive action -> new_s = %d, r = %.2f, pe = %.2f, V(a,s) = %.2f\n', spaces, new_s, r, pe, self.max_nodes{a}.V(s));
                elseif ismember(s, self.max_nodes{a}.B)
                    % Illegal move to a subtask for which s is a boundary state
                    % => penalize so we don't do it again
                    % TODO is this okay? or even necessary?
                    %
                    new_s = s;
                    r = MAXQ.R_illegal;

                    %pe = r - self.max_nodes{a}.V(s); % Eq 11: PE = R(s') - Vi(s)
                    %self.max_nodes{a}.V(s) = self.max_nodes{a}.V(s) + MDP.alpha * pe; % Eq 11: Vi(s) = Vi(s) + alpha * PE

                    fprintf('%s       Illegal action -> new_s = %d, r = %.2f, pe = %.2f, V(a,s) = %.2f\n', spaces, new_s, r, pe, self.max_nodes{a}.V(s));

                else
                    % a is a subroutine -> find reward r and new state s' recursively from children
                    %
                    fprintf('%s Complex action -> calling maxQ0(%d, %d)\n', spaces, s, a);

                    [r, new_s] = self.maxQ0(s, a);

                    fprintf('%s ...back to maxQ0(%d, %d) after calling maxQ0(%d, %d): new_s = %d, r = %.2f\n', spaces, s, i, s, a, new_s, r);
                end

                assert(~ismember(s, self.max_nodes{i}.B)); % cannot update a boundary state

                % Update total reward
                %
                Rtot = Rtot + r;

                % Update Ci(s, a)
                %
                pe = self.max_nodes{i}.V(new_s) - self.q_nodes{ia}.C(s); % Eq 10: PE = Vi(s') - Ci(s,a)
                self.q_nodes{ia}.C(s) = self.q_nodes{ia}.C(s) + MDP.alpha * pe; % Eq 10: Ci(s,a) = Ci(s,a) + alpha * PE
                self.q_nodes{ia}.Q(s) = self.q_nodes{ia}.C(s) + self.max_nodes{a}.V(s); % Eq 7: Qi(s,a) = Ci(s,a) + Va(s)

                Q = [];
                for j = 1:numel(max_children)
                    Q = [Q, self.q_nodes{q_children(j)}.Q(s)]; % Qi(s,a)
                end
                self.max_nodes{i}.V(s) = max(Q); % Eq 8: Vi(s) = max Qi(s,:)

                fprintf('%s Qs = [%s], pe = %.2f, C(p,s,a) = %.2f, V(p,s) = %.2f, Rtot = %.2f\n', spaces, sprintf('%.2f ', Q), pe, self.q_nodes{ia}.C(s), self.max_nodes{i}.V(s), Rtot);

                if ismember(new_s, self.max_nodes{i}.B)
                    % Boundary state
                    %
                    fprintf('%s DONE!\n', spaces);
                    done = true;
                end

                s = new_s;
            end
        end
%}
        
        function [p, Q] = glie(self, s, i, pseudorewards)
            % Actions = child q-nodes of the max_node
            % TODO implement actual greedy in the limit with infinite exploration
            %
            max_node = self.max_nodes{i};
            q_children = max_node.children;
            q_nodes = [self.q_nodes{q_children}];
            max_children = [q_nodes.a]; % max children of the current max node (subroutine) p

            Q = [];
            for j = 1:numel(q_children)
                ia = q_children(j);
                a = max_children(j);

                if pseudorewards
                    q_value = self.q_nodes{ia}.pseudoQ(s);
                else
                    q_value = self.q_nodes{ia}.Q(s);
                end


                if self.max_nodes{a}.is_primitive
                    % going to a primitive action
                    % => make sure we can execute it
                    %
                    if self.mdp.P(s, s, a) == 1
                        Q = [Q, -Inf];
                    else
                        Q = [Q, q_value];
                    end
                else
                    % going to a subroutine
                    %
                    if ismember(s, self.max_nodes{a}.B)
                        % don't go to subtasks for which s is a boundary state, unless they're primitive actions
                        %
                        Q = [Q, -Inf];
                    else
                        Q = [Q, q_value];
                    end
                end
            end

            p = eps_greedy(Q, MDP.eps);
            fprintf('   glie (%d) for %s: %s (p = %s)\n', pseudorewards, max_node.name, sprintf('%.2f ', Q), sprintf('%.2f ', p));
        end


        %
        % Generic function that samples paths using a nice GUI
        %

        % TODO dedupe with MDP 
        function sample_gui_helper(self, init_fn, step_fn, s)
            self.mdp.gui_state = init_fn(s);
			self.mdp.gui_map = figure;
			self.gui_fig2 = figure;
            self.plot_gui();

            step_callback = @(hObject, eventdata) self.step_gui_callback(step_fn, hObject, eventdata);
            start_callback = @(hObject, eventdata) self.mdp.start_gui_callback(hObject, eventdata);
            reset_callback = @(hObject, eventdata) self.reset_gui_callback(init_fn, hObject, eventdata);
            stop_callback = @(hObject, eventdata) stop(self.mdp.gui_timer);
            sample_callback = @(hObject, eventdata) self.sample_gui_callback(step_fn, hObject, eventdata);

            self.mdp.gui_timer = timer('Period', 1, 'TimerFcn', step_callback, 'ExecutionMode', 'fixedRate', 'TasksToExecute', 1000000);
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

            %disp(state);
            x_font = 20;
            v_font = 10;

            % plot MAXQ graph + V- and Q-values
            %
            n = 12;
            l = 0;
            for layer = 2:-1:0
                % V values of max nodes
                %
                max_nodes = {};
                for i = 1:numel(self.max_nodes)
                    if self.max_nodes{i}.layer == layer
                        max_nodes = [max_nodes, self.max_nodes{i}];
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
                    colormap('Gray');
                    if max_node.i == state.a
                        text(y, x, 'X', 'FontSize', x_font, 'FontWeight', 'bold', 'Color', 'red');
                    elseif max_node.i == state.i
                        text(y, x, 'X', 'FontSize', x_font, 'FontWeight', 'bold', 'Color', 'white');
                    elseif ismember(max_node.i, [stack.i])
                        text(y, x, 'X', 'FontSize', x_font, 'FontWeight', 'bold', 'Color', 'black');
                    end

                    for ss = self.mdp.I
                        [xx, yy] = ind2sub(size(self.map), ss);
                        text(yy, xx, num2str(max_node.V(ss)), 'FontSize', v_font, 'Color', 'green');
                    end

                    if i == 1, ylabel('V(s)'); end
                    title(sprintf('%s (%d)', max_node.name, max_node.i));
                    axis off;
                end

                % Pseudorewards
                %
                l = l + 1;
                for i = 1:numel(max_nodes)
                    max_node = max_nodes{i};
                    subplot(n, m, i + (l-1)*m + offs);
                    ri = max_node.pseudoR(self.mdp.I);
                    imagesc(reshape(vi, size(self.map)));
                    colormap('Gray');
                    if max_node.i == state.a
                        text(y, x, 'X', 'FontSize', x_font, 'FontWeight', 'bold', 'Color', 'red');
                    elseif max_node.i == state.i
                        text(y, x, 'X', 'FontSize', x_font, 'FontWeight', 'bold', 'Color', 'white');
                    elseif ismember(max_node.i, [stack.i])
                        text(y, x, 'X', 'FontSize', x_font, 'FontWeight', 'bold', 'Color', 'black');
                    end

                    for ss = self.mdp.I
                        [xx, yy] = ind2sub(size(self.map), ss);
                        text(yy, xx, num2str(max_node.pseudoR(ss)), 'FontSize', v_font, 'Color', 'green');
                    end
                    title(sprintf('R~_%d', max_node.i));
                    axis off;
                end

                % Q values of Q nodes
                %
                if layer == 0
                    break;
                end
                layer = layer - 0.5; % the Q nodes

                q_nodes = {};
                for i = 1:numel(self.q_nodes)
                    if abs(self.q_nodes{i}.layer - layer) < 1e-8
                        q_nodes = [q_nodes, self.q_nodes{i}];
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
                    colormap('Gray');
                    if q_node.a == state.a && q_node.i == state.i
                        text(y, x, 'X', 'FontSize', x_font, 'FontWeight', 'bold', 'Color', 'red');
                    elseif find(q_node.a == [stack.a]) == find(q_node.i == [stack.i])
                        text(y, x, 'X', 'FontSize', x_font, 'FontWeight', 'bold', 'Color', 'black');
                    end

                    for ss = self.mdp.I
                        [xx, yy] = ind2sub(size(self.map), ss);
                        text(yy, xx, num2str(q_node.Q(ss)), 'FontSize', v_font, 'Color', 'green');
                    end

                    if i == 1, ylabel('Q(s,a)'); end
                    title(sprintf('%s (%d)', q_node.name, q_node.idx));
                    axis off;
                end

                % C values of Q nodes
                %
                m = max(8, numel(q_nodes));
                offs = max(0, floor((8 - numel(q_nodes)) / 2));
                l = l + 1;
                for i = 1:numel(q_nodes)
                    q_node = q_nodes{i};
                    subplot(n, m, i + (l-1)*m + offs);
                    ci = q_node.C(self.mdp.I);
                    imagesc(reshape(ci, size(self.map)));
                    colormap('Gray');
                    if q_node.a == state.a && q_node.i == state.i
                        text(y, x, 'X', 'FontSize', x_font, 'FontWeight', 'bold', 'Color', 'red');
                    elseif find(q_node.a == [stack.a]) == find(q_node.i == [stack.i])
                        text(y, x, 'X', 'FontSize', x_font, 'FontWeight', 'bold', 'Color', 'black');
                    end

                    for ss = self.mdp.I
                        [xx, yy] = ind2sub(size(self.map), ss);
                        text(yy, xx, num2str(q_node.C(ss)), 'FontSize', v_font, 'Color', 'green');
                    end
                    title(sprintf('C_%d(:,%d)', q_node.i, q_node.a));

                    if i == 1, ylabel('C(s,a)'); end
                    axis off;
                end

                % pseudoQ values of Q nodes
                %
                m = max(8, numel(q_nodes));
                offs = max(0, floor((8 - numel(q_nodes)) / 2));
                l = l + 1;
                for i = 1:numel(q_nodes)
                    q_node = q_nodes{i};
                    subplot(n, m, i + (l-1)*m + offs);
                    qi = q_node.pseudoQ(self.mdp.I);
                    imagesc(reshape(qi, size(self.map)));
                    colormap('Gray');
                    if q_node.a == state.a && q_node.i == state.i
                        text(y, x, 'X', 'FontSize', x_font, 'FontWeight', 'bold', 'Color', 'red');
                    elseif find(q_node.a == [stack.a]) == find(q_node.i == [stack.i])
                        text(y, x, 'X', 'FontSize', x_font, 'FontWeight', 'bold', 'Color', 'black');
                    end
                    title(sprintf('Q~_%d(:,%d)', q_node.i, q_node.a));

                    for ss = self.mdp.I
                        [xx, yy] = ind2sub(size(self.map), ss);
                        text(yy, xx, num2str(q_node.pseudoQ(ss)), 'FontSize', v_font, 'Color', 'green');
                    end

                    if i == 1, ylabel('Q~(s,a)'); end
                    axis off;
                end
            end

            axis on;
            label = sprintf('Total reward: %.2f, steps: %d', state.Rtot, numel(state.path));
            if state.done
                xlabel(['FINISHED!: ', label]);
            else
                xlabel(label);
            end


            % plot PEs
            %
            figure(self.gui_fig2);
            n = numel(stack);
            m = 1;
            for j = 1:numel(stack)
                state = stack(j);
                i = state.i;
                max_node = self.max_nodes{i};

                subplot(n, m, j);
                plot(state.rs);
                hold on;
                plot(state.pes);
                hold off;
                title(sprintf('%s (%d)', max_node.name, max_node.i));
                legend({'reward', 'PE'});
            end

            figure(self.mdp.gui_map);
        end

    end
end
