% MAXQ algorithm from Dietterich (2000)
% Works with 'rooms' domain only. Similar to the maze in Figure 6 of his paper
%
classdef MAXQ < handle

    properties (Constant = true)
        R_I = -1; % penalty for remaining stationary. TODO dedupe with MDP.R_I 

        % Maze
        %
        subtask_symbols = 'ABCDEF';
    end

    properties (Access = public)
        Max_nodes = {}; % Max nodes
        Q_nodes = {}; % Q nodes

        root = []; % root node of the MAXQ graph

        mdp = []; % low-level MDP

        % Maze
        %
        map = [];
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
                max_node.name = ['MaxSubtask ', c];
                max_node.I = subtask_inds'; % internal "active" states (S in the paper) = all states labeled for the subtask
                max_node.B = setdiff(mdp.S, max_node.I); % boundary "terminated" states (T in the paper)
                max_node.V = zeros(numel(mdp.S), 1); % Vi(s) from paper
      
                % create children q-nodes == the primitive actions.
                % they are parents of the primitve action max nodes
                max_node.children = [];
                for i = mdp.A
                    q_node = struct;
                    q_node.child = i; % max node i
                    q_node.layer = 0.5;
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
                    q_node.child = i;
                    q_node.layer = 1.5;
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
        
        function stack = step0(self, stack)
            spaces = repmat(' ', 1, numel(stack) * 5);

            state = stack(end);
            s = state.s;
            p = state.p;

            max_node = self.Max_nodes{p};
            q_children = max_node.children;
            q_nodes = [self.Q_nodes{q_children}];
            max_children = [q_nodes.child]; % max children of the current max node (subroutine) p

            if ~state.second_half
                % Do the regular loop from maxQQ
                % up to calling maxQQ recursively, or executing a primitive action
                %

                % Choose action / subroutine a
                %
                pi = self.glie(s, p);
                j = samplePF(pi);
                pa = q_children(j);
                a = max_children(j);
                stack(end).a = a;
                stack(end).pa = pa;

                fprintf('\n\n\n%s At maxQQ(%d, %d), executing a = %d (pa = %d, pi = %s, q children = %s, max children = %s)\n', spaces, s, p, a, pa, sprintf('%.2f ', pi), sprintf('%d ', q_children), sprintf('%d ', max_children));
                disp(max_node);

                % Execute a
                %
                if self.Max_nodes{a}.is_primitive
                    % If a is primitive, get reward r and new state s' from low-level MDP
                    %
                    new_s = samplePF(self.mdp.P(:,s,a));
                    r = self.mdp.R(new_s);

                    % Update Vi(s) for primitive node here (b/c we don't call maxQQ on it)
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
                    fprintf('%s Complex action -> calling maxQQ(%d, %d)\n', spaces, s, a);

                    state = self.init_state0(s, a);
                    stack = [stack, state];
                    %[r, new_s] = self.maxQQ(s, a);
                end

            else
                % Do the second half of the recursion
                %

                % get locals from stack
                %
                a = stack(end).a;
                pa = stack(end).pa;
                new_s = stack(end).new_s;

                fprintf('%s ...back to maxQQ(%d, %d) after calling maxQQ(%d, %d): new_s = %d, r = %.2f\n', spaces, s, p, s, a, new_s, stack(end).r);

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

        function [Rtot, new_s] = maxQQ(self, s, p)
            if ~exist('p', 'var')
                p = self.root; % start from the top
            end
            spaces = repmat(' ', 1, (2 - self.Max_nodes{p}.layer) * 5);
            
            Rtot = 0;
            done = false;
            while ~done
                max_node = self.Max_nodes{p};
                q_children = max_node.children;
                q_nodes = [self.Q_nodes{q_children}];
                max_children = [q_nodes.child]; % max children of the current max node (subroutine) p

                % Choose action / subroutine a
                %
                pi = self.glie(s, p);
                j = samplePF(pi);
                pa = q_children(j);
                a = max_children(j);

                fprintf('\n\n\n%s At maxQQ(%d, %d), executing a = %d (pa = %d, pi = %s, q children = %s, max children = %s)\n', spaces, s, p, a, pa, sprintf('%.2f ', pi), sprintf('%d ', q_children), sprintf('%d ', max_children));
                disp(max_node);

                % Execute a
                %
                if self.Max_nodes{a}.is_primitive
                    % If a is primitive, get reward r and new state s' from low-level MDP
                    %
                    new_s = samplePF(self.mdp.P(:,s,a));
                    r = self.mdp.R(new_s);

                    % Update Vi(s) for primitive node here (b/c we don't call maxQQ on it)
                    %
                    pe = r - self.Max_nodes{a}.V(s); % Eq 11: PE = R(s') - Vi(s)
                    self.Max_nodes{a}.V(s) = self.Max_nodes{a}.V(s) + MDP.alpha * pe; % Eq 11: Vi(s) = Vi(s) + alpha * PE
                    
                    fprintf('%s       Primitive action -> new_s = %d, r = %.2f, Rtot = %.2f, pe = %.2f, V(a,s) = %.2f\n', spaces, new_s, r, Rtot, pe, self.Max_nodes{a}.V(s));

                else
                    % a is a subroutine -> find reward r and new state s' recursively from children
                    %
                    fprintf('%s Complex action -> calling maxQQ(%d, %d)\n', spaces, s, a);

                    [r, new_s] = self.maxQQ(s, a);

                    fprintf('%s ...back to maxQQ(%d, %d) after calling maxQQ(%d, %d): new_s = %d, r = %.2f\n', spaces, s, p, s, a, new_s, r);
                end

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


        function plot_gui(self)
            figure;

            n = 5;
            l = 0;
            for layer = 2:-1:0
                max_nodes = {};
                for i = 1:numel(self.Max_nodes)
                    if self.Max_nodes{i}.layer == layer
                        max_nodes = [max_nodes, self.Max_nodes{i}];
                    end
                end

                m = numel(max_nodes);
                l = l + 1;
                for i = 1:m
                    max_node = max_nodes{i};
                    subplot(n, m, i + (l-1)*m);
                    vi = max_node.V(self.mdp.I);
                    imagesc(reshape(vi, size(self.map)));
                    ylabel('V(s)');
                    title(max_node.name);
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

                m = numel(q_nodes);
                l = l + 1;
                for i = 1:m
                    q_node = q_nodes{i};
                    subplot(n, m, i + (l-1)*m);
                    qi = q_node.Q(self.mdp.I);
                    imagesc(reshape(qi, size(self.map)));
                    ylabel('Q(s,a)');
                    title(q_node.name);
                end
            end
        end

    end
end
