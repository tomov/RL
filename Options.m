% Two-layer options framework as in Sutton et al (1999).
% Works with 'rooms' domain only.
% Uses subgoals to define the options (as Sec 7)
%
classdef Options < handle

    properties (Constant = true)
        R_I = -1; % penalty for remaining stationary. Different from TD.R_i = 0

        % Maze
        %
        pseudoreward_symbol = '1';
        subtask_symbol = 'S';
    end

    properties (Access = public)
        O = []; % all options

        pi = {}; % pi{o}(:) = deterministic policy for option o: pi{o}(s) = what (lower-level) action to take at state s while executing option o. Roughly corresponds to Zi(o,:) which tells you the policy for subtask o in the MLMDP framework
        T = {}; % TD learner for each option, for debugging

        aT = []; % augmented TD learner with the options

        % Maze
        %
        map = [];
    end

    methods

        % Initialize an Options framework from a maze
        %
        function self = Options(map)
            self.map = map;

            %
            % Set up options and their policies based on subtask states S
            %

			subtask_inds = find(map == Options.subtask_symbol)';
            goal_inds = find(ismember(map, TD.absorbing_symbols));

			map(subtask_inds) = TD.empty_symbol; % erase subtask states
			map(goal_inds) = TD.empty_symbol; % erase goal states

            O = 1:numel(subtask_inds); % 
            self.pi = cell(numel(O), 1); % set of policies for each option

            % For each subtask, find the optimal policy that gets you
            % from anywhere to its goal state. 
            %
			for s = subtask_inds
                % Set rewards of 0 (TD.R_I) for all internal states
                % and reward of 1 at the subtask state
                %
                map(s) = Options.pseudoreward_symbol;

                T = TD(map);
                T.solveGPI();
                
                o = find(subtask_inds == s);
                self.pi{o} = T.pi; % policy for option o corresponding to subtask with goal state s
                self.T{o} = T; % for debugging

                map(s) = TD.empty_symbol;
			end

            %
            % Create a augmented TD learner with the options
            %

            map = self.map;
            map(subtask_inds) = '.'; % erase subtask states

            T = TD(map);

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

            T.R(T.I) = Options.R_I; % different penalty for staying still

            self.O = O;
            self.aT = T;
        end 

        % Run an episode and update Q-values using Q-learning
        % TODO dedupe with sampleQ for TD learner
        %
        function [Rtot, path] = sampleQ(self, s)
            if ~exist('s', 'var')
                s = find(self.aT.map == self.aT.agent_symbol);
            end
            assert(numel(find(self.aT.I == s)) == 1);

            Rtot = 0;
            path = [];

            map = self.aT.map;
            disp(map);
            while true
                Rtot = Rtot + self.aT.R(s);
                path = [path, s];

                [x, y] = self.aT.I2pos(s);
            
                a = NaN;
                while isnan(a) || max(self.aT.P(:,s,a)) == 0 % this means the action is not allowed; we could be more explicit about this TODO
                    a = self.aT.eps_greedy(s);
                end
                new_s = samplePF(self.aT.P(:,s,a));
            
                oldQ = self.aT.Q(s,a); % for debugging
                if ismember(a, self.O)
                    % option -> execute optioopolicy until the end
                    %
                    o = find(self.O == a);
                    [~, option_path] = self.T{o}.sampleGPI(s); % execute option policy
                    option_path = option_path(2:end-1); % don't count s and fake B state
                    k = numel(option_path);
                    assert(k > 0); % b/c can't take an option from its goal state
                    r = self.aT.R(option_path) .* TD.gamma .^ (0:k-1)'; % from Sec 3.2 of Sutton (1999)
                    fprintf('       option! %d -> path = %s, r = %s\n', o, sprintf('%d ', option_path), sprintf('%.2f ', r));
                    pe = sum(r) + (TD.gamma ^ k) * max(self.aT.Q(new_s, :)) - self.aT.Q(s, a);
                else
                    % primitive action -> regular Q learning
                    %
                    pe = self.aT.R(new_s) + TD.gamma * max(self.aT.Q(new_s, :)) - self.aT.Q(s, a);
                end

                self.aT.Q(s,a) = self.aT.Q(s,a) + self.aT.alpha * pe;
                
                if ismember(new_s, self.aT.B)
                    % Boundary state
                    %
                    fprintf('(%d, %d), %d --> END [%.2f%%], old Q = %.2f, pe = %.2f, Q = %.2f\n', x, y, a, self.aT.P(new_s, s, a) * 100, oldQ, pe, self.aT.Q(s, a));

                    Rtot = Rtot + self.aT.R(new_s);
                    path = [path, new_s];
                    break;
                end

                % Internal state
                %
                [new_x, new_y] = self.aT.I2pos(new_s);
                
                map(x, y) = self.aT.empty_symbol;
                map(new_x, new_y) = self.aT.agent_symbol;
                
                fprintf('(%d, %d), %d --> (%d, %d) [%.2f%%], old Q = %.2f, pe = %.2f, Q = %.2f\n', x, y, a, new_x, new_y, self.aT.P(new_s, s, a) * 100, oldQ, pe, self.aT.Q(s, a));
                disp(map);
                
                s = new_s;
            end
            fprintf('Total reward: %d\n', Rtot);
        end
    end
end
