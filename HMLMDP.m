% Hierarchical multitask LMDP as described in Saxe et al (2017)
% Customized for 'rooms' domain only, supports 2 layers only.
% Can decompose the desirability basis matrix Zi using nonnegative matrix factorization
% as described in Earle et al (2017) by passing full=true and k in the constructor.
%
% Given states S (split into internal states I, boundary states B, and subtask states St) and passive transition dynamics P(s'|s),
% creates a two-layer hierarchical MLMDP. 
%
% The lower layer is just an augmented MLMDP (see AMLMDP).
% The higher layer is a MLMDP with the following properties:
% The I states of the higher layer correspond to the St states of the lower layer.
% Each I state of the higher layer has a corresponding B state.
% The I --> I transition dynamics Pi of the higher layer are defined by the transition dynamics Pi
% of the lower layer, s.t. St states (s, s') that are closer in the lower layer have higher probabiliy P(s'|s)
% in the higher layer.
% The I --> B transition dynamics Pb = identity * small probability, i.e. an I states can go to its
% corresponding B state with some fixed small probability.
% Like regular MLMDP's, the higher layer basis set of tasks Qb is the identity matrix,
% i.e. each task has -Inf rewards at all B states except for one, which has reward 0.
%
% Then, given a new task in the form of a specific reward structure rb = rewards for all B states in the
% lower layer, it finds a step-by-step solution via the hierarchy.
% First, it solves the lowest layer (notice that right now we do the full-blown solution so we don't 
% really need the hierarchy; however, it becomes valuable if we do Z-iteration with only a few steps instead).
% by finding an optimal blend of basis tasks (defined by the weights w), computing the blend desirability f'n
% zi = Zi * w, and deriving a(s'|s) directly from that.
% Initially, St states are only given small rewards to encourage exploration.
% Then starts following the lower-level policy defined by a(s'|s).
% Once it enters a St state, it goes to the higher level of the hierarchy. The first thing it does then
% is to come up with a corresponding higher-layer 'task' i.e. a set of rewards rb for the B states
% of the higher level. Currently, this is simply the expected boundary reward under the lower-layer
% passive dynamics, i.e. what you would get when starting at the lower-layer I state corresponding
% to the lower-layer St state corresponding to the higher-layer I state corresponding to the higher-layer
% B state, and ending up at a lower-level B state.
% So on the higher layer, we compute rb => qb = exp(rb/lambda) => w = Qb^-1 * qb => zi = Zi * w => a(s'|s).
% These actions tell us which I states on the higher layer are preferred when starting at s => which
% St states on the lower layer are preferred from wherever we currently are (i.e. when we entered the higher
% layer). We use this to re-compute the rewards rt for the St states on the lower layer (based on Eq 10),
% and in turn re-compute qb, w, zi, and a(s'|s) on the lower layer.
% Then we go back to the lower layer and continue sampling.
%
classdef HMLMDP < handle

    properties (Constant = true)
        subtask_symbol = 'S';
        goal_symbol = '$'; % we use absorbing_symbol to distinguish all boundary states, however not all of them are goals in the actual task.

        R_goal = 7; % the actual reward for completing the task; don't confuse with R_B_goal; interacts w/ rt -> has to be > 0 o/w the rt's of undersirable St's are 0 and compete with it, esp when X passes through them -> it is much better to go into the St state than to lose a few more -1's to get to a cheap goal state; but if too high -> never enter St states...
        R_nongoal = -Inf; % the rewards at the other boundary states for the task; not to be confused with R_B_nongoal
        R_St = -1; % reward for St states to encourage entering them every now and then; determines at(:,:); too high -> keeps entering St state; too low -> never enters St state... TODO

        alpha = 0.05; % how often agent transitions to higher layer; used for full HMLMDP with non-negative matrix factorization (Earle et al 2017) TODO dedupe with ALMDP.P_I_to_St

        rt_coef = 100; % coefficient by which to scale rt when recomputing weights on current level based on higher-level solution; for big HMLMDP's, must be large, on the order of the size of the grid -- since the difference in a(.|s) - P(.|s) is on the order of 1, and we want to push the agent away from negative St's (far from the reward) to positive St's (closer to it), however if it's reward is e.g. -40 and the goal state is 10 but it's 70 squares away (each with -1), it will prefer to go back to the St state
        rb_next_level_coef = 10; % coefficient by which to scale rb_next_level
    end

    properties (Access = public)
        M = []; % current level (augmented) MLMDP
        next = []; % next level HMLMDP #recursion

        Ptb = []; % probability of ending at a given B state under passive dynamics, given that we've started at a St state. Used to compute the next-level boundary rewards

        state_gui = []; % state for the GUI step-through
        map_gui = []; % figure for the GUI
        Zi_gui = []; % figure for the GUI
    end

    methods 
        function self = HMLMDP(arg, full, k)
            if ~exist('full', 'var')
                full = false; % whether to have S and B state for every I state, or to use the map
            end

            if isa(arg, 'HMLMDP')
                %
                % We're at the next level of the hierarchy
                %

                M1 = arg.M; 
                assert(isa(M1, 'AMLMDP'));

                Ni = numel(M1.St); % St from lower level == I on current level

                % use a fake map to create the current level MLMDP
                % it sets up stuff, including S, I, R and Qb
                %
                M = MLMDP(repmat('0', [1 Ni]));

                % Set up states
                %
                Nb = Ni;
                N = Ni + Nb;
                assert(isequal(M.I, 1 : Ni));
                assert(isequal(M.B, Ni + 1 : 2 * Ni));

                % Set up passive transition dynamics P according to lower level
                %
                M1_Ni = numel(M1.I);
                M1_Pi = M1.P(M1.I, M1.I);
                M1_Pt = M1.P(M1.St, M1.I);
                M1_Pb = M1.P(setdiff(M1.B, M1.St), M1.I); % note we exclude St from these

                Pi = M1_Pt * inv(eye(M1_Ni) - M1_Pi) * M1_Pt'; % I --> I from low-level dynamics, Eq 8 from Saxe et al (2017) % note the diagonal entries of Pi will be high (b/c random walk likes to stay around same place rather than go to another specific St state) -> this is crucial; b/c then the active dynamics will prefer going elsewhere -> ai(:,s) - Pi(:,s) will be large negative -> it will discourage agent from wandering back into the same state
                %Pb = M1.Pb * inv(eye(M1.Ni) - M1.Pi) * M1.Pt'; Eq 9 from Saxe et al (2017)
                Pb = eye(Ni) * LMDP.P_I_to_B; % small prob I --> B

                % TODO dedupe with Augment P in AMLMDP
                M.P = zeros(N, N);
                M.P(M.I, M.I) = Pi;
                M.P = (1 - LMDP.P_I_to_B) * M.P ./ sum(M.P, 1); % normalize but leave room for P_I_to_B
                M.P(isnan(M.P)) = 0;
                M.P(M.B, M.I) = Pb; % b/c these are normalized to begin with
                % normalize
                M.P = M.P ./ sum(M.P, 1);
                M.P(isnan(M.P)) = 0;
                assert(sum(abs(sum(M.P, 1) - 1) < 1e-8 | abs(sum(M.P, 1)) < 1e-8) == N);

                % Compute P(end up at given B state | start at given St state) based on the lower level passive dynamics; we need this for calculating the boundary rewards
                %
                Ptb = M1_Pb * inv(eye(M1_Ni) - M1_Pi) * M1_Pt'; % when you start at s in St on lower level, what prob you will end up at b in B\St (lower level) under passive dynamics
                Ptb = Ptb ./ sum(Ptb, 1); % normalize
                Ptb(isnan(Ptb)) = 0;
                self.Ptb = Ptb;

                self.M = M;

                % Presolve it -> generate desirability basis matrix Z
                self.M.presolve();
            else
                %
                % We're at the lowest level of the hierarchy
                %

                map = arg;
                assert(ischar(map));

                if full
                    %
                    % the 'full' case.
                    % Each I state has a corresponding B state and St state.
                    % Optionally perform non-negative matrix factorization as in Earle et al (2017)
                    % to find a suitable basis set of tasks and shrink the number of St states
                    %
                    subtask_inds = 1:numel(map);
                    absorbing_inds = 1:numel(map);
                    assert(isempty(find(map == HMLMDP.subtask_symbol))); % in this case, do not supply S states, just pass a regular ol' map

                    if exist('k', 'var')
                        % We supplied k => perform non-negative matrix factorization to fin
                        % a suitable basis set of tasks.
                        % that's the interesting case
                        %

                        % Create regular MLMDP to find the basis matrix Z for the bundary states
                        M = MLMDP(map, absorbing_inds);

                        % Presolve it -> generate desirability basis matrix Z
                        M.presolve();

                        % find a basis set of tasks using nonnegative matrix factorization (Earle et al 2017)
                        %
                        Zi = M.Zi;
                        [D, W] = nnmf(Zi, k); % Eq. 1 from Earle et al (2017)

                        Pt = self.alpha * (D / max(D(:))); % scale D to [0, 1] to get Pt

                        % Create an augmented MLMDP and 'tweak' it to use the uncovered basis tasks
                        %
                        M = AMLMDP(map, 1:k, absorbing_inds); % pass fake subtask_inds; we fix stuff later

                        % Fix Pt to reflect the uncovered basis tasks D
                        %
                        M.P(M.St, M.I) = Pt';
                        % re-normalize
                        M.P = M.P ./ sum(M.P, 1);
                        M.P(isnan(M.P)) = 0;
                        assert(sum(abs(sum(M.P, 1) - 1) < 1e-8 | abs(sum(M.P, 1)) < 1e-8) == numel(M.S));

                        % Pre-solve augmented MLMDP with the basis tasks TODO a bit redundant since already solved for non-St B states in first MLMDP
                        %
                        M.presolve();

                        self.M = M;

                        % Initialize next layer of the hierarchy
                        self.next = HMLMDP(self);
                    else
                        % Each state has a corresponding St state. Pretty useless
                        % TODO dedupe with non-full case
                        %
                        % Create augmented MLMDP
                        self.M = AMLMDP(map, subtask_inds, absorbing_inds);
                       
                        % Presolve it -> generate desirability basis matrix Z
                        self.M.presolve();

                        % Initialize next layer of the hierarchy
                        self.next = HMLMDP(self);
                    end

                else
                    %
                    % the old-fashioned, non-full case
                    % S = subtask (St) state, $ or number = boundary (B) state
                    %
                    subtask_inds = find(map == HMLMDP.subtask_symbol)';
                    absorbing_inds = []; % infer them from the map
                    map(subtask_inds) = LMDP.empty_symbol; % even though AMLMDP's are aware of subtask states, they don't want them in the map b/c they just pass it down to the MLMDP's which don't know what those are

                    % Create augmented MLMDP
                    self.M = AMLMDP(map, subtask_inds, absorbing_inds);
                   
                    % Presolve it -> generate desirability basis matrix Z
                    self.M.presolve();

                    % Initialize next layer of the hierarchy
                    self.next = HMLMDP(self);
                end

            end
        end

        % Sample a path through the HMLMDP
        %
        function [Rtot, path] = sample(self, s)
            if ~exist('s', 'var')
                state = self.init_sample();
            else
                state = self.init_sample(s);
            end

            map = self.M.map;
            disp(map)

            iter = 1;
            while ~state.done
                [x, y] = self.M.I2pos(state.s);
                old_s = state.s;
               
                state = self.step(state);

                if ismember(state.s, self.M.I)
                    % Internal state -> just move to it
                    %
                    [new_x, new_y] = self.M.I2pos(state.s);
                    map(x, y) = LMDP.empty_symbol;
                    map(new_x, new_y) = LMDP.agent_symbol;
                    fprintf('(%d, %d) --> (%d, %d) [%.2f%%]\n', x, y, state.s, new_y, self.M.a(state.s,old_s) * 100);
                    disp(map);

                elseif ismember(new_s, self.M.St)
                    % Higher layer state i.e. subtask state
                    %
                    fprintf(' ... BACK FROM NEXT LEVEL! (%d, %d) --> %d [%.2f%%] !!!\n', x, y, state.s_next_level, self.M.a(new_s, s) * 100);
                else
                    fprintf('(%d, %d) --> END [%.2f%%]\n', x, y, self.M.a(state.s, old_s) * 100);
                end

                iter = iter + 1;
                if iter >= 40, break; end
            end

            fprintf('Total reward: %d\n', state.Rtot);

            Rtot = state.Rtot;
            path = state.path;
        end

        % Initialize a sampler through the HMLMDP
        %
        function state = init_sample(self, s)
            % Find starting state
            %
            if ~exist('s', 'var')
                s = find(self.M.map == LMDP.agent_symbol);
            end

            % Find goal state(s)
            %
            e = find(self.M.map == HMLMDP.goal_symbol);
            e = self.M.I2B(e); % get corresponding boundary state(s)
            assert(~isempty(e));
            assert(isempty(find(e == 0))); % make sure they all have corresponding boundary states

            % Set up reward structure according to goal state(s)
            %
            state.rb = HMLMDP.R_nongoal * ones(numel(self.M.B), 1); % non-goal B states have q = 0
            state.rb(ismember(self.M.B, e)) = HMLMDP.R_goal; % goal B states have an actual reward
            state.rb(ismember(self.M.B, self.M.St)) = HMLMDP.R_St; % St states have a small reward to encourage exploring them every now and then

            % Find solution on current level based on reward structure
            % Notice that if we do this directly using inversion, it makes the whole hierarchy a bit futile.
            % The hierarchy makes sense if we use Z-iteration (e.g. b/c the space is big and sparce)
            % and we only run a constant number of steps, e.g. the agent can reach a subtask state
            % and the subtask states can reach each other => then she can jump from one subtask state
            % to another until she reaches the end.
            %
            self.M.solveMLMDP(state.rb);

            % Solve the HMLMDP by sampling from multiple levels
            %
            state.Rtot = 0;
            state.path = [];
            state.s = s;
            state.done = false;
            state.s_next_level = [];
            state.rb_next_level = zeros(numel(self.M.St), 1);
            state.rt = state.rb(ismember(self.M.B, self.M.St));
        end

        % step through a sample path in the HMLMDP
        %
        function state = step(self, state)
            % Some helper variables
            %
            Pt = self.M.P(self.M.St, self.M.I);

            new_s = samplePF(self.M.a(:,state.s));

            if ismember(new_s, self.M.I)
                % Internal state -> just move to it
                %
                state.Rtot = state.Rtot + self.M.R(new_s);
                state.path =[state.path, new_s];
                state.s = new_s;

            elseif ismember(new_s, self.M.St)
                % Higher layer state i.e. subtask state
                %
                s_next_level = find(self.M.St == new_s); % St state on current level == I state on higher level

                state.s_next_level = s_next_level;

                % solve next level MLMDP
                %

                % andrew's suggestion for computing rb_next_level
                % doesn't work well with -Inf's though... but non--Inf's don't make much sense
                %{
                assert(~isinf(HMLMDP.R_nongoal)); % andrew's suggestion doesn't work with inf's
                rb_next_level = self.next.Ptb' * state.rb(~ismember(self.M.B, self.M.St)); % Andrew's suggestion
                rb_next_level = rb_next_level * self.rb_next_level_coef;
                %}

                % my chicanery
                assert(isinf(HMLMDP.R_nongoal));
                w = self.M.w;
                w(ismember(self.M.B, self.M.St)) = 0;
                zi = self.M.Zi * w;
                %rb_next_level = log(zi(self.M.St2I(self.M.St))) * LMDP.lambda;
                rb_next_level = log(Pt * zi) * LMDP.lambda;

                state.rb_next_level = rb_next_level;

                %rb_next_level = [4 -1]';

                fprintf('\n\n ------- NEXT LEVEL ---------- \n\n');
                fprintf('                rb_next_level = [%s]\n', sprintf('%.3f, ', rb_next_level));
                self.next.M.solveMLMDP(rb_next_level);

                % sample until a boundary state
                %
                [~, path] = self.next.M.sample(s_next_level); % not really necessary here

                % recalculate reward structure on current level
                % based on the optimal policy on the higher level
                %
                ai = self.next.M.a(self.next.M.I, :);
                Pi = self.next.M.P(self.next.M.I, :);
                rt = (ai(:, s_next_level) - Pi(:, s_next_level)) * self.rt_coef; % Eq 10 from Saxe et al (2017)
                state.rt = rt;

                assert(size(rt, 1) == numel(self.M.St));
                assert(size(rt, 2) == 1);
                fprintf('               ai(:,s) = [%s]\n', sprintf('%.3f, ', ai(:, s_next_level)));
                fprintf('               Pi(:,s) = [%s]\n', sprintf('%.3f, ', Pi(:, s_next_level)));
                fprintf('               ai - Pi = [%s]\n', sprintf('%.3f, ', rt / self.rt_coef));
                fprintf('                    rt = [%s]\n', sprintf('%.3f, ', rt));
                fprintf('                old rb = [%s]\n', sprintf('%.3f, ', state.rb));
                state.rb(ismember(self.M.B, self.M.St)) = rt;
                fprintf('                new rb = [%s]\n', sprintf('%.3f, ', state.rb));

                % recompute the optimal policy based on the 
                % new reward structure
                %
                w = self.M.solveMLMDP(state.rb);
                fprintf('                    wt = [%s]\n', sprintf('%.3f, ', w(ismember(self.M.B, self.M.St))));
                fprintf('                new zi = [%s]\n', sprintf('%.3f, ', self.M.z(self.M.I)'));
                fprintf('                new zb = [%s]\n', sprintf('%.3f, ', self.M.z(self.M.B)'));
                a_I_to_St = max(self.M.a(self.M.St,:), [], 2)';
                fprintf('               a(St|I) = [%s]\n', sprintf('%.3f, ', a_I_to_St));

            else
                % regular ol' boundary state
                %

                state.Rtot = state.Rtot + self.M.R(new_s);
                state.path = [state.path, new_s];
                state.done = true;
            end
        end

        % Sample a path through the HMLMDP with a nice GUI
        %
        function [Rtot, path] = sample_gui(self, s)
            if ~exist('s', 'var')
                self.state_gui = self.init_sample();
            else
                self.state_gui = self.init_sample(s);
            end

			self.map_gui = figure;
            self.Zi_gui = figure;
            self.plot_gui();

			step_button = uicontrol('Style', 'pushbutton', 'String', 'Step', ...
									 'Position', [10 10 70 20], ...
									 'Callback', @self.step_gui_callback);
        end

        % plot state of sample in the GUI
        %
        function plot_gui(self)
            figure(self.map_gui);

            % plot map and current desirability f'n
            %
            subplot(1, 2, 1);
            zi = self.M.z(self.M.I);
            imagesc(log(reshape(zi, size(self.M.map))));
            [x, y] = ind2sub(size(self.M.map), self.state_gui.s);
            text(y, x, 'X', 'FontSize', 10, 'FontWeight', 'bold');
            if self.state_gui.done
                xlabel(sprintf('FINISHED Total reward: %.2f', self.state_gui.Rtot));
            else
                xlabel(sprintf('Total reward: %.2f', self.state_gui.Rtot));
            end
            title('Desirability function z_i^1');

            % plot the weights of the boundary state tasks
            %
            subplot(1, 2, 2);
            wb = self.M.w(~ismember(self.M.B, self.M.St));
            if numel(wb) < numel(self.M.I)
                w = zeros(numel(self.M.I), 1);
                w(self.M.I2B > 0) = wb;
                wb = w;
            end
            imagesc(reshape(wb, size(self.M.map)));
            text(y, x, 'X', 'FontSize', 10, 'FontWeight', 'bold');
            title('Boundary state weights w_b^1');

            % plot desirability f'n of basis tasks 
            %
            figure(self.Zi_gui);
            self.plotZi();

            % back to map gui
            figure(self.map_gui);
        end

        % step thru sample with GUI
        %
        function step_gui_callback(self, hObject, eventdata, handles)
            if self.state_gui.done
                return
            end

            [x, y] = self.M.I2pos(self.state_gui.s);
            old_s = self.state_gui.s;
               
            self.state_gui = self.step(self.state_gui);
            disp(self.state_gui.s);
            self.plot_gui();

            map = self.M.map;
            if ismember(self.state_gui.s, self.M.I)
                % Internal self.state_gui -> just move to it
                %
                [new_x, new_y] = self.M.I2pos(self.state_gui.s);
                map(new_x, new_y) = LMDP.agent_symbol;
                fprintf('(%d, %d) --> (%d, %d) [%.2f%%]\n', x, y, new_x, new_y, self.M.a(self.state_gui.s,old_s) * 100);
                disp(map);

            elseif ismember(new_s, self.M.St)
                % Higher layer self.state_gui i.e. subtask self.state_gui
                %
                fprintf(' ... BACK FROM NEXT LEVEL! (%d, %d) --> %d [%.2f%%] !!!\n', x, y, self.state_gui.s_next_level, self.M.a(new_s, s) * 100);
            else
                fprintf('(%d, %d) --> END [%.2f%%]\n', x, y, self.M.a(self.state_gui.s, old_s) * 100);
            end

        end

		% Plot the subtask desirability f'ns
		%
        function plotZi(self)
            St_in_B = find(ismember(self.M.B, self.M.St));
            if numel(self.M.St) == numel(self.M.I)
                n = size(self.M.map, 1);
                m = size(self.M.map, 2);
            else
                n = ceil(sqrt(numel(self.M.St)));
                m = ceil(sqrt(numel(self.M.St)));
            end

			for s = 1:numel(self.M.St)
				zi = self.M.Zi(:, St_in_B(s));
				
				[x, y] = ind2sub(size(self.M.map), s);
                if numel(self.M.St) == numel(self.M.I)
                    ind = sub2ind([size(self.M.map, 2) size(self.M.map, 1)], y, x);
                else
                    ind = s;
                end
				subplot(n, m, ind);
				imagesc(log(reshape(zi, size(self.M.map))));

                if ~isempty(self.state_gui)
                    xlabel(sprintf('r_b^2 = %.2f, r_t^1 = %.2f, w_t^1 = %.2f', self.state_gui.rb_next_level(s), self.state_gui.rt(s), self.M.w(St_in_B(s))));
                end
			end
        end

		% Plot the subtasks after non-negative matrix factorization; only works for full HMLMDP's TODO make better
		%
        function plotD(self, k)
			assert(numel(self.M.I) == numel(self.M.St));

			figure;
            St_in_B = find(ismember(self.M.B, self.M.St));
	        Zi = self.M.Zi(:, St_in_B);
            [D, W] = nnmf(Zi, k);

			for s = 1:k
				di = D(:,s);
				
				subplot(1, k, s);
				imagesc(reshape(di, size(self.M.map)));
			end
        end
    end

end
