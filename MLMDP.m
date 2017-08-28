% Multitask LMDP as described in Saxe et al (2017)
% Customized for 'rooms' domain only
%
% Given states S (split into internal states I and boundary states B) and passive transitions P(s'|s), 
% creates a basis set of tasks where each task has reward -Inf at all B states except for one, which has reward 0.
% So each task from the basis set can be interpreted as saying 'Go to state Y'.
% Thus the task basis matrix Qb is an identity matrix.
% Notice that this implies that task differ only in their reward structure (which states get reward and how much).
%
% Finds a desirability basis matrix Zi where each column c with rows zi(s) = desirability f'n for basis task c
% = exponentiated expected reward V(s) when starting at state s in I and acting optimally,
% according to the rewards specified by task c. Notice that Zb = Qb.
% Does this by solving the single-task LMDP for each basis task c successively, i.e.
% sets qb = Qb(:,c) and then saves the resulting desirability f'n as Zi(:,c) = zi.
%
% Then, given a new task in the form of a specific reward structure rb = rewards for all states s in B,
% finds a set of active transitions a(s'|s) that maximize the total expected reward directly,
% using the solutions for the basis set of tasks.
%
% It works like this:
% First computes the corresponding exponentiated rewards qb = exp(rb/lambda).
% Then computes a task weighing w such that Qb w = qb (or is at least as close as possible).
% This essentially means that the new task is expressed as a linear combination of the basis tasks.
% Because of the magical properties of LMDPs (Todorov 2009), this means that the desirability f'n
% for the optimal policy is Zi w = zi, and from that we directly derive the actual optimal policy a(s'|s).
%
classdef MLMDP < LMDP

    properties (Constant = true)
        R_B_goal = 0; % reward for goal boundary state for the given basis task
        R_B_nongoal = -Inf; % reward for the other boundary states for the given basis task
    end

    properties (Access = public)
        qi = []; % exponentiated rewards for internal states only
        Qb = []; % exponentiated rewards for boundary states (row) for each task (col)
        Zi = []; % desirability function for all states (row) each task (col)
        w = []; % basis task weights that were used for the last task
    end

    methods 

        % Create a MLMDP from a maze with multiple goal states;
        % creates a separate task for each goal state
        %
        function self = MLMDP(map, absorbing_inds)
            if ~exist('absorbing_inds', 'var'), absorbing_inds = []; end

            self = self@LMDP(map, absorbing_inds);
            
            Nb = numel(self.B);

            self.qi = self.q(self.I); % internal states have the same rewards for each task

            % Qb = Nb x Nb identity matrix
            % => each column is a task with 1 goal state = the corresponding boundary state
            % and has the exponentiated rewards for all boundary states (as rows)
            %
            Rb = ones(Nb, Nb) * self.R_B_nongoal;
            Rb(logical(eye(Nb))) = self.R_B_goal;
            Qb = exp(Rb / self.lambda);
            self.Qb = Qb;

            self.q = []; % doesn't make sense any more 
            self.sanityMLMDP();
        end

        % 'Pre-solve' an initialized MLMDP for all basis tasks
        % => compute desirability matrix Z
        %
        function presolve(self)
            Zi = [];
            a = [];
            for i = 1:size(self.Qb, 2) % for each basis task

                % set rewards according to basis task
                %
                qb = self.Qb(:,i); % basis task i
                self.q = [self.qi; qb];

                % call regular LMDP solver
                %
                self.solveLMDP();
                
                if isempty(Zi)
                    Zi = self.z(self.I);
                else
                    Zi = [Zi, self.z(self.I)];
                end
            end    
            assert(size(Zi, 1) == numel(self.I));
            assert(size(Zi, 2) == size(self.Qb, 2));

            self.Zi = Zi;
            self.q = []; % clean up
            self.z = [];
            self.a = [];

            self.sanityMLMDP();
        end

        % Given a task = reward structure for the B states,
        % compute the best combination of basis tasks
        % and the corresponding actions
        %
        function w = solveMLMDP(self, rb)
            assert(size(rb, 1) == numel(self.B));
            assert(size(rb, 2) == 1);
            N = numel(self.S);

            qb = exp(rb / self.lambda);
            w = pinv(self.Qb) * qb; % Eq 7 from Saxe et al (2017)
            self.w = w;

            % find desirability f'n z
            %
            z = nan(N, 1);
            zi = self.Zi * w; % specified approximately by the task and the pre-solved desirability matrix Z
            zb = qb; % specified exactly by the task
            z(self.I) = zi;
            z(self.B) = zb;
            self.z = z;

            % find optimal policy a*
            %
            a = self.policy(z);
            self.a = a;

            % Adjust the instantaneous rewards according to rb;
            % useful for the simulations
            %
            self.R(self.B) = rb;
        end

        % Sanity check that a LMDP is correct
        %
        function sanityMLMDP(self)
            % States
            %
            N = numel(self.S);
            Ni = numel(self.I);
            Nb = numel(self.B);

            % Rewards
            %
            assert(isempty(self.Zi) || size(self.Zi, 1) == Ni);
            assert(isempty(self.Zi) || size(self.Zi, 2) == Nb); % note that this is not a strict requirements for MLMDPs; it's self-imposed
            assert(size(self.Qb, 1) == Nb);
            assert(size(self.Qb, 2) == Nb); % note that this is not a strict requirements for MLMDPs; it's self-imposed
            assert(size(self.qi, 1) == Ni);
            assert(size(self.qi, 2) == 1);
            assert(isempty(self.q));
        end

    end
end
