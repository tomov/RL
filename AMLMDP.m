% Augmented Multitask LMDP as described in Saxe et al (2017)
% Customized for 'rooms' domain only.
%
% Given states S (split into internal states I, boundary states B, and subtask states St) and passive transition dynamics P(s'|s),
% creates a regular MLMDP with states {I, B} and dynamics {Pi, Pb},
% and then augments it with the subtask states St, the transition dynamics Pt, and the subtask basis tasks Qt.
%
% The subtask states St are incorporated as additional boundary states and from the point of view of the MLMDP,
% they become indistinguishable from the regular boundary states B. Thus the augmented MLMDP has new B = {B, St},
% new Pb = {Pb, Pt}, new Qb = [Qb 0; 0 Qt].
% The only difference is that we keep a separate list of the subtask states St so the 
% HMLMDP can go up a level in the hierarchy when it reaches one of them.
%
classdef AMLMDP < MLMDP

    properties (Constant = true)
        R_St_goal = 0; % reward for goal subtask state 
        R_St_nongoal = -Inf; % reward the other subtask states

        P_I_to_St = LMDP.P_I_to_B; % (normalized) move from I state to corresponding St state

    end

    properties (Access = public)
        St = []; % subtask states
    end

    methods 

        function self = AMLMDP(map, subtask_inds, absorbing_inds)
            if ~exist('absorbing_inds', 'var'), absorbing_inds = []; end

            self = self@MLMDP(map, absorbing_inds);
            N = numel(self.S);
            Nb = numel(self.B);

            assert(size(subtask_inds, 1) == 1); % must be a row vector, just like S and St

            % Augment state space S with St states
            % Same for boundary states B
            %
            I_with_St = subtask_inds; % subtask squares = internal states with corresponding subtask states
            Nt = numel(I_with_St);
            St = N + 1 : N + Nt;
            S_augm = [self.S, St];
            B_augm = [self.B, St];
            N_augm = N + Nt;
            Nb_augm = Nb + Nt;

            % Mappings between I states an St states
            %
            %I2St(I_with_St) = St; % mapping from I states to corresponding St states
            %St2I(I2St(I2St > 0)) = I_with_St; % mapping from St states to corresponding I states
           
            % To be consistent
            %
            R_augm = zeros(N_augm, 1);
            R_augm(self.S) = self.R;
            R_augm(St) = self.R_St_nongoal;

            % Create Qt
            %
            Rt = ones(Nt, Nt) * self.R_St_nongoal;
            Rt(logical(eye(Nt))) = self.R_St_goal;
            Qt = exp(Rt / self.lambda);

            % Created augmented Qb = [Qb 0; 0 Qt]
            %
            Qb_augm = zeros(Nb_augm, Nb_augm);
            Qb_augm(1:Nb, 1:Nb) = self.Qb;
            Qb_augm(Nb+1:Nb+Nt, Nb+1:Nb+Nt) = Qt;

            % Augment P 
            P_augm = zeros(N_augm, N_augm);
            P_augm(self.S, self.S) = self.P;
            P_augm(:, I_with_St) = P_augm(:, I_with_St) * (1 - self.P_I_to_St); % since P_I_to_St is normalized, we need to make 'room' for it; notice that this also de-normalizes the I --> B transition for states in I_with_St (makes it smaller than P_I_to_B)
            which = ~ismember(map(I_with_St), LMDP.impassable_symbols); % exclude I's that you can't get to anyway
            I_to_St = sub2ind(size(P_augm), St(which), I_with_St(which));
            P_augm(I_to_St) = self.P_I_to_St;
            assert(sum(abs(sum(P_augm, 1) - 1) < 1e-8 | abs(sum(P_augm, 1)) < 1e-8) == N_augm);

            % Change object
            % we do this in the end b/c we need the old object as we're creating the augmented data structures
            %
            self.St = St;
            self.S = S_augm;
            self.B = B_augm;
            self.Qb = Qb_augm;
            self.P = P_augm;
            self.R = R_augm;
        end
    end
end
