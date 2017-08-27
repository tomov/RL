% LMDP as described in Saxe et al (2017) and Todorov (2009).
% Off-policy, model-based approach.
% Customized to work with the 'rooms' domain only.
%
% Given states S (split into internal states I and boundary states B),
% boundary rewards R(s) for s in B, and passive transitions P(s'|s).
%
% Finds a set of active transitions a(s'|s) that maximize the total expected reward (before reaching a B state),
% without deviating too much from the passive dynamics P(s'|s) (controlled by lambda)
%
% First computes the exponentiated rewards q(s) = exp(R(s)/lambda).
% Then finds the (optimal) desirability function z(s) = exp(V(s)/lambda),
% where V(s) is the cost-to-go-function  i.e. the expected reward of starting at state s
% and acting optimally thereafter.
% Notice that V(s) = R(s) => z(s) = q(s) for s in B, so only zi is interesting (i.e. z(s) for
% s in I). This is computed directly by matrix inversion or by Z-iteration.
% Finally, computes the active transitions a(s'|s) directly from z(s)
%
classdef LMDP < handle

    properties (Constant = true)
        % General LMDP
        % 
        lambda = 1; % how much to penalize deviations from passive dynamics P(s'|s)
        R_I = -1; % penalty for staying in one place

        % Maze
        %
        absorbing_symbols = '-0123456789$';
        agent_symbol = 'X';
        empty_symbol = '.';
        impassable_symbols = '#';

        % Unnormalized transition probabilities
        %
        uP_I_to_self = 2; % unnormalized P(s|s)
        uP_I_to_neighbor = 1; % unnormalized random walk
        P_I_to_B = 0.05; % (normalized) move from I state to corresponding B state
    end

    properties (Access = public)
        % General LMDP stuff
        %
        S = []; % all states
        I = []; % interior states
        B = []; % boundary states
        R = []; % R(s) = instantaneous reward at state s

        q = []; % q(s) = exp(R(s)/lambda) = exponentiated reward at state s
        z = []; % z(s) = exp(V(s)/lambda) = desirability function at state s
        P = []; % P(s'|s) = passive transition probability from s to s'
        a = []; % a(s'|s) = active transition probability from s to s'

        I2B = []; % I2B(s) = corresponding B state for given I state s, or 0 if none
        B2I = []; % B2I(s) = corresponding I state for given B state s

        % Maze stuff
        %
        map = [];
    end

    methods

        % Initialize a LMDP from a maze
        %
        function self = LMDP(map)
            self.map = map; % so we can use pos2I

            absorbing_inds = find(ismember(map, self.absorbing_symbols)); % goal squares = internal states with corresponding boundary states
            I_with_B = absorbing_inds;
            Ni = numel(map); % internal states = all squares, including walls (useful for (x,y) --> state)
            Nb = numel(I_with_B); % boundary states
            N = Ni + Nb;
            
            S = 1 : N; % set of states
            I = 1 : Ni; % set of internal states
            B = Ni + 1 : Ni + Nb; % set of boundary states
            self.S = S;
            self.I = I;
            self.B = B;

            I2B = zeros(N, 1);
            B2I = zeros(N, 1);
            I2B(I_with_B) = B; % mapping from I states to corresponding B states
            B2I(I2B(I2B > 0)) = I_with_B; % mapping from B states to corresponding I states
            self.I2B = I2B;
            self.B2I = B2I;

            P = zeros(N, N); % passive transitions P(s'|s); defaults to 0
            R = nan(N, 1); % instantaneous reward f'n R(s)
            
            % adjacency list
            % each row = [dx, dy, non-normalized P(s'|s)]
            % => random walk, but also bias towards staying in 1 place
            %
            adj = [0, 0, self.uP_I_to_self; ...
                -1, 0, self.uP_I_to_neighbor; ...
                0, -1, self.uP_I_to_neighbor; ...
                1, 0, self.uP_I_to_neighbor; ...
                0, 1, self.uP_I_to_neighbor];

            % iterate over all internal states s
            %
            for x = 1:size(map, 1)
                for y = 1:size(map, 2)
                    s = self.pos2I(x, y);
                    %fprintf('(%d, %d) --> %d = ''%c''\n', x, y, s, map(x, y));
                    assert(ismember(s, S));
                    assert(ismember(s, I));
                    
                    R(s) = self.R_I; % time is money for internal states

                    if ismember(map(x, y), self.impassable_symbols)
                        % Impassable state e.g. wall -> no neighbors
                        %
                        continue;
                    end
                    
                    % Iterate over all adjacent states s --> s' and update passive
                    % dynamics P
                    %
                    for i = 1:size(adj, 1)
                        new_x = x + adj(i, 1);
                        new_y = y + adj(i, 2);
                        if new_x <= 0 || new_x > size(map, 1) || new_y <= 0 || new_y > size(map, 2)
                            continue % outside the map
                        end
                        if ismember(map(new_x, new_y), self.impassable_symbols)
                            continue; % impassable neighbor state s'
                        end
                        
                        new_s = self.pos2I(new_x, new_y);
                        %fprintf('      (%d, %d) --> %d = ''%c''\n', new_x, new_y, new_s, map(new_x, new_y));
                        assert(ismember(new_s, S));
                        assert(ismember(new_s, I));
                            
                        % passive transition f'n P(new_s|s)
                        % will normalize later
                        %
                        P(new_s, s) = adj(i, 3);
                    end
                  
                    P(:, s) = P(:, s) / sum(P(:, s)); % normalize P(.|s)

                    % Check if there's a corresponding boundary state
                    %
                    if I2B(s)
                        % There's also a boundary state in this square
                        %
                        assert(ismember(s, absorbing_inds));
                        assert(ismember(map(x, y), self.absorbing_symbols));

                        b = I2B(s);

                        % Adjust the probabilities -- we want the probability of going to a B state to be uniform across all I states
                        %
                        P(:, s) = P(:, s) * (1 - self.P_I_to_B);
                        P(b, s) = self.P_I_to_B; % go to corresponding boundary state w/ small prob
                        
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
                end
            end
            
            assert(isequal(I, setdiff(S, B)));
            q = exp(R / self.lambda); % exponentiated reward f'n
            
            self.P = P;
            self.R = R;
            self.q = q;

            self.sanityLMDP();
        end


        % Solve an initialized LMDP
        %
        function solveLMDP(self)
            qi = self.q(self.I);
            qb = self.q(self.B);
            Pi = self.P(self.I, self.I);
            Pb = self.P(self.B, self.I);
            N = numel(self.S);
            Ni = numel(self.I);

            Mi = diag(qi);

            % find desirability f'n z
            %
            z = nan(N, 1);
            zb = qb; % boundary states are absorbing -> V(s) = R(s) for s in B

            % learn the whole thing in one go
            %
            zi = inv(eye(Ni) - Mi * Pi') * (Mi * Pb' * zb); % Eq 4 from Saxe et al (2017)

            % Z-learning
            %
            %zi = zeros(Ni,1);
            %for i = 1:30
            %    zi = Mi * Pi' * zi + Mi * Pb' * zb;
            %end

            z(self.I) = zi;
            z(self.B) = zb;
            self.z = z;
                
            % find optimal policy a*
            %
            a = self.policy(z);
            self.a = a;

            self.sanityLMDP();
        end

        % Compute an optimal policy a*(s',s) from passive transition dynamics P(s'|s)
        % and a desirability f'n z(s)
        %
        function a = policy(self, z)
            N = numel(self.S);
            assert(size(z, 2) == 1);
            assert(size(z, 1) == N);
            
            a = zeros(N, N);
            G = @(s) sum(self.P(:,s) .* z);
            for s = 1:N
                if G(s) == 0
                    continue;
                end
                a(:,s) = self.P(:,s) .* z / G(s); % Eq 6 from Saxe et al (2017)
            end
        end

        % sample paths from a solved LMDP
        % optionally accepts a starting state; otherwise, uses the X from the map
        %
        function [Rtot, path] = sample(self, s)
            if ~exist('s', 'var')
                s = find(self.map == self.agent_symbol);
            end
            assert(numel(find(self.I == s)) == 1);

            Rtot = 0;
            path = [];
            
            map = self.map;
            disp(map);
            while true
                Rtot = Rtot + self.R(s);
                path = [path, s];

                [x, y] = self.I2pos(s);
                
                new_s = samplePF(self.a(:,s));        
                
                if ismember(new_s, self.B)
                    % Boundary state
                    %
                    fprintf('(%d, %d) [%.2f%%] --> END\n', x, y, self.a(new_s, s) * 100);

                    Rtot = Rtot + self.R(new_s);
                    path = [path, new_s];
                    break;
                end

                % Internal state
                %
                [new_x, new_y] = self.I2pos(new_s);
                
                map(x, y) = self.empty_symbol;
                map(new_x, new_y) = self.agent_symbol;
                
                fprintf('(%d, %d) --> (%d, %d) [%.2f%%]\n', x, y, new_x, new_y, self.a(new_s, s) * 100);
                disp(map);
                
                s = new_s;
            end
            fprintf('Total reward: %d\n', Rtot);
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
        

        % Sanity check that a LMDP is correct
        %
        function sanityLMDP(self)
            % States
            %
            N = numel(self.S);
            assert(N == size(self.S, 2));
            assert(size(self.S, 1) == 1);
            assert(sum(self.I2B > 0) == numel(self.B));
            assert(sum(self.B2I > 0) == numel(self.B));
            
            % States <--> maze -- these are our custom things designed for the task
            %
            assert(isequal(self.I, 1:numel(self.map)));
            assert(max(self.I) < min(self.B));
            
            % Transition dynamics
            %
            assert(size(self.P, 1) == N);
            assert(size(self.P, 2) == N);
            assert(sum(abs(sum(self.P, 1) - 1) < 1e-8 | abs(sum(self.P, 1)) < 1e-8) == N);
            
            % Rewards
            %
            assert(size(self.R, 1) == N);
            assert(size(self.R, 2) == 1);
            assert(size(self.q, 1) == N);
            assert(size(self.q, 2) == 1);

            % Solution
            %
            assert(isempty(self.z) || size(self.z, 1) == N);
            assert(isempty(self.z) || size(self.z, 2) == 1);
            assert(isempty(self.a) || size(self.a, 1) == N);
            assert(isempty(self.a) || size(self.a, 2) == N);
        end


    end
end
