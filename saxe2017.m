% rooms domain map from Saxe et al. 2017
%
rooms = [
    'SX...#.....';
    '.....#.....';
    '..#S.......';
    '.#...#.S##.';
    '.....#.....';
    '#.####.....';
    '.....###.##';
    '..#..#...S.';
    '..#..#.....';
    '..S....S#..';
    '.....#..$..'];

% single task for LMDP
%
minimap = [
    '$####';
    '.#X..';
    '.....'];

% basis set of tasks for MLMDP
%
minimaps = {
    ['1####';
     '.#X..';
     '0...0'];
     
    ['0####';
     '.#X..';
     '1...0'];
     
    ['0####';
     '.#X..';
     '0...1'];
};

% map from my 3D cannon.js maze
%
map = [
    '###################';
    '#.................#';
    '##.########...#.#.#';
    '#...#.....###.#...#';
    '###...#.#...#.##.##';
    '#.#.###.###.#.#...#';
    '#......X....#.#.#.#';
    '#.###########.#.#.#';
    '#..$.#..........#$#';
    '###################'];

% Test LMDP
%
L = createLMDP(minimap, 1);
L = solveLMDP(L);
sampleLMDP(L, L.a, minimap);
%sampleLMDP(L, L.P, minimap);

% Test multitask LMDP
%
M1 = createMLMDP(minimaps, 1);
M1 = solveMLMDP(M1);

% Test hierarchical multitask LMDP
%
M = createHMLMDP(rooms, 1);
%M{1} = solveMLMDP(M{1});
%M{2} = solveMLMDP(M{2});
fprintf('\n\n\n\n -------------------------- \n\n\n\n');
M = solveHMLMDP(M, rooms, 1);

% Create a hierarchical MLMDP from a maze
%
function M = createHMLMDP(map, lambda)
    % Create basis set of tasks from the reward locations,
    % where each task = go to a single reward location
    %
    maps = {};
    boundary = find(map ~= '#');
    original_map = map;
    % Make all passable squares boundary states (in addition to internal states)
    % Then create a task for each boundary state s.t. its reward = 0 and
    % all other boundary state have reward = -Inf
    %
    for g = boundary'
        map(boundary) = '-'; % all boundary squares are -Inf
        map(g) = '0'; % ...except the 'goal' boundary state for this task
        maps = [maps; {map}];
    end
    map = original_map; % restore map
    assert(~isempty(maps));
    
    % Create MLMDP from basis tasks based on reward locations only
    %
    M = createMLMDP(maps, lambda);
    
    % Augment it based on the subtask locations
    %
    subtasks = find(map == 'S');
    assert(~isempty(subtasks));
    M = augmentMLMDP(M, map, lambda);
    
    % Create second level of hierarchy
    %
    M2 = nextMLMDP(M, lambda);

    % Stack hierarchy
    %
    M = {M, M2};
end

% Take a low-level MLMDP initialized from a maze
% and construct the higher-level MLMDP
%
function M2 = nextMLMDP(M1, lambda)
    I = 1:numel(M1.St); % I of next level = St of lower level, but remapped to 1..Nt
    %B = numel(I) + 1 : numel(I) + size(M.Pb, 1); % B of next level = B of lower level, also remapped
    B = numel(I) + 1 : 2 * numel(I);
    S = [I B];
    Pi = M1.Pt * inv(eye(M1.Ni) - M1.Pi) * M1.Pt'; % I --> I from low-level dynamics
    %Pb = M.Pb * inv(eye(M.Ni) - M.Pi) * M.Pt';
    Pb = eye(numel(I)) * 0.05; % small prob I --> B #KNOB
    P = [Pi; Pb];
    P = P ./ sum(P, 1); % normalize
    P(isnan(P)) = 0;
    P = [P zeros(size(P, 1), numel(B))]; % prob B --> anything = 0

    % Define instantaneous rewards 
    %
    R = zeros(numel(S), 1);
    R(I) = -1; % time is money #KNOB
    R(B) = 1; % #KNOB
    q = exp(R / lambda);

    % Define subtasks Qb = eye(Nb)
    %
    R(B) = -Inf;
    Qb = [];
    for b = B
        R(b) = 0; % #KNOB
        q = exp(R / lambda);
        qb = q(B);
        Qb = [Qb, qb];
        R(b) = -Inf; % #KNOB
    end
   
    % create struct for level 2
    %
    M2.N = numel(S);
    M2.S = S;
    M2.Nb = numel(B);
    M2.B = B;
    M2.Ni = numel(I);
    M2.I = I;

    M2.P = P;
    M2.Pb = Pb;
    M2.Pi = Pi;

    M2.R = R;
    M2.qi = q(I);

    M2.lambda = lambda;

    M2.Qb = Qb;
end

% Augment a MLMDP with subtasks from a maze;
% helper function for building a hierarchical MLMDP
%
function M = augmentMLMDP(M, map, lambda)
    % Create helper MLMDP whose B corresponds to St
    %
    maps = {};
    goals = find(map == '$');
    subtasks = find(map == 'S');
    map(goals) = '.'; % erase all goal states
    for s = subtasks'
        map(subtasks) = '-'; % zero out all subtask states (but keep 'em as goals)
        map(s) = '0'; % ...except for one
        maps = [maps; {map}];
    end
    assert(~isempty(maps));
    
    Mt = createMLMDP(maps, lambda);
    
    % Augment M with subtask states
    %
    Qb = M.Qb;
    Qt = Mt.Qb;
    Nt = Mt.Nb; 
    St = M.N + 1 : M.N + Nt;

    % Augment state space and subtask space
    %
    M.St = St;
    M.Qt = Qt;
    M.Qb = [Qb zeros(M.Nb, numel(St)); zeros(numel(St), size(Qb, 2)) Qt]; % new Qb = [Qb 0; 0; Qt]
    
    M.Nt = Nt; % Note this is the number of subtasks NOT the number of tasks! i.e. = |St| NOT |B| = |old B| + |St|

    M.S = [M.S, M.St]; % new S = S union St
    M.N = M.N + Nt;

    M.B = [M.B, M.St]; % new B = B union St (!!!) technically wrong, but necessary to work with solveMLMDP
    M.Nb = M.Nb + Nt;
    
    % Find internal states that have corresponding subtask states (i.e.
    % boundary states in the helper Mt)
    % There should be 1 per boundary state (see createLMDP)
    %
    [x, y] = ind2sub(size(Mt.P(Mt.B,:)), find(Mt.P(Mt.B,:) ~= 0));
    assert(numel(x) == Mt.Nb); % by our design
    I_under_St = y';
    
    % Augment passive dynamics
    %
    P = [M.P zeros(size(M.P, 1), M.N - size(M.P, 2)); zeros(Nt, M.N)];
    ind = sub2ind(size(P), St, I_under_St); 
    P(ind) = 0.1; % P(subtask state | corresponing internal state) = s.th. small #KNOB
    P = P ./ sum(P, 1); % normalize
    P(isnan(P)) = 0; % fix the 0/0's
    M.P = P;
    M.Pt = M.P(M.St, M.I);
    M.Pb = M.P(M.B, M.I); % new B = B union St ! see above
    
    %assert(isequal(M.P(M.St, I_under_St), eye(Nt) * 0.05));
end


function M = solveHMLMDP(M, map, lambda)
    start = find(map == 'X');
    goal = find(map == '$');
    get_coords = @(s) ind2sub(size(map), s);
    
    % Solve MLMDPs for their basis set of tasks
    %
    M{1} = solveMLMDP(M{1});
    M{2} = solveMLMDP(M{2});
   
    % Set up starting state
    %
    s = start; % starting state; should be in I
    assert(ismember(s, M{1}.I));
    
    % Set up goal state
    %
    e = find(goal == M{1}.absorbing_squares); % ending state; should be in B 
    assert(~isempty(e));
    e = e + M{1}.Ni; % by design, that's the corresponding B state
    assert(ismember(e, M{1}.B));
    
    % Find basis task weights based on goal state
    %
    r = (-Inf) * ones(M{1}.N, 1); % rewards
    r(e) = 0; % #KNOB
    r(M{1}.St) = -10; % #KNOB TODO ?????
    q = exp(r / lambda);
    q(M{1}.I) = M{1}.qi;
    qb = q(M{1}.B); % boundary states only
    w = pinv(M{1}.Qb) * qb; % Eq 7 from Saxe et al 2017
    
    % Find the desirabiltiy f'n and actions based on the basis task weights
    % Sec 2.3 from Saxe et al (2017)
    %
    zi = M{1}.Zi * w;
    z(M{1}.I) = zi;
    z(M{1}.B) = qb;
    a = policy(M{1}.P, z');
    
    % Set up actual rewards
    %
    R_tot = 0; % total rewards
    R = M{1}.R(M{1}.I); % rewards for 'real' states = internal states = the squares in the grid = -1
    R(goal) = 20; % #KNOB
    
    % Solve the hierarchical multitask LMDP
    % Algorithm 1 from Saxe et al (2017) supplement
    %
    iter = 1;
    while true
        [x, y] = get_coords(s);
        
        R_tot = R_tot + R(s);
        
        new_s = samplePF(a(:,s));
        
        if ismember(new_s, M{1}.I) 
            % Internal state -> just move to it
            %
            [new_x, new_y] = get_coords(new_s);
            
            fprintf('(%d, %d) --> (%d, %d)\n', x, y, new_x, new_y);
            map(s) = '.';
            map(new_s) = 'X';
            disp(map);
            
            s = new_s;
            
        elseif ismember(new_s, M{1}.St)
            % Higher layer state i.e. subtask state
            %
            s2 = find(new_s == M{1}.St); % mapping from St to I of next level; by design
            
            fprintf('NEXT LEVEL! (%d, %d) --> %d !!!\n', x, y, s2);
            
            rb2 = [1 2 3 4 5 6]' - 10; % TODO HARDCODED FIXME
            qb2 = exp(rb2 / lambda);
            w2 = pinv(M{2}.Qb) * qb2; % Eq 7 from Saxe et al 2017

            % Find the desirabiltiy f'n and actions based on the basis task weights
            % Sec 2.3 from Saxe et al (2017)
            %
            zi2 = M{2}.Zi * w2;
            z2(M{2}.I) = zi2;
            z2(M{2}.B) = qb2;
            a2 = policy(M{2}.P, z2');
                        
            while true
                new_s2 = samplePF(a2(:, s2));
                
                if ismember(new_s2, M{2}.I)
                    fprintf('         %d --> %d\n', s2, new_s2);
                    
                    s2 = new_s2;
                else
                    fprintf('         %d --> BOUNDARY --> back to lower level\n', s2);
                    
                    % Boundary state => recalculate boundary rewards and actions at
                    % lower level
                    %
                    % # KNOB
                    rt = 0.5 * (a(M{2}.I, s2) - M{2}.Pi(:, s2)); % Eq 10 -- notce it's s2 and not new_s2 b/c new_s2 is in B and has nothing coming out of it...
                    
                    % Sec 2.3
                    r(M{1}.St) = rt;
                    q = exp(r / lambda);
                    q(M{1}.I) = M{1}.qi;
                    qb = q(M{1}.B); % boundary states only
                    w = pinv(M{1}.Qb) * qb; % Eq 7 from Saxe et al 2017
                    
                    zi = M{1}.Zi * w;
                    z(M{1}.I) = zi;
                    z(M{1}.B) = qb;
                    a = policy(M{1}.P, z');
                    
                    save shit.mat
                    
                    break;
                end
            end
            
        else % NOTE: B includes St too! which is why we check it last
            % Terminal boundary state
            %
            fprintf('(%d, %d) --> END\n', x, y);
            break;
        end
        
        
        iter = iter + 1;
        if iter >= 20, break; end
    end
    
    fprintf('Total reward: %d\n', R_tot);
end

%
% ---------------------------------- Multitask LMDPs -------------------------------
%



% Create a MLMDP from multiple mazes;
% assumes mazes represent a valid basis task set
%
function M = createMLMDP(maps, lambda)
    M = [];
    
    for i = 1:numel(maps)
        map = maps{i};
        L = createLMDP(map, lambda);
    
        if isempty(M)
            M = L;
            M.Qb = L.qb;
        else
            assert(isequal(L.S, M.S));
            assert(isequal(L.P, M.P));
            assert(isequal(L.Pi, M.Pi));
            assert(isequal(L.Pb, M.Pb));
            assert(isequal(L.I, M.I));
            assert(isequal(L.B, M.B));
            assert(isequal(L.qi, M.qi));
            
            M.Qb = [M.Qb, L.qb];
        end
    end
    
    assert(size(M.Qb, 2) == numel(maps));
end

% Solve an initialized MLMDP
%
function M = solveMLMDP(M)
    Zi = [];
    a = [];
    for i = 1:size(M.Qb,2) % for each subtask
        M.qb = M.Qb(:,i); % subtask i
        L = solveLMDP(M);
        
        if isempty(Zi)
            Zi = L.z(L.I);
            a = L.a;
        else
            Zi = [Zi, L.z(L.I)];
            a(:,:,i) = L.a;
        end
    end    
    assert(size(Zi, 2) == size(M.Qb, 2));
    assert(size(Zi, 1) == M.Ni);
    
    M.Zi = Zi;
    M.a = a;
end


%
% ---------------------------------- LMDPs -------------------------------
%



% Initialize an unsolved LMDP from a maze
%
function L = createLMDP(map, lambda)
    state = @(x, y) sub2ind(size(map), x, y);
    
    absorbing = '-0123456789$'; % squares that correspond to boundary states
    agent = 'X'; % the starting square
    passable = ['.', agent, absorbing]; % squares that are correspond to passable / allowed states
    
    absorbing_squares = find(ismember(map, absorbing));
    
    Ni = numel(map); % internal states = all squares, including walls (useful for (x,y) --> state)
    Nb = numel(absorbing_squares); % boundary states
    N = Ni + Nb;
    
    S = 1 : N; % set of states {s}
    I = 1 : Ni; % set of internal states
    B = Ni + 1 : Ni + Nb; % set of boundary states; they follow I in indexing, for convenience
    P = zeros(N, N); % passive transitions P(s'|s)
    R = zeros(N, 1); % instantaneous reward f'n R(s)
    start = nan; % the starting state
    
    % adjacency list
    % each row = [dx, dy, non-normalized P(s'|s)]
    % => random walk, but also bias towards staying in 1 place
    %
    adj = [0, 0, 2; ...
        -1, 0, 1; ...
        0, -1, 1; ...
        1, 0, 1; ...
        0, 1, 1];
    % small probability to go from an internal state to the boundary state
    % for the corresponding square
    %
    adj_I_to_B = 0.1; % #KNOB

    % iterate over all internal states s
    %
    for x = 1:size(map, 1)
        for y = 1:size(map, 2)
            s = state(x, y);
            %fprintf('(%d, %d) --> %d = ''%c''\n', x, y, s, map(x, y));
            assert(ismember(s, S));
            assert(ismember(s, I));
            
            R(s) = -1; % time is money for internal states

            % Check if there's a corresponding boundary state
            %
            b = find(s == absorbing_squares);
            if ~isempty(b)
                % There's also a boundary state in this square
                %
                assert(ismember(map(x, y), absorbing))
                
                b = b + Ni; % get the actual boundary state
                P(b, s) = adj_I_to_B; % go to corresponding boundary state w/ small prob
                
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

            if ismember(map(x, y), agent)
                % Starting state (for convenience; not really necessary)
                %
                start = s;
            end
            
            if ~ismember(map(x, y), passable)
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
                if ~ismember(map(new_x, new_y), passable)
                    continue; % impassable neighbor state s'
                end
                
                new_s = state(new_x, new_y);
                %fprintf('      (%d, %d) --> %d = ''%c''\n', new_x, new_y, new_s, map(new_x, new_y));
                assert(ismember(new_s, S));
                assert(ismember(new_s, I));
                    
                % passive transition f'n P(new_s|s)
                % will normalize later
                %
                P(new_s, s) = adj(i, 3);
            end
            
            P(:, s) = P(:, s) / sum(P(:, s)); % normalize P(.|s)
        end
    end
    
    assert(isequal(I, setdiff(S, B)));
    q = exp(R / lambda); % exponentiated reward f'n
    
    % return LMDP
    %
    L.N = N;
    L.S = S;
    L.Nb = Nb;
    L.B = B;
    L.Ni = Ni;
    L.I = I;
    assert(Ni == numel(I));
    assert(Nb == numel(B));
    assert(N == numel(S));
    
    L.P = P;
    L.Pb = P(B, I); % P(s'|s) for s' in B and s in I
    L.Pi = P(I, I); % P(s'|s) for s' and s in I
    
    L.R = R;
    L.q = q;
    L.qb = q(B); % q(s) for s in B
    L.qi = q(I); % q(s) for s in I
    
    L.start = start;
    L.lambda = lambda;
    L.absorbing_squares = absorbing_squares;
    
    sanityLMDP(L, map);
end


% Solve an initialized LMDP
%
function L = solveLMDP(L)
    Mi = diag(L.qi);
    zb = L.qb;
    Pi = L.Pi;
    Pb = L.Pb;
    P = L.P;
    N = L.N;
    
    % find desirability f'n z
    %
    z = zeros(N, 1);
    zi = inv(eye(L.Ni) - Mi * Pi') * (Mi * Pb' * zb); % Eq 4 from Saxe et al (2017)
    z(L.I) = zi;
    z(L.B) = zb;
        
    % find optimal policy a*
    %
    a = policy(P, z);
    
    L.z = z;
    L.a = a;
end

% Sanity check that a LMDP is correct
%
function sanityLMDP(L, map)
    % States
    %
    assert(L.N == size(L.S, 2));
    assert(size(L.S, 1) == 1);
    assert(L.Ni == size(L.I, 2));
    assert(size(L.I, 1) == 1);
    assert(L.Nb == size(L.B, 2));
    assert(size(L.B, 1) == 1);
    
    % States <--> maze -- these are our custom things designed for the task
    %
    assert(isequal(L.I, 1:numel(map)));
    assert(max(L.I) < min(L.B));
    assert(numel(L.absorbing_squares) == L.Nb);
    assert(sum(ismember(L.absorbing_squares, L.I)) == L.Nb);
    
    % Transition dynamics
    %
    assert(size(L.P, 1) == L.N);
    assert(size(L.P, 2) == L.N);
    assert(size(L.Pi, 1) == L.Ni);
	assert(size(L.Pi, 2) == L.Ni);
    assert(size(L.Pb, 1) == L.Nb);
	assert(size(L.Pb, 2) == L.Ni);
    assert(isequal(L.P(L.I, L.I), L.Pi));
    assert(isequal(L.P(L.B, L.I), L.Pb));
    
    % Rewards
    %
    assert(size(L.R, 1) == L.N);
    assert(size(L.R, 2) == 1);
    assert(size(L.q, 1) == L.N);
    assert(size(L.q, 2) == 1);
    assert(size(L.qi, 1) == L.Ni);
    assert(size(L.qi, 2) == 1);
    assert(size(L.qb, 1) == L.Nb);
    assert(size(L.qb, 2) == 1);
    assert(isequal(L.q(L.I), L.qi));
    assert(isequal(L.q(L.B), L.qb));
end

% Compute an optimal policy a*(s',s) from passive transition dynamics P(s'|s)
% and a desirability f'n z(s)
%
function a = policy(P, z)
    assert(size(z, 2) == 1);
    
    N = size(z, 1);
    a = zeros(N, N);
    G = @(s) sum(P(:,s) .* z);
    for s = 1:N
        if G(s) == 0
            continue;
        end
        a(:,s) = P(:,s) .* z / G(s); % Eq 6 from Saxe et al (2017)
    end
end

% sample paths from a solved LMDP
%
function sampleLMDP(L, a, map)
    s = L.start;
    r = 0;
    
    get_coords = @(s) ind2sub(size(map), s);
    
    agent = 'X';
    empty = '.';
    
    disp(map);
    while true
        r = r + L.R(s);
        [x, y] = get_coords(s);
        
        new_s = samplePF(a(:,s));        
        
        if ismember(new_s, L.B)
            % Boundary state
            %
            fprintf('(%d, %d) --> END\n', x, y);
            r = r + L.R(new_s);
            break;
        end

        % Internal state
        %
        [new_x, new_y] = get_coords(new_s);
        
        map(x, y) = empty;
        map(new_x, new_y) = agent;
        
        fprintf('(%d, %d) --> (%d, %d)\n', x, y, new_x, new_y);
        disp(map);
        
        s = new_s;        
    end
    fprintf('Total reward: %d\n', r);
end

