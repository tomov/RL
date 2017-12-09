domain = [ ...
    'A.#..'; ...
    '#.D..'; ...
    '.....'; ...
    '.#.C#'; ...
    '.#.B.'; ...
    ];


pickup = 'D';
dropoff = 'B';

close all; 

S = numel(domain);

actions = {'right', 'down', 'left', 'up'};
dirs = {[0, 1], [1, 0], [0, -1], [-1, 0]};

A = numel(dirs);

T = zeros(S,S,A); % transitions (s,s') for each action a = 1..A

for x = 1:size(domain,1)
    for y = 1:size(domain,2)
        if domain(x,y) == '#'
            continue;
        end

        s = sub2ind(size(domain), x, y);
        for a = 1:A
            dir = dirs{a};
            x_new = x + dir(1);
            y_new = y + dir(2);
            if x_new < 1 || x_new > size(domain, 1) || y_new < 1 || y_new > size(domain, 2) || domain(x_new, y_new) ~= '.'
                continue;
            end
            s_new = sub2ind(size(domain), x_new, y_new);
            
            T(s, s_new, a) = 1;
        end
    end
end


TT = zeros(S*2, S*2, A + 2);
TT(1:S, 1:S, 1:A) = T;
TT(S+1:2*S, S+1:2*S, 1:A) = T;

s_pickup = find(domain == pickup);
s_dropoff = find(domain == dropoff);

% pickup action
TT(s_pickup, s_pickup + S, A + 1) = 1;
% dropoff action
TT(s_dropoff + S, s_dropoff, A + 2) = 1;


avg_TT = mean(TT, 3);
figure;
imagesc(avg_TT(:, :));