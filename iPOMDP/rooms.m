domain = [ ...
    '##################'; ...
    '#.......#........#'; ...
    '#................#'; ...
    '#.......#........#'; ...
    '##.######........#'; ...
    '#.......####.#####'; ...
    '#.......#........#'; ...
    '#................#'; ...
    '#.......#........#'; ...
    '##################'; ...
    ];

clust = [ ...
    '##################'; ...
    '#1111111#22222222#'; ...
    '#1111111222222222#'; ...
    '#1111111#22222222#'; ...
    '##1######22222222#'; ...
    '#3333333####4#####'; ...
    '#3333333#44444444#'; ...
    '#3333333344444444#'; ...
    '#3333333#44444444#'; ...
    '##################'; ...
    ];

% solway
domain = [...
    '...#...'; ...
    '.......'; ...
    '...#...'; ...
    ];
clust = [...
    '111#222'; ...
    '1113222'; ...
    '111#222'; ...
    ];


% t maze
domain = [...
    '.........'; ...
    '####.####'; ...
    '####.####'; ...
    '####.####'; ...
    '####.####'; ...
    '####.####'; ...
    '####.####'; ...
    '####.####'; ...
    '####.####'; ...
    ];
clust = [...
    '222213333'; ...
    '####1####'; ...
    '####1####'; ...
    '####1####'; ...
    '####1####'; ...
    '####1####'; ...
    '####1####'; ...
    '####1####'; ...
    '####1####'; ...
    ];


close all; 

S = numel(domain);

actions = {'right', 'down', 'left', 'up'};
dirs = {[0, 1], [1, 0], [0, -1], [-1, 0]};

A = numel(dirs);

T = zeros(S,S,A); % transitions (s,s') for each action a = 1..A
C = zeros(1,S); % clusters

for x = 1:size(domain,1)
    for y = 1:size(domain,2)
        if domain(x,y) ~= '.'
            continue;
        end

        s = sub2ind(size(domain), x, y);
        if clust(x,y) ~= '.'
            C(s) = num2str(clust(x,y));
        end
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



[~, ord] = sort(C);
ord = ord(C(ord) > 0);

figure;
for a = 1:A
    subplot(2,2,a);
    imagesc(T(ord,ord,a));
end


avg_T = mean(T, 3);
figure;
imagesc(avg_T(ord, ord));