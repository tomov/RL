rooms = [ ...
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

close all; 

S = numel(rooms);

actions = {'right', 'down', 'left', 'up'};
dirs = {[0, 1], [1, 0], [0, -1], [-1, 0]};

A = numel(dirs);

T = zeros(S,S,A); % transitions (s,s') for each action a = 1..A
C = zeros(1,S); % clusters

for x = 1:size(rooms,1)
    for y = 1:size(rooms,2)
        if rooms(x,y) ~= '.'
            continue;
        end

        s = sub2ind(size(rooms), x, y);
        if clust(x,y) ~= '.'
            C(s) = num2str(clust(x,y));
        end
        for a = 1:A
            dir = dirs{a};
            x_new = x + dir(1);
            y_new = y + dir(2);
            if x_new < 1 || x_new > size(rooms, 1) || y_new < 1 || y_new > size(rooms, 2) || rooms(x_new, y_new) ~= '.'
                continue;
            end
            s_new = sub2ind(size(rooms), x_new, y_new);
            
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