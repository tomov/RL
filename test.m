
map = [
    '............#...#.';
    '........#.#.#.#.#.';
    '........#.#.#.#.#.';
    '........#.#.#.#.#.';
    '........#.#...#...';
    '.#####.####.#####.';
    '.$..........#.....';
    '..#..#......#.###.';
    '.###.#..........#.';
    '..#..#......#.###.';
    '.....#......#.....';
    '.#####.####.#####.';
    '.#...#...#........';
    '...#...#.#..#####.';
    '.#...#...#....#...';
    '...#...#.....X....'];


%% TD(0) learning
%
close all;

fprintf('\n\n\n---------- TD(0) learning ------------\n\n\n');

%{
map = [
    '$####';
    '.#..X';
    '.....'];
%}

%{
M = MDP(map);
rs_0 = [];
for i = 1:100
    [r, path] = M.sampleAC(find(map == 'X'), false);
    disp(r);
    rs_0 = [rs_0, r];
end
%}

M = MDP(map, 0.9);
rs_lambda = [];
for i = 1:100
    [r, path] = M.sampleAC(find(map == 'X'), false);
    disp(r);
    rs_lambda = [rs_lambda, r];
end


figure;
plot(R_0);
hold on;
plot(R_lambda);
plot(rs_lambda);  
hold off;
legend({'TD(0)', 'TD(\lambda)', 'TD(\lambda) ++'});


%%
%

M = MDP(map, 0.9);
M.sampleAC_gui();


%% Options framework
%

fprintf('\n\n\n\n\n\n\n\n--------------------------------- Options ----------------------------------\n\n\n\n\n\n\n');


S = SMDP(map);
%S.sampleQ(find(map == 'X'), true);
S.sampleQ_gui();


%% HSM framework
%

fprintf('\n\n\n\n\n\n\n\n--------------------------------- HSM ----------------------------------\n\n\n\n\n\n\n');

S = SMDP(map, true);
%S.sampleQ(find(map == 'X'), true);
S.sampleQ_gui();


%% HMLMDP
%


H = HMLMDP(map, true, 9);
%H.sample();
H.sample_gui();


