fprintf('\n\n\n----------  LMDP ------------\n\n\n');


map = ['$####';
       '.#X..';
       '.....'];
%L = LMDP(map, find(ismember(map, '.$')));
L = LMDP(map);
L.solveLMDP();
L.sample();


%% MLMDP
%
fprintf('\n\n\n--------- MLMDP -----------\n\n\n');


M = MLMDP([
    '0####';
    '.#X..';
    '0...0']);
M.presolve();

M.solveMLMDP([10 -1 -1]');
M.sample();

M.solveMLMDP([-1 -1 10]');
M.sample();


%% augmented MLMDP
%
fprintf('\n\n\n--------- AMLMDP -----------\n\n\n');


A = AMLMDP([
    '0####';
    '.#X..';
    '0...0'], [2 6]);
A.presolve();

A.solveMLMDP([10 -1 -1 -1 -1]');
A.sample();

A.solveMLMDP([-1 -1 -1 -1 10]');
A.sample();


%% HMLMDP
%

fprintf('\n\n\n\n\n\n\n\n--------------------------------- HMLMDP ----------------------------------\n\n\n\n\n\n\n');

map = [
    '$####';
    'S#X..';
    '0.S.0'];
H = HMLMDP(map);
%H.sample();
H.sample_gui();


%% Full HMLMDP
%

fprintf('\n\n\n\n\n\n\n\n--------------------------------- Full HMLMDP ----------------------------------\n\n\n\n\n\n\n');

map = [
    '$####';
    '.#X..';
    '.....'];
H = HMLMDP(map, true);
H.sample();

H.plotZi();


%% Full HMLMDP + decomposed Zi
%

fprintf('\n\n\n\n\n\n\n\n--------------------------------- Full HMLMDP decomp ----------------------------------\n\n\n\n\n\n\n');

map = [
    '$####';
    '.#X..';
    '.....'];
H = HMLMDP(map, true, 2);
H.sample();

H.plotZi();


%% Big HMLMDP
%

fprintf('\n\n\n\n\n\n\n\n--------------------------------- big HMLMDP ----------------------------------\n\n\n\n\n\n\n');

map = [
    'SX...#.....';
    '.....#.....';
    '..#S.......';
    '.#...#.S##.';
    '.....#...0.';
    '#.####.....';
    '.....###.##';
    '..#..#...S.';
    '..#S.#.....';
    '.0.....S#..';
    '.....#..$..'];
H = HMLMDP(map);
%H.sample();
H.sample_gui();


%% Big full HMLMDP
%

fprintf('\n\n\n\n\n\n\n\n--------------------------------- big full HMLMDP ----------------------------------\n\n\n\n\n\n\n');

map = [
    '.X...#.....';
    '.....#.....';
    '..#........';
    '.#...#..##.';
    '.....#.....';
    '#.####.....';
    '.....###.##';
    '..#..#.....';
    '..#..#.....';
    '........#..';
    '.....#..$..'];
H = HMLMDP(map, true);
%H.sample();
H.sample_gui();

%H.plotZi();


%% Big full HMLMDP + decomposition of Zi
%

fprintf('\n\n\n\n\n\n\n\n--------------------------------- big full HMLMDP + decomp ----------------------------------\n\n\n\n\n\n\n');

close all;
map = [
    '.X...#.....';
    '.....#.....';
    '..#........';
    '.#...#..##.';
    '.....#.....';
    '#.####.....';
    '.....###.##';
    '..#..#.....';
    '..#..#.....';
    '........#..';
    '.....#..$..'];

map = [
    '.SX..#..S..';
    '...........';
    '##.#####.##';
    '.S...#..S..';
    '...........';
    '##.#####.##';
    '..S..#..S..';
    '.....#.....';
    '.######.###';
    '.S......S..';
    '.....#..$..'];

map = [
    '..X..#.....';
    '...........';
    '##.#####.##';
    '.....#.....';
    '...........';
    '##.#####.##';
    '.....#.....';
    '.....#.....';
    '.######.###';
    '...........';
    '.....#..$..'];

H = HMLMDP(map, true, 8);
%H.sample();
H.sample_gui();












%% TD(0) learning
%

fprintf('\n\n\n---------- TD(0) learning ------------\n\n\n');

map = [
    '$####';
    '.#X..';
    '.....'];
M = MDP(map);

for i = 1:20
    M.sampleSARSA(find(map == 'X'), true);
end
M.sampleSARSA_gui();

for i = 1:20
    M.sampleQ(find(map == 'X'), true);
end
M.sampleQ_gui();

for i = 1:20
    M.sampleAC(find(map == 'X'), true);
end
M.sampleAC_gui();

M.solveGPI();
M.sampleGPI();
M.sampleGPI_gui();


%% Options framework
%

fprintf('\n\n\n\n\n\n\n\n--------------------------------- Options ----------------------------------\n\n\n\n\n\n\n');

map = [
    '$####';
    'S#..X';
    '..S..'];
S = SMDP(map);
%S.sampleQ(find(map == 'X'), true);
S.sampleQ_gui();


%% Big options framework
%

fprintf('\n\n\n\n\n\n\n\n--------------------------------- big options framework ----------------------------------\n\n\n\n\n\n\n');

map = [
    'SX...#.....';
    '.....#.....';
    '..#S.......';
    '.#...#.S##.';
    '.....#...0.';
    '#.####.....';
    '.....###.##';
    '..#..#...S.';
    '..#S.#.....';
    '.0.....S#..';
    '.....#..$..'];
S = SMDP(map);
%O.sampleQ(find(map == 'X'), true);
%O.sampleQ();
S.sampleQ_gui();


%% HSM framework
%

fprintf('\n\n\n\n\n\n\n\n--------------------------------- HSM ----------------------------------\n\n\n\n\n\n\n');

map = [
    '$####';
    'S#..X';
    '..S..'];
S = SMDP(map, true);
%S.sampleQ(find(map == 'X'), true);
S.sampleQ_gui();




%% MAXQ framework
%

fprintf('\n\n\n\n\n\n\n\n--------------------------------- MAXQ ----------------------------------\n\n\n\n\n\n\n');

map = [
    '$####';
    'B#AAA';
    'BBAAA'];
%map = ['$A'; '##'];
M = MAXQ(map);
%M.maxQ0(3);
M.sample0_gui(11);
%M.sampleQ(find(map == 'X'), true);
%M.sampleQ_gui();


%% MAXQ framework larger
%

fprintf('\n\n\n\n\n\n\n\n--------------------------------- larger MAXQ ----------------------------------\n\n\n\n\n\n\n');

map = [
    'AAAAA#CCCCC';
    'AAAAACCCCCC';
    '##A#####C##';
    'BBBBB#HHHHH';
    'BBBBBBHHHHH';
    '##E#####D##';
    'EEEEE#DDDDD';
    'EEEEE#DDDDD';
    'E######D###';
    'FFFFFGGGGGG';
    'FFFFF#GG$GG'];
M = MAXQ(map);
%M.maxQ0(11);
M.sample0_gui(2);
%M.sampleQ(find(map == 'X'), true);
%M.sampleQ_gui();

