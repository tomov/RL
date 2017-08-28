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
    '..#..#.....';
    '.0S....S#..';
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
H = HMLMDP(map, true, 5);
%H.sample();
H.sample_gui();



%% TD(0) learning
%

fprintf('\n\n\n---------- TD(0) learning ------------\n\n\n');


T = TD([
    '$####';
    '.#X..';
    '.....']);

%for i = 1:20
%    T.sampleSARSA();
%end

%for i = 1:20
%    T.sampleQ();
%end

%for i = 1:20
%    T.sampleAC();
%end

T.solveGPI();
T.sampleGPI();

%% Options framework
%

fprintf('\n\n\n\n\n\n\n\n--------------------------------- Options ----------------------------------\n\n\n\n\n\n\n');

map = [
    '$####';
    'S#X..';
    '0.S.0'];
O = Options(map);
O.sampleQ();

