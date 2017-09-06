next = containers.Map;
next('A') = {'B', 'C', 'D'};
next('C') = {'E', 'F', 'G'};
next('F') = {'H', 'I', 'J'};
next('I') = {'K', 'L', 'M'};
next('L') = {'N', 'O', 'P'};

next('B') = {'b1', 'b2', 'b3'};
next('D') = {'d1', 'd2', 'd3'};
next('E') = {'e1', 'e2', 'e3'};
next('G') = {'g1', 'g2', 'g3'};
next('H') = {'h1', 'h2', 'h3'};
next('J') = {'j1', 'j2', 'j3'};
next('K') = {'k1', 'k2', 'k3'};
next('M') = {'m1', 'm2', 'm3'};
next('N') = {'n1', 'n2', 'n3'};
next('P') = {'p1', 'p2', 'p3'};

next('b1') = {'b11', 'b12', 'b13'};
next('b2') = {'b21', 'b22', 'b23'};
next('b3') = {'b31', 'b32', 'b33'};
next('d1') = {'d11', 'd12', 'd13'};
next('d2') = {'d21', 'd22', 'd23'};
next('d3') = {'d31', 'd32', 'd33'};

rewards = containers.Map;
rewards('F') = 10;

terminal = {'E', 'F', 'G', 'b1', 'b2', 'b3', 'd1', 'd2', 'd3'};

start = 'A';

%%
%
close all;

M = MDP_dag(keys(next), values(next), rewards, terminal);
%M.sampleAC_gui('A'); 

rs = [];
for i = 1:20
    [Rtot, path] = M.sampleAC(M.get_state_by_name(start));
    rs = [rs, Rtot];
end

figure;
subplot(1, 2, 1);
plot(rs);
subplot(1, 2, 2);
M.plot_DAG();

%M.add_terminal({'B', 'C'});
%M.remove_terminal({'B'});
%M.sampleAC_gui(start);

M.remove_terminal({'E', 'F', 'G', 'b1', 'b2', 'b3', 'd1', 'd2', 'd3'});
M.add_terminal({'H', 'I', 'J'});

M.R(M.get_state_by_name('F')) = 0;
M.R(M.get_state_by_name('I')) = 10;

%rs = [];
for i = 1:20
    [Rtot, path] = M.sampleAC(M.get_state_by_name(start));
    rs = [rs, Rtot];
end

figure;
subplot(1, 2, 1);
plot(rs);
subplot(1, 2, 2);
M.plot_DAG();

%%
%
S = SMDP();
S.init_from_dag(keys(next), values(next), rewards, []);

for i = 1:20
    [Rtot, path] = S.sampleAC(S.mdp.get_state_by_name('A'));
    disp(Rtot)
end

%S.sampleAC_gui(S.mdp.get_state_by_name('A'));
