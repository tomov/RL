next = containers.Map;
next('A') = {'B', 'C', 'D'};
next('B') = {'0', 'E', '0'};
next('C') = {'0', 'F', '0'};
next('D') = {'0', 'G', '0'};

rewards = containers.Map;
rewards('A') = -1;
rewards('B') = -1;
rewards('C') = -1;
rewards('D') = -1;
rewards('E') = -1;
rewards('F') = -1;
rewards('G') = 10;
rewards('0') = 0;

%%
%
M = MDP_dag(keys(next), values(next), rewards);
M.sampleAC_gui('A');


%%
%
S = SMDP();
S.init_from_dag(keys(next), values(next), rewards, {'D', 'C'});

figure;
S.mdp.plot();

S.sampleAC_gui(S.mdp.get_state_by_name('A'));