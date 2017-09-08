next = containers.Map;
next('12') = {'1', '2'};
M = MDP_dag(keys(next), values(next), containers.Map, {});
figure;
subplot(2,2,1);
path = [M.get_state_by_name('12'), M.get_state_by_name('2')];
h = M.plot_DAG(path);
%labeledge(h, '12', '1', 'left');



next = containers.Map;
next('34') = {'3', '4'};
M = MDP_dag(keys(next), values(next), containers.Map, {});
subplot(2,2,2);
path = [M.get_state_by_name('34'), M.get_state_by_name('3')];
h = M.plot_DAG(path);
%labeledge(h, '34', '4', 'right');



next = containers.Map;
next('21') = {'2', '1'};
M = MDP_dag(keys(next), values(next), containers.Map, {});
subplot(2,2,3);
path = [M.get_state_by_name('21'), M.get_state_by_name('2')];
h = M.plot_DAG(path);
%labeledge(h, '12', '1', 'left');


next = containers.Map;
next('43') = {'4', '3'};
M = MDP_dag(keys(next), values(next), containers.Map, {});
subplot(2,2,4);
path = [M.get_state_by_name('43'), M.get_state_by_name('3')];
h = M.plot_DAG(path);
%labeledge(h, '34', '4', 'right');



