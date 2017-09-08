next = containers.Map;

next('AB') = {'CD', 'EF'};

next('CD') = {'GH', 'IJ'};
next('EF') = {'KL', 'MN'};

next('GH') = {'g', 'h'};
next('IJ') = {'i', 'j'};
next('KL') = {'k', 'l'};
next('MN') = {'m', 'n'};

rewards = containers.Map;
rewards('k') = 10;

terminal = {};

start = 'AB';

M = MDP_dag(keys(next), values(next), rewards, terminal);
%M.sampleAC_gui('A'); 

close all;
figure;


subplot(2,2,1);
path = [ ...
    M.get_state_by_name('AB'), ...
    M.get_state_by_name('EF'), ...
    M.get_state_by_name('KL'), ...
    M.get_state_by_name('k'), ...
    ];
h = M.plot_DAG(path);
labeledge(h, 'AB', 'EF', 'right');
labeledge(h, 'EF', 'KL', 'left');
labeledge(h, 'KL', 'k', 'left');


subplot(2,2,2);
path = [ ...
    M.get_state_by_name('AB'), ...
    M.get_state_by_name('EF'), ...
    M.get_state_by_name('KL'), ...
    M.get_state_by_name('k'), ...
    ];
h = M.plot_DAG(path);
labeledge(h, 'AB', 'EF', 'b');
labeledge(h, 'EF', 'KL', 'e');
labeledge(h, 'KL', 'k', 'k');


%% reversed
%


next = containers.Map;

next('AB') = {'CD', 'FE'};

next('CD') = {'GH', 'IJ'};
next('FE') = {'MN', 'LK'};

next('GH') = {'g', 'h'};
next('IJ') = {'i', 'j'};
next('LK') = {'l', 'k'};
next('MN') = {'m', 'n'};

rewards = containers.Map;
rewards('k') = 10;

terminal = {};

start = 'AB';

M = MDP_dag(keys(next), values(next), rewards, terminal);
%M.sampleAC_gui('A'); 

subplot(2,2,3);
path = [ ...
    M.get_state_by_name('AB'), ...
    M.get_state_by_name('FE'), ...
    M.get_state_by_name('MN'), ...
    M.get_state_by_name('m'), ...
    ];
h = M.plot_DAG(path);
labeledge(h, 'AB', 'FE', 'right');
labeledge(h, 'FE', 'MN', 'left');
labeledge(h, 'MN', 'm', 'left');


subplot(2,2,4);
path = [ ...
    M.get_state_by_name('AB'), ...
    M.get_state_by_name('FE'), ...
    M.get_state_by_name('LK'), ...
    M.get_state_by_name('k'), ...
    ];
h = M.plot_DAG(path);
labeledge(h, 'AB', 'FE', 'b');
labeledge(h, 'FE', 'LK', 'e');
labeledge(h, 'LK', 'k', 'k');


