fprintf('\n\n\n---------- SARSA ------------\n\n\n');


L = SARSA([
    '$####';
    '.#X..';
    '.....']);

for i = 1:20
    L.sample();
end
