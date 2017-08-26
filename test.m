fprintf('\n\n\n---------- SARSA ------------\n\n\n');


L = TD([
    '$####';
    '.#X..';
    '.....']);

%for i = 1:20
%    L.sampleSARSA();
%end

for i = 1:20
    L.sampleQ();
end
