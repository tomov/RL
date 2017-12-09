S = 10;

T = zeros(S,S);

for s = 1:9
    T(s,s+1) = 1;
end
T(S,1) = 1;

figure;
imagesc(T);