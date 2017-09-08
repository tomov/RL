rng default;

D = 10;
N = 100;
mu = zeros(D, 1);
sigma = diag(ones(D, 1));
X = mvnrnd(mu, sigma, N);

Z = zeros(N, D);

A = 10;
Z(1:A:end, :) = mvnrnd(mu, sigma * 30, N/A);
X = X + Z;

Y = cumsum(X);
imagesc(corr(Y', 'type', 'Pearson'));

as = {'left', 'straight', 'right', 'straight', 'left', 'straight', 'right', 'straight', 'left', 'left'};
for i = 1:numel(as)
    text(A*i - A*0.8, A*i - A*0.5, as{i});
end
xlabel('time');
ylabel('time');
title('Neural RDM for given ROI');

