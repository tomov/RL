function i = samplePF(PF)
    % sample from a discrete probability distribution
    % using the universality of the uniform
    % i.e. F(x) ~ Unif(0, 1)
    %
    CDF = cumsum(PF);
    r = rand(1);
    i = find(CDF > r);
    i = i(1);
    assert(PF(i) > 0);
end
