T = ones(5,5);
T(1,1) = 0;
T(2,2) = 0;
T(3,3) = 0;
T(4,4) = 0;
T(5,5) = 0;
T(1,5) = 0;
T(5,1) = 0;

TT = zeros(15,15);
TT(1:5,1:5) = T;
TT(6:10,6:10) = T;
TT(11:15,11:15) = T;

TT(1, 5 + 5) = 1;
TT(5 + 5, 1) = 1;

TT(1 + 5, 5 + 10) = 1;
TT(5 + 10, 1 + 5) = 1;

TT(1 + 10, 5) = 1;
TT(5, 1 + 10) = 1;

imagesc(TT);