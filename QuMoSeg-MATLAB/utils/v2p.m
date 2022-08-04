function [ P ] = v2p( v,c )
% da vettore permutazione a matrice di permutazione
% input in one-line o word notation
% gestisce anche permutazioni parziali (con 0 dove non definita)
n =  length(v);
if nargin < 2
    c = n;
end
i = (1:n);
i(v==0)=[];
v(v==0)=[];
P = sparse(i,v,1,n,c);

% double stochastic
assert(all(sum(P,1)<=1) && all(sum(P,2)<=1))

end

