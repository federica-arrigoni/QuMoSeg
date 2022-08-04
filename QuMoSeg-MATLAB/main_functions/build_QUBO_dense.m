

function [Q,A,b]=build_QUBO_dense(Z,d) % QUBO

p=size(Z,1);

Q=-kron(eye(d),2*Z-ones(p)); % minimize

A=kron(ones(1,d),eye(p)); % rows sum to 1 (linear constraint)
b=ones(p,1);

end