

function [Q,A,b]=build_QUBO_tracemin(Z,d) % QUBO

p=size(Z,1);

Q=-kron(eye(d),Z); % minimize

A=kron(ones(1,d),eye(p)); % rows sum to 1 (linear constraint)
b=ones(p,1);

end