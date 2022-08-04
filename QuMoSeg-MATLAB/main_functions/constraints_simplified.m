
function [A,b]=constraints_simplified(dim,d,n,M)

% force each motion in each image to have a predefined number of points

%A=zeros(d*n,d*p);
A=[];
for i=1:n
    Ai=ones(1,dim(i));
    A=blkdiag(A,Ai);
end

A=kron(eye(d),A);   
b=M(:);

end