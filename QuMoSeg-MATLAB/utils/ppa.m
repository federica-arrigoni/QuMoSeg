function Q = ppa( X, Y , method)
% Permutation Procrustes Analisys: risolve il problema procustiano
%  Y=X*Q con Q matrice di permutazione (caso speciale di ortogonale)
% con  X=I trova la nearest permutation
% rif. Gower sec. 7.1, Higham: matrix procrustes problems

% default
if nargin < 3
    method = 'full';
end

n = size(X,2); n2 = n*n;

A = -speye(n2);
b = sparse(n2,1) ;

if strcmp(method,'full')
    %options=optimset('Display', 'off' , 'Algorithm', 'interior-point');
    options = optimoptions('linprog','Algorithm','dual-simplex','Display', 'off');

    C = -(X'*Y);
    Aeq = [ kron(ones(1,n), speye(n));  kron(speye(n), ones(1,n))];
    beq = ones(2*n,1);
    x = linprog(C(:),A,b,Aeq,beq,[],[],[], options);
    Q = spones(ivec(x,n)> 0.9);
   
elseif  strcmp(method,'partial') % partial permutation: relax equality constraint
    %options=optimset('Display', 'off' , 'Algorithm', 'interior-point');
    options = optimoptions('linprog','Algorithm','dual-simplex','Display', 'off');

    C = wthresh(-(X'*Y),'s', 0.0065); % linprog finds a  minimum
    A = [ A; kron(ones(1,n), speye(n));  kron(speye(n), ones(1,n))];
    b = [b; ones(2*n,1)];
    x = linprog(C(:),A,b,[],[],[],[],[], options);
    Q = spones(ivec(x,n)> 0.9);
    Q = spones(Q.*C); % elimina 1 di Q che hanno contributo zero (grazie a  wthresh)
    
elseif strcmp(method,'hungarian') % use hungarian algorithm
    C = wthresh(-(X'*Y),'s', 0.0065); 
    Q = v2p(munkres(C),size(Y,2));
    Q = spones(Q.*C); % elimina 1 di Q che hanno contributo zero (grazie a  wthresh)
else
    error('option not supported')
    
end







