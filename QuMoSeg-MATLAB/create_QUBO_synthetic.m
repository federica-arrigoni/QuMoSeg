
clear,clc
addpath(genpath('./'))

%% parameters of the problem

d = 2; % number of motions
n = 4; % number of images
pavg = 10; % average number of points per image
no_error=2; % number of errors in each pair (should be a fraction of pavg)

%% generate ground-truth abolute/relative segmentation

[X_gt,p,dim,M,pairwiseEst] = generate_data_matches(d,n,pavg,true);
qbits=p*d % number of qubits
group_gt=nan(p,1);
for ppp=1:p
    group_gt(ppp)=find(X_gt(ppp,:));
end

Z_gt=X_gt*X_gt'; % relative segmentation
cumDim = [0;cumsum(dim(1:end-1))];
for i=1:n
    Z_gt(1+cumDim(i):cumDim(i)+dim(i),1+cumDim(i):cumDim(i)+dim(i)) = 0; % zeros on the diagonal
end

%% Add noise

[labels_pairwise]=globalMatrix2Labels(pairwiseEst,Z_gt,dim,n,ones(n)); % ground-truth pairwise labels
[labels_pairwise,Z]=globalAddNoise(labels_pairwise,pairwiseEst,dim,n,ones(n),d,no_error); % noisy pairwise labels

%% build matrices for Quantum

x_gt=X_gt(:); % ground-truth solution (binary vector)

[Q,A1,b1]=build_QUBO_tracemin(Z,d); % QUBO with constraints: rows sum to 1
[A2,b2]=constraints_simplified(dim,d,n,M); % simplified approach

% % check that constraints are satisfied
% norm(A2*x_gt-b2)
% norm(A1*x_gt-b1)

[Qd]=build_QUBO_dense(Z,d); % QUBO with dense matrix

%% save all permutations of ground-truth

pp=perms(1:d); % possible permutations of d motions
n_perm=size(pp,1);

x_gt_perm=zeros(length(x_gt),n_perm);
for k=1:n_perm
    PP=v2p(pp(k,:)); % permutation matrix
    Y=X_gt*PP; % apply permutation
    x_gt_perm(:,k)=Y(:);
end

%% save data for quantum approach

filename='synthetic_data';
save(filename,'Q','A1','b1','A2','b2','x_gt','Qd','x_gt_perm')

% % Note:
% % QuMoSeg-v1 uses Qd, A1, b1
% % QuMoSeg-v2 uses Q, A1, b1, A2, b2

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NOTE: quantum methods needs to be run separately, using the python code! 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Comparison with SOTA methods
%% Motion segmentation via synchronization - ICCVW 2019 (Arrigoni & Pajdla)

tau=0.001;
th=1.5;
[group_synch] = segment_synch(sparse(Z),d,tau,th); % labels
rows=1:p;
Y_synch=sparse(rows(group_synch~=0),group_synch(group_synch~=0),1,p,d); % segmentation

% Permutation Procrustes Analisys: X=Y*P with P permutation matrix
P = full(ppa(Y_synch,X_gt,'hungarian'));
Y_synch=Y_synch*P;

acc_ICCVW19=1-nnz(Y_synch(:)-X_gt(:))/qbits % accuracy

%% Robust motion segmentation from pairwise matches - ICCV 2019 (Arrigoni & Pajdla)

method='hungarian';
A=ones(n);
for i=1:n
    for j=i+1:n
        if isempty(labels_pairwise{i,j})
            A(i,j)=0; A(j,i)=0;
        end
    end
end

[I,J]=find(triu(A,1));
npairs=length(I);
pairs=mat2cell([I J],ones(1,npairs),2);

labels_pairs=cell(npairs,1);
tracks_pairs=cell(npairs,1);
for kk=1:npairs
    i=I(kk); j=J(kk);
    labels_pairs{kk}=labels_pairwise{i,j};
    tracks_pairs{kk}=[pairwiseEst{i,j}.ind1' pairwiseEst{i,j}.ind2'];
end

tic
group_pairs=segment_mode_general(labels_pairs,pairs,tracks_pairs,dim,n,d,method); % labels
toc
Y_mode=sparse(rows(group_pairs~=0),group_pairs(group_pairs~=0),1,p,d); % segmentation

% Permutation Procrustes Analisys: X=Y*Q with Q permutation matrix
P = full(ppa(Y_mode,X_gt,'hungarian'));
Y_mode=Y_mode*P;
acc_ICCV19=1-nnz(Y_mode(:)-X_gt(:))/qbits % accuracy

