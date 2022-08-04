
% Author: Federica Arrigoni, 2019
% Reference: Robust motion segmentation from pairwise matches. 
% Federica Arrigoni and Tomas Pajdla. ICCV 2019.

%%
function [group]=segment_mode_general(labels_subsets,subsets,tracks,dim,ncams,d,method_permutation)

% method_permutation='hungarian';
m=sum(dim);
cumDim = [0;cumsum(dim(1:end-1))];
group=zeros(m,1);
nsubsets=length(subsets);

%% Compute pairwise permutations

fprintf('Computing permutations between triplets...')

% Construct a graph where each node is a triplet
Q_pair=sparse(d*nsubsets,d*nsubsets);

for k=1:nsubsets
    tr_k=subsets{k};
    groups_k=labels_subsets{k}; % labels of triplet k
    
    Q_pair(d*k-d+1:d*k,d*k-d+1:d*k)=speye(d);
    
    %%%%% PRINT!!!!!!!
    %%%%%[k nsubsets]
    
    for h=k+1:nsubsets
        tr_h=subsets{h};
        groups_h=labels_subsets{h}; % labels of triplet k
        
        [images]=intersect(tr_k,tr_h); % common cameras
        
        if ~isempty(images)
            for l=1:length(images)

                % transform labels of the tracks in the triplet into labels of
                % points in the current image
                labels_k=labels_full(tracks,subsets,dim,images(l),k,groups_k);
                labels_h=labels_full(tracks,subsets,dim,images(l),h,groups_h);
                
                % matrix representation
                Pk=a_labels2matrix(dim,images(l),labels_k,d);
                Ph=a_labels2matrix(dim,images(l),labels_h,d);
                Qhk(:,:,l)=full(ppa(Pk,Ph,method_permutation)');
            end
            
            Qhk=sparse(single_averaging_permutations(Qhk,method_permutation,d));
            
            Q_pair(d*h-d+1:d*h,d*k-d+1:d*k)=Qhk;
            Q_pair(d*k-d+1:d*k,d*h-d+1:d*h)=Qhk';
            
            clear Qhk
        end
    end
end

%% Permutation synchronization

fprintf('\nPerforming permutation synchronization...')

Q_synch = pachauri_synch(Q_pair,d,nsubsets,method_permutation);
%lambda=eig(Q_pair);
%lambda=sort(lambda,'descend');
%figure,plot(lambda,'o')
%figure,spy(Q_pair)

% Update labels
for k=1:nsubsets

    groups_k=labels_subsets{k}; % labels of triplet k
    P_current=labels2matrix(groups_k,d); % matrix representation
    
    Q=Q_synch(d*k-d+1:d*k,:); % optimal permutation
    P_current=P_current*Q; %full(Q)
    
    % Go back to labels
    labels_subsets{k}=matrix2labels(P_current);
    
end

%% Do segmentation for each image independently 

fprintf('\nPerforming the mode...')

%subsets_matrix=cell2mat(subsets); % DA CAMBIARE SE VOGLIAMO ESTENDERE A SOTTOGRAFI DI DIMENSIONI DIVERSE INVECE CHE TRIPLE
subsets_matrix=subsets2matrix(subsets,nsubsets);

for i=1:ncams
    
    % find triplets involving image i 
    index_i=find(sum(subsets_matrix==i,2)>0);
    n_estimates=length(index_i);
    estimates=nan(dim(i),n_estimates);
    
    for k=1:n_estimates
        
        tr=index_i(k); % current triplet
        groups_tr=labels_subsets{tr}; % labels of current triplet
        
        % transform labels of the tracks in the triplet into labels of
        % points in image i
        [current_estimate]=labels_full(tracks,subsets,dim,i,tr,groups_tr);
        
        estimates(:,k)=current_estimate;
    end
    
    % CHECK
    %[estimates mode(estimates,2)]
    
    %% Do single averaging
    
    n_observations=sum(~isnan(estimates),2); % estimates(n_observations<=1)=0;
    
    % ignore all outliers/mismatches
    estimates(estimates==0)=nan; 

    % perform the mode (most frequent label)
    final_estimate=mode(estimates,2);
    final_estimate(isnan(final_estimate))=0; 
    
    % count the number of entries that are equal to the mode: we require at
    % least two observations equal to the mode
    n_estimates=size(estimates,2);
    count=zeros(dim(i),1);
    for l=1:n_estimates
        count=count+( estimates(:,l)==final_estimate );
    end
    final_estimate( (count<=1) & (n_observations>=2) ) =0;
    
    % classify points in image i
    index_i=1+cumDim(i):cumDim(i)+dim(i); % points in image i
    group(index_i)=final_estimate;
    
end

fprintf('\nDone!\n')

end


%%

function [current_estimate]=labels_full(tracks,triplets,dim,i,tr,groups_tr)

current_triplet=triplets{tr};
[~,ind]=ismember(i,current_triplet);

ind_points=tracks{tr}(:,ind); % current triplet

current_estimate=nan( dim(i),1 );
current_estimate(ind_points)=groups_tr;

end

%%

function M=single_averaging_permutations(M_values,method,d)

% only one element: no averaging
if size(M_values,3)==1
    M=M_values;
else
    % sum of the entries
    M=sum(M_values,3);
end

% Project onto Permutations
% M=matrix2permutation(M); % fast approximate projection
M=ppa(eye(d),M,method);

end

%%

function Q_synch = pachauri_synch(Q_pair,d,ncams,method)

% Compute the d leading eigenvectors
%[U,Ds] = eigs(Q_pair,d,'la');
[U,Ds] = eigs(Q_pair,d,'lm');

% % Ensure that the column sums are non-negative
% U=U*diag(sign(sum(U))); 

% Rescale eigenvectors by corresponding eigenvalues
U=real(U)*sqrt(abs(Ds));

% Multiply by inverse of first block
U1=U(1:d,:);
U=U*U1';

% Project onto permutations
Q_synch=zeros(d*ncams,d);
for i=1:ncams
    Q_synch(d*i-d+1:d*i,:)=ppa(eye(d),U(d*i-d+1:d*i,:),method);
end

end


%%

function [P_reference]=a_labels2matrix(dim,i,labels,d) % matrix

ind_inliers=find(labels~=0 & ~isnan(labels));
P_reference=sparse(ind_inliers,labels(ind_inliers),1,dim(i),d); % image i

end

function labels_current=a_matrix2labels(P,dim,i,d)

%[ind1,ind2]=find(P);
labels_current=zeros(dim(i),1);

% for k=1:length(ind1)
%     
%     if ~isempty(find(P(ind1(k),:)))
%         labels_current(k)=find(P(ind1(k),:));
%     end
% end

for k=1:d
    labels_current(find(P(:,k)))=k;
end

end

%%
function [P]=labels2matrix(labels,d)

ind_inliers=find(labels~=0);
P=sparse(ind_inliers,labels(ind_inliers),1,length(labels),d);

end

function labels=matrix2labels(P)

n=size(P,1);
labels=zeros(n,1);

for k=1:n
    lab=find(P(k,:));
    if ~isempty(lab)
        labels(k)=lab;
    end
end

end

%%

function subsets_matrix=subsets2matrix(subsets,nsubsets)

dim_subsets=zeros(nsubsets,1);
for k=1:nsubsets
    dim_subsets(k)=length(subsets{k});
end

subsets_matrix=nan(nsubsets,max(dim_subsets));
for k=1:nsubsets
    subsets_matrix(k,1:dim_subsets(k))=subsets{k};
end


end



