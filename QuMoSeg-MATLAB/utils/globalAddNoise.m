
function [labels_noise,Znoise]=globalAddNoise(labels_pairwise,pairwiseEst,dim,n,A,d,n_error)

cumDim = [0;cumsum(dim(1:end-1))];
m=sum(dim);

Znoise=sparse(m,m);
labels_noise=cell(n);
for i=1:n
    
    for j=i+1:n
        
        if A(i,j)==1
            
            %% extract current labels
            group=labels_pairwise{i,j};
            
            %% ... add noise 
            group_noisy=singlePair_AddNoise(group,n_error,d);
            
            %% save noisy pairwise labels
            labels_noise{i,j}=group_noisy;
            labels_noise{j,i}=group_noisy;
            
            %% save noisy matrix
            
            ind1=pairwiseEst{i,j}.ind1;
            ind2=pairwiseEst{i,j}.ind2;
            
            % assign a label to ALL points in image h (zero means no label)
            groups_i=zeros(dim(i),1);
            groups_i(ind1)=group_noisy;
            
            % assign a label to ALL points in image k (zero means no label)
            groups_j=zeros(dim(j),1);
            groups_j(ind2)=group_noisy;
            
            % construct a binary matrix that encodes the segmentation
            Zhk=segment2matrix(groups_i,groups_j,d);
            
            Znoise(1+cumDim(i):cumDim(i)+dim(i),1+cumDim(j):cumDim(j)+dim(j)) = Zhk;
            Znoise(1+cumDim(j):cumDim(j)+dim(j),1+cumDim(i):cumDim(i)+dim(i)) = Zhk';
            
        end
    end
end


end

%%

function group_noisy=singlePair_AddNoise(group,n_error,d)

group_noisy=group;

n_points=length(group); % number of points

% select points where errors will be introduced
selected_points=randperm(n_points,n_error);

for k=1:n_error
    point=selected_points(k);
    old_label=group(point);
    
    if d==2 % 2 motions: we switch the motions
        if old_label==1
            new_label=2;
        else
            new_label=1;
        end
    else
        available_labels=setdiff(1:d,old_label);
        new_label=available_labels(randi(length(available_labels)));
    end
    
    group_noisy(point)=new_label;
end


end
