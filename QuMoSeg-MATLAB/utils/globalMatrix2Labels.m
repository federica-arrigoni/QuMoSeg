
function [labels_pairwise]=globalMatrix2Labels(pairwiseEst,Z,dim,n,A)

% transform segmentation matrices into labels

cumDim = [0;cumsum(dim(1:end-1))];

labels_pairwise=cell(n);
for i=1:n
    
    for j=i+1:n
        
        if A(i,j)==1
            
            ind1=pairwiseEst{i,j}.ind1;
            ind2=pairwiseEst{i,j}.ind2;
            
            Zij=Z(1+cumDim(i):cumDim(i)+dim(i),1+cumDim(j):cumDim(j)+dim(j));
            [x,y] = matrix2segment(Zij);
            assert(all(x(ind1)==y(ind2)))
            
            % save pairwise labels
            labels_pairwise{i,j}=x(ind1);
            labels_pairwise{j,i}=x(ind1);
            
        end
    end
end


end




