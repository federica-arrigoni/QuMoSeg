

function pairwiseEst=ZtoMatches(Z,dim,ncams)

% transform permutation matrices into matches

cumDim = [0;cumsum(dim(1:end-1))];
pairwiseEst = cell(ncams,ncams);

for i = 1:ncams
    for j=i+1:ncams
        
        Zij=Z(1+cumDim(i):cumDim(i)+dim(i),1+cumDim(j):cumDim(j)+dim(j));
        [ind1,ind2]=find(Zij);
        
        pairwiseEst{i,j}.ind1 = ind1';
        pairwiseEst{i,j}.ind2 = ind2';
        
    end
end


end