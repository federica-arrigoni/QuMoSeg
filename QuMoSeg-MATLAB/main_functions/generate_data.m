
function [X,p,dim,M] = generate_data(d,n,pavg)
%
% SIMPLIFIED SITUATION: each motion is equally probable,
% all images have the same number of points
% create labels as segmentation matrices without using matches explicitly
%
% INPUT
% d = number of motions
% n = number of images
% pavg = number of points per image
%
% OUTPUT
% X = (p x d) matrix of absolute segmentations
% p = total number of points over all images
% dim(i) = number of points in image i
% M(i,k) = number of points in motion k in image i

cond=true;
while cond % check that we have at least one point per motion per image
    
    %% Parameters of the problem
    
    p=n*pavg;  %total number of points over all images
    dim=pavg*ones(1,n)'; % all images have the same number of points
    cumDim = [0;cumsum(dim(1:end-1))];
    
    q = ones(d,1)/d; % probability that a point belongs to a motion (each motion is equally probable)
    q(end)=1-sum(q(1:end-1)); % ensure it is a probability
    
    %% Create absolute segmentation
    
    X=zeros(p,d); % absolute segmentation
    prob=rand(p,1);
    cumProb = [0;cumsum(q(1:end))]; % cumulative probability
    for k=1:d
        ind = find(prob<=cumProb(k+1) & prob>cumProb(k));
        X(ind,k)=1;
    end
    X(prob==0,1)=1; % ensure that each point is assigned a motion
    
    %% Fix the ambiguity
    
    mm=find(X(1,:)); % motion the first point belongs to
    if mm~=1 % exchange columns
        a=X(:,1);
        b=X(:,mm);
        X(:,1)=b;
        X(:,mm)=a;
    end
    
    %% Count the effective number of points per motion per image
    
    M=zeros(n,d);
    for i=1:n
        M(i,:)=sum(X(1+cumDim(i):cumDim(i)+dim(i),:),1); % sum columns (count points per motion)
    end
    
    cond=~isempty(find(M==0)); % check that we have at least one point per motion per image
    
end

end
