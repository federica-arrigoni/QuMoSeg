
function [X,p,dim,M,pairwiseEst] = generate_data_matches(d,n,pavg,balanced)
%
% SIMPLIFIED SITUATION: each motion is equally probable,
% all images have the same number of points
% create labels from matches
%
% INPUT
% d = number of motions
% n = number of images
% pavg = number of points per image
% balanced = true (each motion is equally probable)
% balanced = false (you need to specify probabilities of each motion, e.g. line 33)
%
% OUTPUT
% X = (p x d) matrix of absolute segmentations
% p = total number of points over all images
% dim(i) = number of points in image i
% M(i,k) = number of points in motion k in image i
% pairwiseEst is a cell containing matches between image pairs

cond=true;
while cond % check that we have at least one point per motion per image
    
    %% Parameters of the problem
    
    p=n*pavg;  %total number of points over all images
    dim=pavg*ones(1,n)'; % all images have the same number of points
    cumDim = [0;cumsum(dim(1:end-1))];
    
    if balanced
        q = ones(d,1)/d; % probability that a point belongs to a motion (each motion is equally probable)
        q(end)=1-sum(q(1:end-1)); % ensure it is a probability
    else
        q=[0.4;0.6];
    end    
    
    %% Create ground-truth trajectories/permutations/matches
    
    % P=zeros(p,pavg);
    P=[]; % absolute permutations
    for i=1:n
        v = randperm(pavg);
        P=[P;v2p(v)];
    end
    
    Zmatch=P*P'; % relative permutations
    pairwiseEst=ZtoMatches(Zmatch,dim,n); % matches

    %% Create absolute segmentation
    
    prob=rand(pavg,1); % probability that a track belongs to a motion
    cumProb = [0;cumsum(q(1:end))]; % cumulative probability
    
    X=zeros(p,d); % absolute segmentation
    for k=1:d
        ind = find(prob<=cumProb(k+1) & prob>cumProb(k)); % index associated with tracks % current motion
        for bb=1:length(ind)
            ind_a=find(P(:,ind(bb))); % find points associated with the current track/motion
            X(ind_a,k)=1;
        end
    end
    ind_c=find(sum(X,2)==0);
    X(ind_c,1)=1; % ensure that each point is assigned a motion
    
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
