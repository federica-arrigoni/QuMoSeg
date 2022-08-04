

function [x,y,d] = matrix2segment(P)
% d = number of motions
% x(i)=k if point i in the first image belongs to motion k
% y(i)=k if point i in the second image belongs to motion k
% x(i)=0 if point i does not belong to any motion in common with y

[n,m]=size(P);

x=zeros(1,n);
y=zeros(1,m);

d=0;

if sum(sum(P))==0
    return
end

for i=1:n
    row=P(i,:);
    if sum(row)~=0 % the row is non zero
        
        if i==1
            condition=false;
        else
            [condition,index]=ismember(row,P(1:i-1,:),'rows');
        end
        
        if condition
            % the row belongs to an already found motion
            x(i)=x(index);
        else
            % new motion is found
            d=d+1;
            x(i)=d;
            y(row~=0)=d;
        end
    end
end

end