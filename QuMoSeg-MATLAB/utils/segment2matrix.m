

function P = segment2matrix(x,y,d)
% d = number of motions 
% x(i)=k if point i in the first image belongs to motion k
% y(i)=k if point i in the second image belongs to motion k
% x(i)=0 if point i does not belong to any motion in common with y

n=length(x);
m=length(y);

P=zeros(n,m);

for i=1:d
    P(x==i,y==i)=1;
end


end