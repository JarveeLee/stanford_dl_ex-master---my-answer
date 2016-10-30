function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
datasize=size(data,2);
%%% YOUR CODE HERE %%%
z={};
a={};
%a{1}.w=data;
z=[z 0];
a=[a data];
for i=1:numHidden+1
    zt=zeros(ei.layer_sizes(i),datasize);
    zt=stack{i}.W*a{i};
    for j=1:datasize
        zt(:,j)=zt(:,j)+stack{i}.b;
    end
    z=[z zt];
    a=[a sigmoid(zt)];
end
hx=a{numHidden+2};
ht=a{numHidden+2};
for i=1:datasize
    hx(:,i)=hx(:,i)./sum(hx(:,i));
end
pred_prob=hx;
%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
num_classes=ei.layer_sizes(numHidden+1);
m=datasize;
label_vec=zeros(num_classes,m);
for i=1:m
    label_vec(labels(i),i)=1;
end
cost=sum(sum(label_vec.*log(ht)+(1-label_vec).*log(1-ht)))*(-1)/datasize;
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
Wd={};
bd={};
for i=1:numHidden+1
    Wd=[Wd stack{i}.W];
    bd=[bd stack{i}.b];
end
delta=z;
delta{numHidden+2}=zeros(num_classes,1);
%{
Thx=zeros(num_classes,datasize);
for i=1:datasize
    Thx(:,i)=hx(:,i)-label_vec(i);
end
%Thx=Thx.*sigmoidd(z{numHidden+2});
%}
delta{numHidden+2}=ht-label_vec;
i=numHidden+1;
while i>=2
    delta{i}=(stack{i}.W'*delta{i+1}).*a{i}.*(1-a{i});
    i=i-1;
end
i=numHidden+1;
while i>=1
    Wd{i}=delta{i+1}*a{i}'/datasize;
    bd{i}=sum(delta{i+1},2)/datasize;
    gradStack{i}.W=Wd{i};
    gradStack{i}.b=bd{i};
    i=i-1;
end
%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
lambda=ei.lambda;
for i=1:numHidden
    cost=cost+lambda*sum(sum(stack{i}.W.*stack{i}.W))/(2*m);
    gradStack{i}.W=gradStack{i}.W+lambda*stack{i}.W/m;
end
%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



