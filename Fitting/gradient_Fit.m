function [paramFit,LL,BIC,AIC] = gradient_Fit(arrayValues,initialValue, numBlocks, numTrials, numArms)

%options = optimoptions(@fmincon,'Algorithm','interior-point');
%PSoptions = optimoptions(@patternsearch);
obFunc = @(parameters) gradient_Lik(parameters,arrayValues,initialValue, numBlocks, numTrials, numArms);

X0 = .01 + (2-.01) .* rand();

LowerBound = .01;
UpperBound = 2;
[paramFit, NegLL] = fmincon(obFunc, X0, [], [], [], [], LowerBound, UpperBound);
%[paramFit, NegLL] = patternsearch(obFunc,X0,[],[],[],[],LowerBound, UpperBound,PSoptions);

LL = -NegLL;
BIC = -2*LL + log(numBlocks*numTrials) * length(X0);
AIC = 2*length(X0) - 2*LL;


end