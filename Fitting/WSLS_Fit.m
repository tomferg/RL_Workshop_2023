function [paramFit,LL,BIC,AIC] = WSLS_Fit(arrayValues,initialValue, numBlocks, numTrials, numArms)

obFunc = @(parameters) WSLS_Lik(parameters,arrayValues,initialValue, numBlocks, numTrials, numArms);

X0 = rand; %for WS
LowerBound = 0.01;
UpperBound = 1;
[paramFit, NegLL] = fmincon(obFunc, X0, [], [], [], [], LowerBound, UpperBound);

LL = -NegLL;
BIC = -2*LL + log(numBlocks*numTrials) * length(X0);
AIC = 2*length(X0) - 2*LL;

end