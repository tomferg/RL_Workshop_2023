function [paramFit,LL,BIC,AIC] = eGreedy_Fit_stat(arrayValues,initialValue, numBlocks, numTrials, numArms)

obFunc = @(parameters) eGreedy_Lik_stat(parameters,arrayValues,initialValue, numBlocks, numTrials, numArms);

X0 = rand; % Epsilon
LowerBound = .001;
UpperBound = 1;
[paramFit, NegLL] = fmincon(obFunc, X0, [], [], [], [], LowerBound, UpperBound);

LL = -NegLL;
BIC = -2*LL + log(numBlocks*numTrials) * length(X0);
AIC = 2*length(X0) - 2*LL;

end