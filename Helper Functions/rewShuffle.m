function [rewardValues]  = rewShuffle(armRewards,numTrials,numBlocks)

% Initialize Array
rewardValues = zeros([length(armRewards), numBlocks, numTrials]);
startReward = zeros(numBlocks, length(armRewards));

startReward(:,1) = armRewards(1);
startReward(:,2) = armRewards(2);

[m,n] = size(startReward);
[~,p] = sort(rand(m,n),2);
perm_data = reshape(startReward(repmat((1-m:0).',n,1)+p(:)*m),m,n);

% Loop Around blocks
for block = 1:numBlocks

    rewardValues(1, block, :) = perm_data(block,1); 
    rewardValues(2, block, :) = perm_data(block,2);

end

