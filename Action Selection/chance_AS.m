function [choices, rewards] = chance_AS(parameters,arrayValues,initialValue, numBlocks, numTrials, numArms)

% Assign values to parameters
biasParam = parameters(1);

% Initialize Output Arrays
choices = zeros(1, numBlocks, numTrials);
rewards = zeros(1, numBlocks, numTrials);

% Set up probabilities so they sum to 1
bias(1) = biasParam;
bias(2:end) = (1-biasParam) / (numArms-1);

for block = 1:numBlocks

    % Randomize Bias to one square
    bias = bias(randperm(length(bias)));

    % Loop Across Trials
    for trial = 1:numTrials
    
        %Make choice based on bias
        armChoice = max(find([0 cumsum(bias)] < rand));
    
        %Determine if Choice is rewarded
        reward = rand() < arrayValues(armChoice, trial);
    
        %Assign Arm Choice & Reward Choice
        choices(1, block, trial) = armChoice;
        rewards(1, block, trial) = reward;
    
    end

end

