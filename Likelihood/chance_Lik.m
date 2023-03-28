function [llSum,parameters] = chance_Lik(parameters,behaviouralData, initialValue, numBlocks, numTrials, numArms)

% Set up LL Array
ll = zeros(numBlocks, numTrials);

% Assign values to parameters
biasParam = parameters(1);

% Extract Choices and Reward
choiceObs = squeeze(behaviouralData(1, :, :));
rewardObs = squeeze(behaviouralData(2, :, :));

% Set up probabilities so they sum to 1
bias = zeros(numArms);
bias(1) = biasParam;
bias(2:end) = (1-biasParam) / (numArms-1);

% % Randomize Bias to one square
% bias = bias(randperm(length(bias)));

for block = 1:numBlocks

    % Randomize Bias to one square
    bias = bias(randperm(length(bias)));

    % Loop Across Trials
    for trial = 1:numTrials
       
        % Deal with NaN's
        if choiceObs(block, trial) == -1
            
            % Update Liklihood
            ll(block, trial) = 1;
            
        else
            
            %Extract the Choice the Model Made
            partChoice = choiceObs(block, trial);
            
            %Compute Log liklihood
            ll(block, trial) = bias(partChoice);
    
        end
        
    end
end

%Sum Log likihood
llSum = -sum(log(ll),'all');