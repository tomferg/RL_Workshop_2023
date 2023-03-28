function [llSum,parameters] = eGreedy_Lik(parameters,behaviouralData, initialValue, numBlocks, numTrials, numArms)

% LL Array
ll = zeros([numBlocks,numTrials]);

% Assign values to parameters
epsilon = parameters(1);
learningRate = parameters(2);

% Extract Choices and Reward
choiceObs = squeeze(behaviouralData(1,:, :));
rewardObs = squeeze(behaviouralData(2,:, :));

% Loop around blocks
for block = 1:numBlocks

    %Initialize Bandit Values
    banditValues = zeros(numArms,1)+initialValue;

    %Loop Across Trials
    for trial = 1:numTrials
       
        %Deal with NaN's
        if choiceObs(block, trial) == -1
            
            %Update Liklihood
            ll(block, trial) = 1;
            
        else
            
            greedyResult = (epsilon/(length(banditValues)-1)) * ones(1,length(banditValues));
            
            %Find Choice
            [~,I] = max(banditValues);
            
            % Change to greedy result
            greedyResult(I) = 1-epsilon;
            
            %Extract the Choice the Model Made
            partChoice = choiceObs(block, trial);
            
            %Determine if Choice is rewarded
            reward = rewardObs(block, trial);
            
            %Compute Prediction Error
            rpe = reward - banditValues(partChoice);
            
            %Update Bandit Values
            banditValues(partChoice) = banditValues(partChoice) + learningRate * rpe;
            
            %Compute Log liklihood
            ll(block, trial) = greedyResult(partChoice);
    
        end
    end
end
    
%Sum Log likihood
llSum = -sum(log(ll), 'all');