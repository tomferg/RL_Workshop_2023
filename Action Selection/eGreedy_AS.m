function [choices, rewards] = eGreedy_AS(parameters,arrayValues,initialValue, numBlocks, numTrials, numArms)

%Assign values to parameters
epsilon = parameters(1);
learningRate = parameters(2);
    
%Initialize Bandit Values and Output Arrays
choices = zeros(1, numBlocks, numTrials);
rewards = zeros(1, numBlocks, numTrials);

% Loop around block
for block = 1:numBlocks

    % Initialize Bandit Values
    banditValues = zeros(numArms, 1)+initialValue;

    %Loop Across Trials
    for trial = 1:numTrials
        
        %Find max choice from the Array
        [maxValue,maxChoice] = max(banditValues);
        
        %Find if there are ties present
        tied = find(banditValues == maxValue);
        
        %Randomly Break Ties if they're present
        if length(tied) > 1
            maxChoice = randsample(tied,1);
        end
        
        %Assign maxChoice to arm Choice
        armChoice = maxChoice;
        
        %Randomly explore
        if epsilon > rand()
            
            % Choose Other arm
            [~,armChoice] = min(banditValues);
            
        end
        
        %Determine if Choice is rewarded
        reward =  rand() < arrayValues(armChoice, block, trial);
        
        %Compute Prediction Error
        rpe = reward - banditValues(armChoice);
        
        %Update Bandit Values
        banditValues(armChoice) = banditValues(armChoice) + learningRate * rpe;
       
        %Assign Arm Choice & Reward Choice
        choices(1, block, trial) = armChoice;
        rewards(1, block, trial) = reward;
            
        
    end

end