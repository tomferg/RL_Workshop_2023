function [choices, rewards] = WSLS_AS(parameters,arrayValues,initialValue, numBlocks, numTrials, numArms)

%Assign values to parameters
winStay = parameters(1);
%loseShift = parameters(2);

%Initialize Bandit Values and Output Arrays
choices = zeros(1, numBlocks, numTrials);
rewards = zeros(1, numBlocks, numTrials);

for block = 1:numBlocks
    
    %Loop Across Trials
    for trial = 1:numTrials
        
        if trial == 1
            
            %Randomly select first trial
            p = [.5 .5];
            
        else
            
            %Determine if previous trial was a winner
            if rewards(1, block, trial-1) > 0
                
                % win stay
                p = winStay/2*[1 1];
                p(armLast) = 1-winStay/2;
                
            else
                
                % lose shift (1- win-stay)
                p = (1-winStay/2) * [1 1];
                p(armLast) = winStay / 2;

            end
            
        end
        
        %Make Arm Choice
        armChoice = max(find([eps cumsum(p)] < rand));
        
        %Determine if Choice is rewarded
        reward = rand() < arrayValues(armChoice, block, trial);

        %Assign Arm Choice & Reward Choice
        choices(1, block,trial) = armChoice;
        rewards(1, block,trial) = reward;

        % Assign Arm Choice
        armLast = armChoice;
        
    end
    
end