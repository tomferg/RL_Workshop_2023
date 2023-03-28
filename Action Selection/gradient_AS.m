function [choices, rewards, H] = gradient_AS(parameters,arrayValues,initialValue, numBlocks, numTrials, numArms)

%Assign values to parameters
learningRate = parameters(1);
    
% Initialize Output Arrays
choices = zeros(1, numBlocks, numTrials);
rewards = zeros(1, numBlocks, numTrials);

for block = 1:numBlocks

    % Initialize preference values
    H = zeros(numArms, 1) + initialValue;
    
    % Start timer and reward
    time = 1;
    baseline = 0;

    % Loop Across Trials
    for trial = 1:numTrials
                    
        % Calculate Softmax
        num = exp(H);
        denom = sum(num);
        sm = num / denom;
        
        % Find cumulative sum
        softmaxSum = cumsum(sm);
    
        % Assign Values to softmax options
        softmaxOptions = softmaxSum > rand();
            
        % Find arm choice
        armChoice = find(softmaxOptions, 1, 'first');
        
         % Binary Reward
        reward = rand() < arrayValues(armChoice, block, trial);
    
        baseline = baseline + (reward - baseline) / time;
        
        one_hot = zeros(numArms,1);
        one_hot(armChoice) = 1;
        
        % Update Chosen H Values
        H = H +  learningRate *  (reward - baseline) * (one_hot - sm);
                        
        % Update Times
        time = time + 1;
                   
        %Assign Arm Choice & Reward Choice
        choices(1, block, trial) = armChoice;
        rewards(1, block, trial) = reward;
            
        
    end

end