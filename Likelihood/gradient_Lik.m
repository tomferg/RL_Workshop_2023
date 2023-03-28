function [llSum,parameters] = gradient_Lik(parameters,behaviouralData, initialValue, numBlocks, numTrials, numArms)

learningRate = parameters(1);

% Set up LL Array
ll = zeros(numBlocks, numTrials);
    
% Extract Choices and Reward
choiceObs = squeeze(behaviouralData(1, :, :));
rewardObs = squeeze(behaviouralData(2, :, :));
    
for block = 1:numBlocks

    % Start timer and reward
    H = zeros(numArms, 1) + initialValue;
    time = 1;
    baseline = 0;

    % Loop Across Trials
    for trial = 1:numTrials
       
        %Deal with NaN's
        if choiceObs(block, trial) == -1
            
            %Update Liklihood
            ll(block, trial) = 1;
            
        else
        
            % Calculate Softmax
            num = exp(H);
            denom = sum(num);
            sm = num / denom;
                
            % Extract the Choice the Model Made
            partChoice = choiceObs(block, trial);
            
            % Determine if Choice is rewarded
            reward = rewardObs(block, trial);
        
            baseline = baseline + (reward - baseline) / time;
            
            one_hot = zeros(numArms,1);
            one_hot(partChoice) = 1;
            
            % Update Chosen H Values
            H = H +  learningRate *  (reward - baseline) * (one_hot - sm);
                            
            % Update Times
            time = time + 1;
                       
            % Compute Log liklihood
            ll(block, trial) = sm(partChoice);
    
        end
        
    end

end

%Sum Log likihood
llSum = -sum(log(ll),'all');
   
