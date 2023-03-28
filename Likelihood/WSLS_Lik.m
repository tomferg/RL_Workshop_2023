function [llSum,parameters] = WSLS_Lik(parameters,behaviouralData, initialValue, numBlocks, numTrials, numArms)

% LL Array
ll = zeros([numBlocks,numTrials]);

%Assign values to parameters
winStay = parameters(1);
% loseShift = parameters(2);

% Extract Choices and Reward
choiceObs = squeeze(behaviouralData(1,:, :));
rewardObs = squeeze(behaviouralData(2,:, :));


% Loop around blocks
for block = 1:numBlocks
    
    p = [.5, .5];

    for trial = 1:numTrials
        
        if trial == 1 || choiceObs(block, trial) == -1

            ll(block, trial) = 1;

        else

            %Determine if previous trial was a winner
            if rewardObs(block, trial-1) == 1
                
                p = winStay/2*[1 1];
                
                if choiceObs(block, trial-1) == -1

%                    p(randi(2)) = 1-winStay;
                    p = [.5, .5];

                else

                    % win stay
                    p(armLast) = 1-winStay/2;

                end

            else

                % lose shift
                p = (1-winStay/2) * [1 1];

                if choiceObs(block, trial-1) == -1
    
%                     p(randi(2)) = winStay / 2;
                    p = [.5 .5];
    
                else
    
                    p(armLast) = winStay / 2;
    
                end

            end

            %Update Liklihood
            ll(block, trial) = p(choiceObs(block, trial));

        end

        armLast = choiceObs(block, trial);

    end

end

%Sum Log likihood
llSum = -sum(log(ll), 'all');

