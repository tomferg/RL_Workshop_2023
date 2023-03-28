%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     RL - UVic Workshop        %%%
%%%    Code by: Thomas Ferguson   %%%
%%%     Last Update: 26/3/23      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Step 0.1 - Set-up environment and Parameters
clc;clear; close all;
rng(1000);

% add folders to path
addpath('./Helper Functions/','./Action Selection/',...
    './Data/', './Likelihood/', './Figures/','./Fitting/', './Matrices') 

% Task Parameters
numArms = 2;
numTrials = 20;
numBlocks = 5;
numModels = 4;
numPart = 30;
initialValue = 0.5;
armRewards = [.6, .1];
trials = 1:20;

%% Step 0.2 - Plotting Set-Up
% Labels for Plots
titles = {'eGreedy-S', 'eGreedy' , 'WSLS', 'Gradient'};
xlabels = {'Sim - \epsilon','Sim - \epsilon', 'Sim - \alpha', ...
    'Sim - WS', 'Sim - \alpha'};
ylabels = {'Fit - \epsilon', 'Fit - \epsilon', 'Fit - \alpha',....
    'Fit - WS', 'Fit - \alpha',};

% Plot Colors
color1 = [85/255, 202/255, 237/255];
color2 = [44/255, 158/255, 75/255];
color3 = [129/255, 60/255, 199/255];
color4 = [227/255, 84/255, 41/255];
color5 = [214/255, 34/255, 133/255];

% Colors Set up for Plotting
plotColors = [color2; color3; color4; color5]; % For Figs with Model Only
plotColors2 = [color1; color2; color3; color4; color5]; %  for Figs with Model+Human

%% Step 0.3 - Extract Choices and Reward from Beh files
% Load Behavioural Data
behFiles = dir('./data/*.txt');

% Create Empty Array
behArray = zeros([numPart, numArms, numBlocks, numTrials]);

% Loop around People
for eCt = 1:numPart

    % Load Data
    dat = load(behFiles(eCt).name);
    
    % Clear Arrays
    cBlock = zeros([numBlocks, numTrials]);
    rBlock = zeros([numBlocks, numTrials]);

    % Change into blocks
    startTrial = 1;
    for blockCounter = 1:numBlocks
        cBlock(blockCounter, :) =  dat(startTrial:startTrial+19, 11);
        rBlock(blockCounter, :)  = dat(startTrial:startTrial+19, 15);
        startTrial = startTrial+20;
    end

    % Assign to arrays
    behArray(eCt, 1, :, :) = cBlock;
    behArray(eCt, 2, :, :) = rBlock;

end

% Save Behavioural Array
save('./Matrices/behMatrix.mat','behArray');


%% Step 1.1 - Model Validation - Task Learning
% Parameters for models - Manually Chosen
para_eGs = .1;
para_eG= [.1, .2]; 
para_Gr = 2;
para_WSLS = .001;

% Loop Around People
for pCt = 1:numPart
    
    % Generate Reward Distributions for Both Arms
    rewVal = rewShuffle(armRewards, numTrials, numBlocks);

    % eGreedy - Stationary Action Selection
    [valCho.eGs(pCt,:,:), valRew.eGs(pCt,:,:)] = eGreedy_AS_stat(para_eGs,...
        rewVal, initialValue, numBlocks, numTrials, numArms);

    % eGreedy - Action Selection
    [valCho.eG(pCt,:, :), valRew.eG(pCt,:, :)] = eGreedy_AS(para_eG,...
        rewVal, initialValue, numBlocks, numTrials, numArms);

    % WSLS - Action Selection
    [valCho.WSLS(pCt,:, :), valRew.WSLS(pCt, :, :)] = WSLS_AS(para_WSLS,...
        rewVal, initialValue, numBlocks, numTrials, numArms);

    % Gradient - Action Selection
    [valCho.Gr(pCt,:, :), valRew.Gr(pCt, :, :)] = gradient_AS(para_Gr,...
        rewVal, initialValue, numBlocks, numTrials, numArms);

end 

% Compute Average Wins across blocks and participants for each Model
meanReward(1, :) = squeeze(mean(valRew.eGs, [1, 2]));
meanReward(2, :) = squeeze(mean(valRew.eG, [1, 2]));
meanReward(3, :) = squeeze(mean(valRew.WSLS, [1, 2]));
meanReward(4, :) = squeeze(mean(valRew.Gr, [1, 2]));

% Figure - Model Validation
figure
% Loop Around Model Averages
for plotCount = 1:numModels
    subplot(1, 4, plotCount)
    plot(trials, meanReward(plotCount, :), "Color", plotColors(plotCount, :))
    title(titles{plotCount})
    xlabel('Trials')
    ylabel('Wins')
    ylim([.2, .8])
    ax = gca;
    ax.FontSize = 12;
    ax.FontName = 'Times';
    ax.LineWidth = 1;
    ax.Box = 'off';
end
% Save Figure
x_width=8 ;y_width=4;
set(gcf, 'PaperPosition', [0 0 x_width y_width]);
print('./Figures/Fig1_Simulation', '-dtiff', '-r300');

%% Step 1.2 - Model Validation - Parameter Recovery
% Set up empty parameter arrays
simParam = zeros(numPart, 7);
fitParam = zeros(numPart, 7);

% Generate Simulated Parameters
para = paraGenerate(numPart); 

% Loop Around "Participants"
for mCt = 1:numPart
    
    % Create Task Data for simulation
    taskData = rewShuffle(armRewards, numTrials, numBlocks);

    %  Simulate Models running through Task
    % eGreedy - Stationary
    [ceGS, reGS] = eGreedy_AS_stat(para.egreedy_stat(mCt, :), taskData,...
        initialValue, numBlocks, numTrials, numArms);
    % eGreedy
    [ceG, reG] = eGreedy_AS(para.egreedy(mCt, :), taskData, initialValue,...
        numBlocks, numTrials, numArms);
    % Win-Stay, Lose-Shift
    [cWSLS, rWSLS] = WSLS_AS(para.WSLS(mCt, :), taskData, initialValue,...
        numBlocks, numTrials, numArms);
    % Gradient
    [cGr, rGr] = gradient_AS(para.gradient(mCt, :), taskData, initialValue,...
        numBlocks, numTrials, numArms); 

    % Optimize/Fit Models
    % eGreedy - Stationary
    [para_eGS] = eGreedy_Fit_stat(vertcat(ceGS, reGS), initialValue, numBlocks, numTrials, numArms);
    % eGreedy
    [para_eG] = eGreedy_Fit(vertcat(ceG, reG), initialValue, numBlocks, numTrials, numArms);
    % Win-Stay, Lose-Shift
    [para_WSLS] = WSLS_Fit(vertcat(cWSLS, rWSLS), initialValue, numBlocks, numTrials, numArms);
    % Gradient
    [para_Gr] = gradient_Fit(vertcat(cGr, rGr), initialValue, numBlocks, numTrials, numArms);

    % Assign simulated parameters to arrays for plotting
    simParam(mCt, 1) = para.egreedy_stat(mCt, 1);
    simParam(mCt, 2) = para.egreedy(mCt, 1);
    simParam(mCt, 3) = para.egreedy(mCt, 2);
    simParam(mCt, 4) = para.WSLS(mCt, 1);
    simParam(mCt, 5) = para.gradient(mCt, 1);

    % Same for Fitted
    fitParam(mCt, 1) = para_eGS(1);
    fitParam(mCt, 2) = para_eG(1);
    fitParam(mCt, 3) = para_eG(2);
    fitParam(mCt, 4) = para_WSLS(1);
    fitParam(mCt, 5) = para_Gr;

end

% Figure - Parameter Recovery
% Needed so both eGreedy parameters are on same column
plotStart = 1;
for plotCount2 = 1:5
    % Needed so both eGreedy parameters on same column
    if plotCount2 == 3
        subplot(2,4,plotStart+3)
    else
        subplot(2,4,plotStart)
        plotStart = plotStart + 1;
    end
    scatter(simParam(:,plotCount2),fitParam(:,plotCount2),12,...
        'MarkerEdgeColor',[0 0 0],...
              'MarkerFaceColor',plotColors(plotStart-1, :));
    title(titles{plotStart-1})
    hold on
    corrVal = num2str(corr(simParam(:, plotCount2), fitParam(:, plotCount2)),2);
    text(0.1,.9,strcat('r =',num2str(corrVal)))
    ylim([0 1])
    h = lsline;
    h.LineWidth = 1.5;
    xlabel(xlabels{plotCount2})
    ylabel(ylabels{plotCount2})
    ax =gca;
    ax.FontSize = 12;ax.FontName = 'Times';ax.LineWidth = 1.5;ax.Box = 'off';
end
set(gcf, 'PaperUnits', 'inches');x_width=10 ;y_width=6;
set(gcf, 'PaperPosition', [0 0 x_width y_width]);
print('./Figures/Fig2_ParamRecovery','-dtiff','-r500');

%% Step 2 - Tune Models
% Load Behavioural Data
para = paraGenerate(numPart); 

% Load Matrix
load('./Matrices/behMatrix.mat');

% Loop around Participants to Tune Parameters
for tCt = 1:numPart
    
    % Isolate Data
    partData = squeeze(behArray(tCt, :, : ,:));
    
    % Optimize/Fit Models
    % e-Greedy - Stationary
    [para_eGS, LLeGS, BIC_eGS] = eGreedy_Fit_stat(partData, initialValue, numBlocks, numTrials, numArms);
    % e-Greedy
    [para_eG, LLeG, BIC_eG] = eGreedy_Fit(partData, initialValue, numBlocks, numTrials, numArms);
    % Win-Stay, Lose-Shift
    [para_WSLS, LLWSLS, BIC_WSLS] = WSLS_Fit(partData, initialValue, numBlocks, numTrials, numArms);
    % Gradient
    [para_Gr, LLGr, BIC_Gr] = gradient_Fit(partData, initialValue, numBlocks, numTrials, numArms);

    % Save Fitted Parameters and LL
    humanParam.eGS(tCt) = para_eGS;
    humanParam.eG(tCt,1:2) = para_eG(1:2);
    humanParam.Gr(tCt) = para_Gr;
    humanParam.WSLS(tCt) = para_WSLS;

    % Save LL
    llFit.eGS(tCt) = LLeGS; 
    llFit.eG(tCt) = LLeG; 
    llFit.Gr(tCt) = LLGr;
    llFit.WSLS(tCt) = LLWSLS;

    % Save BIC
    BICFit.eGS(tCt) = BIC_eGS;
    BICFit.eG(tCt) = BIC_eG;
    BICFit.Gr(tCt) = BIC_Gr;
    BICFit.WSLS(tCt) = BIC_WSLS;

end

% Save Model Parameters + LL + BIC
save('./Matrices/fittedParameters.mat','humanParam')
save('./Matrices/fittedLL.mat','llFit')
save('./Matrices/fittedBIC.mat','BICFit')

%% Step 3 - Performance Simulation - Mean
% Load parameters and LL
load('./Matrices/fittedParameters.mat');
load('./Matrices/fittedLL.mat');

% Loop Around Fitted Parameters
for pSCt = 1:numPart
    
    % Generate Reward Distributions for Both Arms
    rewVal = rewShuffle(armRewards,numTrials,numBlocks);
    
    % eGreedy - Stationary
    [modCho.eGS(pSCt,:,:), modRew.eGS(pSCt,:,:)] = eGreedy_AS_stat(humanParam.eGS(pSCt),...
        rewVal, initialValue, numBlocks, numTrials, numArms);
    
    % eGreedy
    [modCho.eg(pSCt,:, :), modRew.eG(pSCt,:, :)] = eGreedy_AS(humanParam.eG(pSCt,1:2),...
        rewVal, initialValue, numBlocks, numTrials, numArms);
    
    % Gradient
    [modCho.Gr(pSCt,:, :), modRew.Gr(pSCt, :, :)] = gradient_AS(humanParam.Gr(pSCt),...
        rewVal, initialValue, numBlocks, numTrials, numArms);
    
    % WSLS
    [modCho.WSLS(pSCt,:, :), modRew.WSLS(pSCt, :, :)] = WSLS_AS(humanParam.WSLS(pSCt),...
        rewVal, initialValue, numBlocks, numTrials, numArms);

end 

% Compute Participant Average
partAvg(:, 1) = mean(behArray(:,2,:,:),[3,4]); % Human Average
partAvg(:, 2) = mean(modRew.eGS, [2,3]); % eGreedy Stationary Model Average
partAvg(:, 3) = mean(modRew.eG, [2,3]); % eGreedy Model Average
partAvg(:, 4) = mean(modRew.WSLS, [2,3]); % Gradient Model Average
partAvg(:, 5) = mean(modRew.Gr, [2,3]); % Gradient Model Average

% Compute Overall Averages
behAvg = mean(partAvg,1);

% Compute 95% CIs
behCI = tinv(.975, numPart) * (std(partAvg) / sqrt(numPart));

% Average Performance Figure
x = 1:5;
figure
bp = bar(x,behAvg);
bp.FaceColor = 'flat'; %Needed to change  barplot colors
hold on
% For Individuals
for plotCount3 = 1:5
    % Change Bar Colors
    bp.CData(plotCount3,:) = plotColors2(plotCount3, :);
    % Plot Scattter
    scatter(plotCount3,partAvg(:,plotCount3), 'jitter', 'on', 'jitterAmount',...
        0.1, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor',...
        plotColors2(plotCount3,:), 'MarkerFaceAlpha', .5);
end
% For Error bars
errorbar(x, behAvg', -behCI, behCI, color = [0, 0, 0], LineStyle = 'none', ....
    LineWidth=1.5, CapSize=15);
% Figure Properties
ylim([0, .7]);
ylabel('Wins');
xticks([1, 2, 3, 4, 5]);
xticklabels({'Behavioural', 'eGreedy-S', 'eGreedy', 'WSLS', 'Gradient'})
ax = gca;
ax.FontSize = 12;
ax.FontName = 'Times';
ax.LineWidth = 1;
ax.Box = 'off';
% Save Figure
set(gcf, 'PaperUnits', 'inches');x_width=12 ;y_width=5;
set(gcf, 'PaperPosition', [0 0 x_width y_width]);
print('./Figures/Fig3_ModelPerf', '-dtiff', '-r500');

%% Step 4 - Best Fit Model
% Get BIC Values
load('./Matrices/fittedBIC.mat')

% Generate Likelihood for Baseline model
LLBase = zeros([numBlocks, numTrials]) + .5;
LLSumBase = sum(log(LLBase ),'all');

% Generate BIC for Chance Model
BICBase(1:numPart) = -2*LLSumBase + log(numBlocks*numTrials) * 0;

% Compute Pseudo R2 using Chance as Baseline
% Pseudo R2 Formula: 1 - BIC(non-baseline) / BIC(baseline)
BICPart(:, 1) = 1 - (BICFit.eGS ./ BICBase); % eGreedy Stationary Model
BICPart(:, 2) = 1 - (BICFit.eG ./ BICBase); % eGreedy Model
BICPart(:, 3) = 1 - (BICFit.WSLS ./ BICBase); % Gradient Model
BICPart(:, 4) = 1 - (BICFit.Gr ./ BICBase); % Gradient Model

% Compute Overall Averages
BICAvg = mean(BICPart, 1);

% Compute 95% CIs
BIC_CI = tinv(.975, numPart) * (std(BICPart) / sqrt(numPart));

% Find Best Fitting Model per BIC
[~, bestFitModel] = max(BICPart, [], 2);

% Average Performance Figure
x = 1:4; % needed for plotting
figure
bp2 = bar(x, BICAvg);
% Change Bar Colors
bp2.FaceColor = 'flat';
hold on
% For Individuals
for plotCount4 = 1:4
    % Change Bar Colors
    bp2.CData(plotCount4, :) = plotColors(plotCount4, :);
    scatter(plotCount4, BICPart(:, plotCount4), 'jitter', 'on', 'jitterAmount', 0.1,...
        'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', plotColors(plotCount4, :),...
        'MarkerFaceAlpha', .6);
    % Count Best Fit Model and Print above Bar
    bestFit = sum(bestFitModel==plotCount4);
    text(plotCount4-.3, .9, strcat('best fit =', {' '}, num2str(bestFit)))
end
% For Errorbars
er = errorbar(x, BICAvg, -BIC_CI, BIC_CI, color = [0, 0, 0], ...
    LineStyle = 'none', LineWidth=1.5, CapSize=15);
% Figure Properties
ylim([-.2, 1]);
ylabel('R^2');
xticks([1, 2, 3, 4]);
xticklabels({'eGreedy-S', 'eGreedy', 'WSLS', 'Gradient'})
ax = gca;
ax.FontSize = 12;
ax.FontName = 'Times';
ax.LineWidth = 1;
ax.Box = 'off';
% Save Fig
set(gcf, 'PaperUnits', 'inches');x_width=12 ;y_width=5;
set(gcf, 'PaperPosition', [0 0 x_width y_width]);
print('./Figures/Fig4_BestFit','-dtiff','-r500');

%% Negative Log Likelihood - Manual search
clc;clear; close all;

numTrials = 20;
numBlocks = 5;
numArms = 2;
initialValue = 0;
armRewards = [.6, .1];

% Set Learning Rate Parameter
learningRate_Sim = .5;

% Shuffle Reward
rewVal = rewShuffle(armRewards,numTrials,numBlocks);

[choices, rewards] = gradient_AS(learningRate_Sim, rewVal, initialValue, ...
    numBlocks, numTrials, numArms);

% Junk
behDat(1, :, :) = choices;
behDat(2, :, :) = rewards;
[learningRate_Fit] = gradient_Fit(behDat, initialValue, numBlocks, numTrials, numArms);

%% Manual Parameter Fitting
numPar = 100;
% Loop around parameters
lrArray = linspace(.01, 2,numPar);

for pCt = 1:numPar

    learningRate = lrArray(pCt);

    % Set up LL Array
    ll = zeros(numBlocks, numTrials);
    
    %Initialize Bandit Values
    
    % Extract Choices and Reward
    choiceObs = squeeze(choices(1, :, :));
    rewardObs = squeeze(rewards(1, :, :));

    for block = 1:numBlocks

        H = zeros(numArms, 1) + initialValue;
        
        % Start timer and reward
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
    llSum = -sum(log(ll), 'all');
    
    % assign to array
    llArray(pCt) = llSum;

end

% Figure for Plotting
figure; 
plot(lrArray, llArray);
hold on
scatter(learningRate_Fit, min(llArray),'filled')
scatter(learningRate_Sim, min(llArray),'filled')
% Figure Properties
legend({'Manual Line', 'Fitted', 'Simulated'})
ylabel('Neg Log Likelihood');
xlabel('Parameter Values');
ax = gca;
ax.FontSize = 12;
ax.FontName = 'Times';
ax.LineWidth = 1;
ax.Box = 'off';
% Save Figure
set(gcf, 'PaperUnits', 'inches');x_width=12 ;y_width=5;
set(gcf, 'PaperPosition', [0 0 x_width y_width]);
print('./Figures/Fig5_ManualFit','-dtiff','-r500');
