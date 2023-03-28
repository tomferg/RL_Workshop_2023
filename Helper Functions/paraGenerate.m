function [para] = paraGenerate(numParticipants)

% Randomly Generate Model Parameters

%Greedy Model
para.egreedy(:, 1) = betarnd(1.1, 1.1, [numParticipants,1]); % Epsilon
para.egreedy(:, 2) = betarnd(1.1, 1.1, [numParticipants,1]); % Learning Rate

% eGreedy Model - Stationary
para.egreedy_stat(:, 1) = betarnd(1.1, 1.1, [numParticipants,1]); % Epsilon

% % Chance Model
% para.chance(:,1) = unifrnd(.001, .99, [numParticipants,1]); % Bias

% Gradient Model
para.gradient(:, 1) = gamrnd(1, .5, [numParticipants,1]); % learning rate

% WSLS
para.WSLS(:, 1) = betarnd(1.1, 1.1, [numParticipants,1]); % Win-Stay

end