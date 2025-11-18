%% Assignment 5
% This script serves to train and test different EEG encoding models and 
% compare them with each other. In the first task, encoding models are 
% trained with different amounts of training image conditions (250, 1000,
% 10000 and 16540) and tested on the same test data. The prediction
% accuracies are then plotted and compared.
% In the second task, the effect of the amount of deep neural network (DNN) 
% features on encoding accuracy will be investigated by training EEG encoding 
% models using different amounts of DNN features (25, 50, 75 and 100),
% testing them on the same test data and plotting the prediction
% accuracies.

%% Task 1
% Train 4 encoding models with either 250, 1000, 10000 or 16540 training
% image conditions. Test them on the same data. Plot their predicition
% accuracies.

% Load the data
load("C:\Users\Irma\Downloads\data_assignment_5.mat")

% Define training sizes
trainSizes = [250, 1000, 10000, 16540]; % define amounts of training image conditions
rng(0); % sets seed to 0 for reproducable results

% Get the data dimension sizes
[numTrials, numChannels, numTime] = size(eeg_train);
numFeatures = size(dnn_train, 2);

numSizes = numel(trainSizes); % number of encoding models to be trained
colors = lines(numSizes);

% Create storage for results
R_all = zeros(numSizes, numChannels, numTime); % correlation per size/channel/time
meanR_all = zeros(numSizes, numTime); % mean correlation per size/time

% Loop over training sizes
for i = 1:numSizes
    N = trainSizes(i);
    fprintf('Training with N = %d images (%d%d)\n', N, i, numSizes);

    % sample subset of training trials
    idx = randperm(numTrials, N); % generate random indices for the amount
    % of training trials (without replacement)
    dnn_train_sub = dnn_train(idx, :); % indexing dnn training data
    eeg_train_sub = eeg_train(idx, :, :); % indexing eeg training data

    % train models
    % Store weights and intercepts
    W = zeros(numFeatures, numChannels, numTime); % regression coefficients
    b = zeros(numChannels, numTime);              % intercepts

    % Progressbar parameters
    totalModels = numChannels * numTime;
    modelCount = 0;

    % Train a linear regression independently for each EEG channel and time
    % point
    for ch = 1:numChannels
        for t = 1:numTime
        
            % Extract EEG responses for this channel/time over all trials
            y = eeg_train_sub(:, ch, t);   
        
            % Fit linear regression: y = DNN*w + b
            mdl = fitlm(dnn_train_sub, y);
        
            % Save parameters
            W(:, ch, t) = mdl.Coefficients.Estimate(2:end); % weights
            b(ch, t)    = mdl.Coefficients.Estimate(1);     % intercept

            % Update progress bar
            modelCount = modelCount + 1;
            fprintf('\rTraining models: %d / %d (%.1f%%)', ...
                modelCount, totalModels, 100*modelCount/totalModels);

        end
    end



%% Use the trained models to predict the EEG responses for the test images

[numTest, numFeatures] = size(dnn_test);

eeg_test_pred = zeros(numTest, numChannels, numTime); % predictions

for ch = 1:numChannels
    for t = 1:numTime
        eeg_test_pred(:, ch, t) = dnn_test * W(:, ch, t) + b(ch, t);
    end
end



%% Compute the prediction accuracy using Pearson's correlation
% If you are not sure what a correlation is, don't worry! For now, all you 
% need to know is that it is a measure of similarity between two vectors,
% where a correlation score of 0 indicates no similarity, and a correlation
% score of 1 indicates perfect similarity.

[Ntest, Nchannels, Ntime] = size(eeg_test);

% Preallocate correlation matrix
R = zeros(Nchannels, Ntime);

for ch = 1:Nchannels
    for t = 1:Ntime
        % Get test responses across images
        real_vec = squeeze(eeg_test(:, ch, t));
        pred_vec = squeeze(eeg_test_pred(:, ch, t));

        % Compute Pearson correlation
        R(ch, t) = corr(real_vec, pred_vec, 'Type', 'Pearson');
    end
end



%% Plot the prediction accuracy over time, averaged across channels

% Average the correlation across channels for each model
R_all(i, :, :) = R;
meanR_all(i, :) = squeeze(mean(R, 1));

end

% Plot the mean correlation over time
figure;
hold on; % ensure that all lines are added to the same plot


% loop for creating 4 curves
for p = 1:numSizes
    plot(1:numTime, meanR_all(p, :), ...
    'Color', colors(p, :), ..." + ...
    'LineWidth', 2, ...
    ...
    'DisplayName', sprintf('%d train images', trainSizes(p)));
end

xlabel('Time (seconds)');
xticks(1:numTime); xticklabels(times);
ylabel('Mean Pearson Correlation');
title('Prediction Accuracy Over Time (average across channels)');
legend('show', 'Location', 'best');
set(gca, 'FontSize', 18);

hold off

%% Task 2
% Train 4 encoding models with either 25, 50, 75 or 100 
% DNN features. Test them on the same data. Plot their predicition
% accuracies.

% Define the number of DNN-features
featureSizes = [25, 50, 75, 100];
rng(0);

% Define the data dimension size 
[numTrials, numChannels, numTime] = size(eeg_train);
totalFeatures = size(dnn_train, 2);

numSizes = numel(featureSizes); % number of models to be trained
colors = lines(numSizes);

% Storage for correlations
R_all_features = zeros(numSizes, numChannels, numTime);
meanR_all_features = zeros(numSizes, numTime);

for i = 1:numSizes
    F = featureSizes(i); % number of DNN features
    fprintf('Training with F = %d features (%d/%d)\n', F, i, numSizes);
    
    % select random indices for DNN features
    idx_feat = randperm(totalFeatures, F);
    dnn_train_sub = dnn_train(:, idx_feat);
    dnn_test_sub = dnn_test(:, idx_feat);
    eeg_train_sub = eeg_train;
    
    % define number of features
    currentFeatures = F;
    
    % train models
    W = zeros(currentFeatures, numChannels, numTime);
    b = zeros(numChannels, numTime);
    
    for ch = 1:numChannels
        for t = 1:numTime
            y = eeg_train_sub(:, ch, t);
            mdl = fitlm(dnn_train_sub, y);
            
            % Save parameters
            W(:, ch, t) = mdl.Coefficients.Estimate(2:end); % weights
            b(ch, t)    = mdl.Coefficients.Estimate(1);     % intercept
        end
    end
    
    % predict on test data
    for ch = 1:numChannels
        for t = 1:numTime
            eeg_test_pred(:, ch, t) = dnn_test_sub * W(:, ch, t) + b(ch, t);
        end
    end
    [Ntest, Nchannels, Ntime] = size(eeg_test);
    R = zeros(Nchannels, Ntime);
    for ch = 1:Nchannels
        for t = 1:numTime
            real_vec = squeeze(eeg_test(:, ch, t));
            pred_vec = squeeze(eeg_test_pred(:, ch, t));
            R(ch, t) = corr(real_vec, pred_vec, 'Type', 'Pearson');
        end
    end

    % save results
    R_all_features(i, :, :) = R;
    meanR_all_features(i, :) = squeeze(mean(R, 1));
end

% plot
figure;
hold on;

for p = 1:numSizes
    plot(1:numTime, meanR_all_features(p, :), 'Color', colors(p, :), 'LineWidth', 2, ...
        'DisplayName', sprintf('%d DNN features', featureSizes(p)));
end

xlabel('Time (seconds)');
xticks(1:numTime); xticklabels(times);
ylabel('Mean Pearson Correlation');
legend('show', 'Location', 'best');
title('Prediction Accuracy Over Time (Effect of DNN Features)');
set(gca, 'FontSize', 16);

hold off
            

