%% Assignment 2: working with EEG data

load("eeg_data_assignment_2.mat") % loading the data into the workspace
%% 2. What is the mean EEG voltage at 0.1 seconds for occipital channels
%%(i.e., channels whose name contains the letter "O")? And for frontal
%%channels (i.e., channels whose name contains the letter "F")?
Index_0_1 = find(times == 0.1); % indices for 0.1 seconds
Index_o = find(contains(ch_names, 'O')); % indices for occipital channels
Index_f = find(contains(ch_names, 'F')); % indices for frontal channels

voltage_O = mean(eeg(Index_o, Index_0_1)); % calculating mean voltage at 0.1s for occipital channels
disp(voltage_O); % mean EEG voltage at 0.1s for occipital channels = -0.1411

voltage_F = mean(eeg(Index_f, Index_0_1)); % calculating mean voltage at 0.1s for frontal channels
disp(voltage_F); % mean EEG voltage at 0.1s for frontal channels = -0.1081
%% 3. Visualize the timecourse of mean EEG voltage across all image conditions & channels (averaged across image conditions)
figure
hold on
for ch = 1:size(eeg, 2) % iterate all channels
    mean_eeg = mean(eeg(:, ch, :), 1); % calculate mean voltage for all times
    mean_eeg = squeeze(mean_eeg); % remove dimensions with length of 1 
    plot(times, mean_eeg) % add mean voltage of that channel to figure
end

% Similarities & differences: Many have similar tendencies (e.g. peaks at
% the same time, especially shortly after 0.0s (stimulus presentation).
% Many also have no peaks, indicating that the recorded regions don't have
% an active role in visual processing(?)
%% 4.i Visualize the timecourse of mean voltage across all image conditions & occipital channels
mean_o = squeeze(mean(eeg(:, Index_o, :), [1 2])); % calculate mean voltage for occipital channels
h1 = plot(times, mean_o, 'b', 'LineWidth', 2); % plot for occipital channels
hold on
%% 4.ii Visualize the timecourse of mean voltage across all image conditions & frontal channels

mean_f = squeeze(mean(eeg(:, Index_f, :), [1 2])); % calculate mean voltage for frontal channels
h2 = plot(times, mean_f, 'r', 'LineWidth', 2); % plot for frontal channels

% Occipital channels show a lot of peaks, frontals only slight
% fluctuations. Possible reason: visual processing happens mainly in
% occipital regions(?)
%% 5.i Visualize the timecourse of mean voltage across all occipital channels for first image condition
mean_o_01 = squeeze(mean(eeg(1, Index_o, :), 2)); % mean voltage of occipital channels image 1
hold on
h3 = plot(times, mean_o_01, 'c', 'LineWidth', 2); % plot image 1

%% 5. Visualize the timecourse of mean voltage across all occipital channels for first image condition
mean_o_02 = squeeze(mean(eeg(2, Index_o, :), 2)); % mean voltage of occipital channels image 2
hold on
h4 = plot(times, mean_o_02, 'm', 'LineWidth', 2); % plot image 2
legend([h1 h2 h3 h4], {'Occipital channels', 'Frontal channels', 'Occipital first image', 'Occipital second image'}); % add legend
xlabel('time (s)'); % add x-axis label
ylabel('EEG voltage') % add y-axis label

% Reasons for similaritiest & differences: No idea tbh
