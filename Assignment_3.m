%% Assignment 3
%{ This scipt has the purpose of creating a condition list for an
%experiment with a 2x3 factorial design. The two factors are FAMILIARITY
%with the levels "familiar" and "unfamiliar" and EMOTION with the levels
%"positive", "neutral" and "negative". The experiment consists of six
%blocks and 60 participants are taking part. The order of blocks should be
%counterbalanced, although the two blocks for each level of EMOTION should
%be presented right after each other.}%

%% 1. Creating string arrays for each factor
familiarity = {"familiar", "unfamiliar"};
emotion = {"positive", "neutral", "negative"};
n_subjects = 60;

%% 2. Creating index lists for indexing emotion & familiarity in Block 1

%{ These lists ensure that 1/3 of participants starts with each level of 
% EMOTION and that half of the participants start with each level of 
% FAMILIARITY.
index_emo_sorted = [
    repmat(1, 1, 20),
    repmat(2, 1, 20),
    repmat(3, 1, 20)
    ];
index_emo = index_emo_sorted(randperm(n_subjects));

index_fam_sorted = [
    repmat(1, 1, 30),
    repmat(2, 1, 30)
    ];
index_fam = index_fam_sorted(randperm(n_subjects));

%% 3. Creating empty table

stimlist = table('Size', [n_subjects 7], ...
    'VariableTypes', {'double', 'string', 'string', 'string', 'string', 'string', 'string'}, ...
    'VariableNames', {'N', 'Block_1', 'Block_2', 'Block_3', 'Block_4', 'Block_5', 'Block_6'});

stimlist.N = (1:n_subjects)';

%% Generating the blocks and adding them to the table

for i = 1:n_subjects

    % creating randomized sequence of emotion indices
    index_emo1 = index_emo(i);
    index_rem_emo_s = setdiff(1:3, index_emo1);
    index_rem_emo = index_rem_emo_s(randperm(2));
    emo_index_sequence = [index_emo1, index_rem_emo];

    % creating sequence of familiarity indices depending on 1st familiarity
    if index_fam(i) == 1
        index_fam_sequence = [1, 2];
    else
        index_fam_sequence = [2, 1];
    end

    % generating the blocks based on the indices
    stimlist{i, 'Block_1'} = emotion{emo_index_sequence(1)} + " and " + familiarity{index_fam_sequence(1)};
    stimlist{i, 'Block_2'} = emotion{emo_index_sequence(1)} + " and " + familiarity{index_fam_sequence(2)};
    stimlist{i, 'Block_3'} = emotion{emo_index_sequence(2)} + " and " + familiarity{index_fam_sequence(1)};
    stimlist{i, 'Block_4'} = emotion{emo_index_sequence(2)} + " and " + familiarity{index_fam_sequence(2)};
    stimlist{i, 'Block_5'} = emotion{emo_index_sequence(3)} + " and " + familiarity{index_fam_sequence(1)};
    stimlist{i, 'Block_6'} = emotion{emo_index_sequence(3)} + " and " + familiarity{index_fam_sequence(2)};
end

