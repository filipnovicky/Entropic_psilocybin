%% Approximate entropy analysis for EEGLAB datasets
%{
This code works as follows:
1) Plug in your datasets to EEGLAB
2) Define m  and r paraneters based on your preference
3) Press run

Tip:
If the code takes too long, reduce the sampling rate

Description:
m = the length of the time-steps that will be analyzed and compared with
each other
r = tolerance for the comparison among different time series of m

Output:
all_ents.apen = saves entropy per electrode
all_ents.labels = electrode labels
all_ents.name = name of the analyzed dataset
%}


tic;
electrode_labels = {'FP1';'Fz';'F3';'F7';'FT9';'FC5';'FC1';'C3';'T7';
    'TP9';'CP5';'CP1';'Pz';'P3';'P7';'O1';'Oz';'O2';'P4';'P8';'TP10';
    'CP6';'CP2';'Cz';'C4';'T8';'FT10';'FC6';'FC2';'F4';'F8';'Fp2';'AF7';
    'AF3';'AFz';'F1';'F5';'FT7';'FC3';'C1';'C5';'TP7';'CP3';'P1';'P5';
    'PO7';'PO3';'POz';'PO4';'PO8';'P6';'P2';'CPz';'CP4';'TP8';'C6';'C2';
    'FC4';'FT8';'F6';'AF8';'AF4';'F2';'Iz'};

all_ents = struct();
all_ents.apen = zeros(64, length(ALLEEG));
all_ents.labels = electrode_labels;
all_ents.name = cell(1, length(ALLEEG));

m = 2;
r = 0.2;
fs = EEG.srate;
windowsize = 5;

for i = 1:length(ALLEEG)
    df = ALLEEG(i);
    apen = approximateEntropyMultiEEG(df.data,m,r,fs,windowsize);
    
    % Get the channel labels
    chan_labels = channel_labels_to_matrix(df.chanlocs);
    all_ents.name{i} = df.filename;
    fprintf('Processing electrodes in dataset %s..\n ',df.filename)
    all_ents.apen(:,i) = match_labels_and_data(chan_labels,electrode_labels,apen);
end

try
    folderPath = 'D:\PhD\psiconnnect_EEG\apen';
    fileName = 'apen.mat';
    fullPath = fullfile(folderPath, fileName);
    save(fullPath, 'all_ents');
catch
    warning('all_ents not saved')
end

toc;

function labels_matrix = channel_labels_to_matrix(chanlocs)
    % Extract labels from chanlocs struct
    labels = {chanlocs.labels};
    
    % Create a 64x1 cell array filled with empty strings
    labels_matrix = cell(64, 1);
    labels_matrix(:) = {''};
    
    % Fill the matrix with available labels
    num_labels = min(length(labels), 64);
    labels_matrix(1:num_labels) = labels(1:num_labels);
end


function matched_data = match_labels_and_data(labels_matrix, electrode_labels, data)
    % Initialize output array with NaNs
    matched_data = nan(size(electrode_labels));
    
    % Convert labels_matrix to cell array if it's a character array
    if ischar(labels_matrix)
        labels_matrix = cellstr(labels_matrix);
    end
    
    % Ensure electrode_labels is a cell array
    if ischar(electrode_labels)
        electrode_labels = cellstr(electrode_labels);
    end
    
    % Find matching indices
    [is_member, idx] = ismember(electrode_labels, labels_matrix);
    
    % Assign corresponding data values
    matched_data(is_member) = data(idx(is_member));
end

function apen = approximateEntropyMultiEEG(X, m, r, fs, windowSize)
% Compute Approximate Entropy (ApEn) for multiple EEG channels simultaneously
%
% Inputs:
%   X: matrix where each row is an EEG channel
%   m: embedding dimension
%   r: tolerance (if not provided, it's calculated as 0.2 times the standard deviation of each window)
%   fs: sampling frequency (Hz)
%   windowSize: size of the sliding window in seconds (default: 5s)
%
% Output:
%   apen: matrix of Approximate Entropy values for each channel and window

[numChannels, N] = size(X);
if nargin < 5 || isempty(windowSize)
    windowSize = 5; % default 5-second window
end

windowSamples = round(windowSize * fs);
numWindows = floor(N / windowSamples);

apen = zeros(numChannels, numWindows);

% Initialize progress tracking
fprintf('Processing windows: 00%%');

for win = 1:numWindows
    startIdx = (win-1) * windowSamples + 1;
    endIdx = win * windowSamples;
    Xwin = X(:, startIdx:endIdx);
    
    if nargin < 3 || isempty(r)
        r = 0.2 * std(Xwin, 0, 2);
    elseif isscalar(r)
        r = r * std(Xwin, 0, 2);
    end

    % Normalize the window
    Xwin = (Xwin - mean(Xwin, 2)) ./ std(Xwin, 0, 2);

    phi = zeros(numChannels, 2);
    
    for i = 1:2
        m_i = m + i - 1;
        C = zeros(numChannels, windowSamples - m_i + 1);
        
        % Create template matrices
        template = zeros(numChannels, m_i, windowSamples - m_i + 1);
        for j = 1:m_i
            template(:, j, :) = Xwin(:, j:windowSamples - m_i + j);
        end
        
        % Count similar patterns using Euclidean distance
        for j = 1:windowSamples - m_i + 1
            dist = squeeze(sqrt(sum((template - template(:, :, j)).^2, 2)));
            C(:, j) = sum(dist <= r, 2) / (windowSamples - m_i + 1);
        end
        
        phi(:, i) = sum(log(C), 2) / (windowSamples - m_i + 1);
    end
    
    apen(:, win) = phi(:, 1) - phi(:, 2);
    
    % Update progress (every 5% or for each window if numWindows < 20)
    if mod(win, max(1, round(numWindows/20))) == 0 || win == numWindows
        fprintf('\b\b\b%02d%%', round(win/numWindows*100));
    end
end

apen = mean(apen,2);
% Finish progress tracking
fprintf('\nApproximate Entropy calculation completed for all channels and windows.\n');
end
