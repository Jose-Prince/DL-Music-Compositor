% Load and process data
midi = readmidi('/MATLAB Drive/DL-Music-Compositor/giantmidi-piano/Bach_Prelude_and_Fugue_in_A-flat_major_BWV862_gCL5Zvnt0TU_a.mid');
notes = midiInfo(midi, 0);

pitches = notes(:, 3);
pitches = pitches - 20;
pitches = pitches(pitches >= 1 & pitches <= 88);

X = pitches(1:end-1)';
Y = pitches(2:end)';

XTrain = {X};
YTrain = {categorical(Y)};

numClasses = numel(unique(Y));

% Neunoral network
layers = [
    sequenceInputLayer(1)
    lstmLayer(100)
    fullyConnectedLayer(numClasses)
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'MiniBatchSize',64, ...
    'SequenceLength','longest', ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress');

% Training
net = trainNetwork(XTrain, YTrain, layers, options);

% Generate music
note = 40;
sequence = [];

for i = 1:100
    input = {note};
    pred = classify(net, input);
    note = double(pred{1});
    sequence(end+1) = note;
end
