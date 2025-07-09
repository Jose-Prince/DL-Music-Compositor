% Load and process data
midi = readmidi('/MATLAB Drive/DL-Music-Compositor/giantmidi-piano/Bach_Prelude_and_Fugue_in_A-flat_major_BWV862_gCL5Zvnt0TU_a.mid');
notes = midiInfo(midi, 0);

X = notes(1:end-1);
Y = notes(2:end);

% Neunoral network
layers = [
    sequenceInputLayer(1)
    lstmLayer(100)
    fullyConnectedLayer(88)
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'MiniBatchSize',64, ...
    'SequenceLength','longest', ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress');

% Training
net = trainingNetwork(XTrain, YTrain, layers, options);

% Generate music
note = 60;
sequence = [];

for i = 1:100
    [pred, scores2label] = classify(net, note);
    note = pred;
    sequence = [sequence; note];
end
