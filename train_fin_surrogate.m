%% Train and evaluate a fin-temperature neural network
clear;
clc;
close all;
rng(42);

%% Load the generated data
data = readtable('temperature_distribution_data.csv');

X = [data.x_m, data.L_m];
Y = data.theta_ratio;

%% Split by fin length: 30 training, 10 validation, 10 test
lengths = unique(data.L_m);
order = randperm(numel(lengths));

% Keep the shortest and longest lengths in training
interior = order(order ~= 1 & order ~= numel(lengths));
trainLengths = lengths([1, numel(lengths), interior(1:28)]);
valLengths = lengths(interior(29:38));
testLengths = lengths(interior(39:48));

trainMask = ismember(data.L_m, trainLengths);
valMask = ismember(data.L_m, valLengths);
testMask = ismember(data.L_m, testLengths);

%% Normalize using training data only
xmin = min(X(trainMask, :), [], 1);
xrange = max(X(trainMask, :), [], 1) - xmin;
xrange(xrange == 0) = 1;

Xs = (X - xmin) ./ xrange;

%% Small neural network
layers = [
    featureInputLayer(2, 'Normalization', 'none')
    fullyConnectedLayer(32)
    tanhLayer
    fullyConnectedLayer(32)
    tanhLayer
    fullyConnectedLayer(1)
];

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 400, ...
    'MiniBatchSize', 128, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {Xs(valMask, :), Y(valMask)}, ...
    'ValidationFrequency', 24, ...
    'ValidationPatience', 20, ...
    'OutputNetwork', 'best-validation', ...
    'ExecutionEnvironment', 'cpu', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

%% Train using mean squared error
net = trainnet(Xs(trainMask, :), Y(trainMask), ...
              layers, 'mse', options);

%% Evaluate on held-out lengths
predicted = minibatchpredict(net, Xs(testMask, :));
actual = Y(testMask);

MAE = mean(abs(predicted - actual));
RMSE = sqrt(mean((predicted - actual).^2));

fprintf('Test MAE:  %.6f\n', MAE);
fprintf('Test RMSE: %.6f\n', RMSE);

%% Compare three held-out fin lengths
selectedLengths = sort(testLengths);
selectedLengths = selectedLengths([1, 5, 10]);

figure;
tiledlayout(1, 3);

for i = 1:numel(selectedLengths)
    L = selectedLengths(i);
    mask = data.L_m == L;

    x = data.x_m(mask);
    theory = Y(mask);
    estimate = minibatchpredict(net, Xs(mask, :));

    nexttile;
    plot(x * 1000, theory, '-', 'LineWidth', 1.5);
    hold on;
    plot(x * 1000, estimate, '--', 'LineWidth', 1.5);
    hold off;

    xlabel('Position (mm)');
    ylabel('\theta / \theta_b');
    title(sprintf('Unseen L = %.1f mm', L * 1000));
    legend('Theory', 'Neural network', 'Location', 'best');
    grid on;
end

exportgraphics(gcf, 'predictions_vs_theory.png', ...
               'Resolution', 200);

%% Save model, scaling values, and evaluation results
save('fin_surrogate_model.mat', ...
     'net', 'xmin', 'xrange', ...
     'trainLengths', 'valLengths', 'testLengths', ...
     'MAE', 'RMSE');

disp('Saved model and prediction comparison plot.');