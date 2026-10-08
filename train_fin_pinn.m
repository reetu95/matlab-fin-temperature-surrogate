%% Physics-informed neural network for a 1D fin
clear; clc; close all;
rng(42);

%% Load data and reuse the baseline's fin-length split
data = readtable('temperature_distribution_data.csv');
baseline = load('fin_surrogate_model.mat');

trainMask = ismember(data.L_m, baseline.trainLengths);
valMask = ismember(data.L_m, baseline.valLengths);
testMask = ismember(data.L_m, baseline.testLengths);

% Dimensionless inputs:
% s = x/L, ell = L/Lref
Lref = 0.070;
X = [data.x_m ./ data.L_m, data.L_m ./ Lref];
Y = data.theta_ratio;

%% Physical parameters: same as the data generator
h = 250;
k = 400;
P = 0.016;
Ac = 1.6e-5;
m2 = h * P / (k * Ac);

%% Network
layers = [
    featureInputLayer(2, 'Normalization', 'none')
    fullyConnectedLayer(32)
    tanhLayer
    fullyConnectedLayer(32)
    tanhLayer
    fullyConnectedLayer(1)
];

net = dlnetwork(layers);

%% Training settings
numIterations = 2000;
batchSize = 256;
learningRate = 0.001;

avgGrad = [];
avgSqGrad = [];
lossHistory = zeros(numIterations, 3);
valHistory = nan(numIterations, 1);

trainRows = find(trainMask);
bestVal = inf;
bestNet = net;

%% Custom training loop
for iteration = 1:numIterations

    % Sample labeled training points
    rows = trainRows(randi(numel(trainRows), 1, batchSize));
    Xbatch = dlarray(single(X(rows, :)'), 'CB');
    Ybatch = dlarray(single(Y(rows)'), 'CB');

    % Sample physics points using training lengths only
    sampledL = baseline.trainLengths( ...
        randi(numel(baseline.trainLengths), 1, batchSize));
    ell = reshape(single(sampledL / Lref), 1, []);

    Xphysics = dlarray([rand(1, batchSize, 'single'); ell], 'CB');
    Xbase = dlarray([zeros(1, batchSize, 'single'); ell], 'CB');
    Xtip = dlarray([ones(1, batchSize, 'single'); ell], 'CB');

    [loss, gradients, dataLoss, physicsLoss, boundaryLoss] = ...
        dlfeval(@pinnLoss, net, Xbatch, Ybatch, ...
                Xphysics, Xbase, Xtip, m2, h, k, Lref);

    [net, avgGrad, avgSqGrad] = adamupdate( ...
        net, gradients, avgGrad, avgSqGrad, ...
        iteration, learningRate);

    lossHistory(iteration, :) = double(gather(extractdata( ...
        [dataLoss, physicsLoss, boundaryLoss])));

    % Select model using validation data, never test data
    if mod(iteration, 50) == 0
        valPred = minibatchpredict(net, single(X(valMask, :)));
        valMSE = mean((double(valPred(:)) - Y(valMask)).^2);
        valHistory(iteration) = valMSE;

        if valMSE < bestVal
            bestVal = valMSE;
            bestNet = net;
        end

        fprintf('Iteration %d/%d | Loss %.6f | Val MSE %.6f\n', ...
            iteration, numIterations, ...
            double(gather(extractdata(loss))), valMSE);
    end
end

net = bestNet;

%% Final test metrics
prediction = minibatchpredict(net, single(X(testMask, :)));
prediction = double(prediction(:));
actual = Y(testMask);

MAE = mean(abs(prediction - actual));
RMSE = sqrt(mean((prediction - actual).^2));

fprintf('\nBaseline MAE: %.6f | PINN MAE: %.6f\n', ...
        baseline.MAE, MAE);
fprintf('Baseline RMSE: %.6f | PINN RMSE: %.6f\n', ...
        baseline.RMSE, RMSE);

%% Plot loss components
figure;
semilogy(max(lossHistory, 1e-12), 'LineWidth', 1);
xlabel('Iteration');
ylabel('Unweighted loss component');
legend('Data', 'Physics', 'Boundary');
title('PINN Training Loss');
grid on;
exportgraphics(gcf, 'pinn_training_loss.png', 'Resolution', 200);

%% Compare theory, baseline, and PINN on unseen lengths
lengths = sort(baseline.testLengths);
selected = lengths([1, 5, 10]);

figure('Position', [100, 100, 1200, 400]);
tiledlayout(1, 3);

for i = 1:numel(selected)
    mask = data.L_m == selected(i);
    x = data.x_m(mask);

    pinnPred = minibatchpredict(net, single(X(mask, :)));

    % Baseline uses its original input scaling
    originalX = [x, data.L_m(mask)];
    baselineX = (originalX - baseline.xmin) ./ baseline.xrange;
    baselinePred = minibatchpredict(baseline.net, baselineX);

    nexttile;
    plot(x * 1000, Y(mask), 'k-', 'LineWidth', 1.5);
    hold on;
    plot(x * 1000, baselinePred, 'b--', 'LineWidth', 1.5);
    plot(x * 1000, pinnPred, 'r:', 'LineWidth', 2);
    hold off;
    xlabel('Position (mm)');
    ylabel('\theta / \theta_b');
    title(sprintf('Unseen L = %.1f mm', selected(i) * 1000));
    legend('Theory', 'Baseline', 'PINN', 'Location', 'best');
    grid on;
end

exportgraphics(gcf, 'pinn_vs_baseline.png', 'Resolution', 200);

%% Save separately
trainLengths = baseline.trainLengths;
valLengths = baseline.valLengths;
testLengths = baseline.testLengths;

save('fin_pinn_model.mat', 'net', 'Lref', 'm2', 'h', 'k', ...
     'MAE', 'RMSE', 'lossHistory', 'valHistory', ...
     'trainLengths', 'valLengths', 'testLengths');

disp('Saved PINN model and plots.');

%% Local loss function -- keep at the end of this script
function [loss, gradients, ld, lp, lb] = pinnLoss( ...
    net, Xdata, Ydata, Xphysics, Xbase, Xtip, m2, h, k, Lref)

    % 1. Match analytical training data
    uData = forward(net, Xdata);
    ld = mean((uData - Ydata).^2, 'all');

    % 2. Fin equation in dimensionless position s=x/L:
    % u_ss - (m*L)^2*u = 0
    u = forward(net, Xphysics);

    firstGrad = dlgradient(sum(u, 'all'), Xphysics, ...
                           'EnableHigherDerivatives', true);
    us = firstGrad(1, :);

    secondGrad = dlgradient(sum(us, 'all'), Xphysics, ...
                            'EnableHigherDerivatives', true);
    uss = secondGrad(1, :);

    L = Xphysics(2, :) * Lref;
    residual = uss - m2 .* L.^2 .* u;
    lp = mean(residual.^2, 'all');

    % 3. Base: u(0)=1
    uBase = forward(net, Xbase);
    baseLoss = mean((uBase - 1).^2, 'all');

    % Tip: u_s(1) + (h*L/k)*u(1)=0
    uTip = forward(net, Xtip);
    tipGrad = dlgradient(sum(uTip, 'all'), Xtip, ...
                         'EnableHigherDerivatives', true);

    Ltip = Xtip(2, :) * Lref;
    tipResidual = tipGrad(1, :) + (h / k) .* Ltip .* uTip;
    tipLoss = mean(tipResidual.^2, 'all');

    lb = baseLoss + tipLoss;

    % Weighted combined objective
    loss = ld + lp + 10 * lb;

    gradients = dlgradient(loss, net.Learnables);
end