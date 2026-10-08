%% Generate analytical temperature data for a single fin
clear;
clc;
close all;

% Physical parameters from your Python study
P = 0.0160;       % Fin perimeter (m)
h = 250;         % Convection coefficient (W/m^2/K)
k = 400;         % Thermal conductivity (W/m/K)
Ac = 1.6000e-5;  % Cross-sectional area (m^2)

m = sqrt(h * P / (k * Ac));

% 50 lengths, with 100 positions per length = 5000 rows
L_values = linspace(0.010, 0.070, 50);
numPositions = 100;

data = zeros(numel(L_values) * numPositions, 3);

figure;
hold on;

for i = 1:numel(L_values)
    L = L_values(i);
    x = linspace(0, L, numPositions);

    % Analytical solution with convection at the fin tip
    thetaRatio = ...
        (cosh(m * (L - x)) + ...
        (h / (m * k)) * sinh(m * (L - x))) ./ ...
        (cosh(m * L) + ...
        (h / (m * k)) * sinh(m * L));

    rows = (i - 1) * numPositions + (1:numPositions);
    data(rows, :) = [x(:), repmat(L, numPositions, 1), ...
        thetaRatio(:)];

    % Plot selected lengths to keep the figure readable
    if ismember(i, [1, 17, 34, 50])
        plot(x * 1000, thetaRatio, ...
            'LineWidth', 1.5, ...
            'DisplayName', sprintf('L = %.1f mm', L * 1000));
    end
end

xlabel('Position along fin (mm)');
ylabel('\theta / \theta_b');
title('Analytical Fin Temperature Profiles');
legend('Location', 'best');
grid on;
hold off;

% Save dataset using simple MATLAB column names
finData = array2table(data, ...
    'VariableNames', {'x_m', 'L_m', 'theta_ratio'});

writetable(finData, 'temperature_distribution_data.csv');
exportgraphics(gcf, 'theory_curves.png', 'Resolution', 200);

disp(finData(1:5, :));
fprintf('Saved %d rows to temperature_distribution_data.csv\n', ...
    height(finData));