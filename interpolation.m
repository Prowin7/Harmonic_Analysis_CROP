function [fitresult, gof, interpolatedX, interpolatedY] = fit2(xData, yData)
    % Prepare the data
    [xData, yData] = prepareCurveData(xData, yData);

    % Set up fit type and options
    ft = 'pchipinterp';
    opts = fitoptions('Method', 'PchipInterpolant');
    opts.ExtrapolationMethod = 'pchip';
    opts.Normalize = 'on';

    % Fit model to data
    [fitresult, gof] = fit(xData, yData, ft, opts);

    % Interpolate 96 points across full index (0 to 107)
    interpolatedX = linspace(0, 107, 108); %  interpolatedX = linspace(0, 59, 60)
    interpolatedY = feval(fitresult, interpolatedX);

    % Plot the result
    figure('Name', 'Continuous Curve Fit');
    plot(xData, yData, 'or', 'MarkerFaceColor','yellow'); hold on;
    plot(interpolatedX, interpolatedY, 'r.-', 'LineWidth', 1.5, 'MarkerSize', 10);
    legend('Grouped Data', 'Interpolated Curve (106 points)', 'Location', 'NorthEast');
    xlabel('Group-Aligned X Index');
    ylabel('Average VH\_dB');
    title('Curve Fit Using PCHIP');
    grid on;
end
