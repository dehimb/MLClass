function plotData(X, y)
positives = find(y == 1);
negatives = find(y == 0);
hold on
scatter(X(positives,1), X(positives,2));
scatter(X(negatives,1), X(negatives,2));
hold off
end
