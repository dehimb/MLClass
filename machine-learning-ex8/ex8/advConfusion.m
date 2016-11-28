function [tp, tn, fp, fn] = advConfusion(predicted, actual)
%% Custom implementation for confusion calculation

% True positives
positive_prediceted = actual(predicted == 1);
tp = sum(positive_prediceted);

% True negatives
false_predicted = actual(predicted == 0);
tn = length(false_predicted) - sum(false_predicted);

% False positives
fp = length(positive_prediceted) - tp;

% False negatives
fn = sum(false_predicted);

end