function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

[U, S, ~] = svd((X' * X) / size(X,1));


% =========================================================================

end
