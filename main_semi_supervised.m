clc;
close all;
clear all;

%% load data

data_name = "NUS_WIDE_MLKNN";

load(data_name + ".mat");

% X = X;
% Y = Y;



r = 100;

num_view = size(X, 2);
N = size(X{1}, 2);
Nl = round(0.2 * N);

k = size(Y, 2);
% YYl = zeros(Nl, k);
% for i = 1:Nl
%     YYl(i, Y(i, 1)) = 1;
% end

YYl = Y(1:Nl, :);

options.k = k; %% cluster number
options.fcm_options = [2; 50; 1; 0];
options.NITER = 18;

params.alpha = 1;
params.beta = 1;
params.gamma = 1;
params.rho = 1;
params.lambda = 1;

P = MESA(X, YYl, options, params);

score = [];
for v = 1:num_view
    tmp_score = sqrt(diag(P{v} * P{v}'));
    score = [score; tmp_score];
end
[~, fea_idx] = sort(score, 'descend');

x_cat = [];
for v = 1:num_view
    x_cat = [x_cat; X{1, v}];
end

tmp_matrix = x_cat(fea_idx, :);

matrix = tmp_matrix(1:r, :);
label = Y;








