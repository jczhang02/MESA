% Multi-view semi-supervised feature selection with soft label learning and
% tensor low-rank approximation.
% Chengrui Zhang <jczhang@live.it>

clear all;
clc;
close all;

%% main.m

load("./data/NUSWIDE_obj.mat");

X = X;
Y = gt;

num_view = size(X, 2);
for v = 1:num_view
    X{v} = X{v}';
end
N = size(X{1, 1}, 2);
Nl = round(0.2 * N);

k = length(unique(Y));
YYl = zeros(Nl, k);

for i = 1:Nl
    YYl(i, Y(i, 1)) = 1;
end

Final_result_acc = [];
Final_result_nmi = [];
Best_result = [];

options.k = k;
options.nl = Nl;
options.fcm_options = [2; 50; 1; 0];
options.NITER = 18;

alpha = logspace(-2, 2, 5);
beta = logspace(-2, 2, 5);
gamma = logspace(-2, 2, 5);
rho = logspace(-2, 2, 5);
lambda = logspace(-2, 2, 5);

% for a = 1:5
for b = 1:5
    for c = 1:5
        for d = 1:5
            for e = 1:5
                %                 params.alpha = alpha(a);
                params.beta = beta(b);
                params.gamma = gamma(c);
                params.rho = rho(d);
                params.lambda = lambda(e);

                P = MESA(X, YYl, options, params);

                score = [];
                for v = 1:num_view
                    tmp_score = sqrt(diag(P{v} * P{v}'));
                    score = [score; tmp_score];
                end
                [~, fea_idx] = sort(score, 'descend');

                X_multi = [];
                for v = 1:num_view
                    X_multi = [X_multi; X{1, v}];
                end

                Fea_fs = X_multi(fea_idx, :);

                tmp_fea_num = 0.02:0.02:0.12;
                ACC_fs = zeros(6, 50);
                NMI_fs = zeros(6, 50);

                for i = 1:6

                    d_fea = size(X_multi, 1);
                    fea_num = ceil(tmp_fea_num(i) * d_fea);
                    %                     fea_num = ceil(0.2 * d_fea);
                    fea_fs = Fea_fs(1:fea_num, :);
                    MAXiter = 500; % Maximum of iterations for KMeans
                    REPlic = 20; % Number of replications for KMeans
                    class_num=max(gt);
                    idx=[];
                    result=[];

                    parfor ii=1:50
                        idx = kmeans(fea_fs',class_num,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
                        result = ClusteringMeasure(gt, idx);

                        ACC_fs(i, ii) = result(1, 1);
                        NMI_fs(i, ii) = result(1, 2);
                    end
                end

                MeanACC_fs_Multi=mean(ACC_fs, 2);
                MeanNMI_fs_Multi=mean(NMI_fs, 2);

                stdACC_fs_Multi = zeros(6, 1);
                stdNMI_fs_Multi = zeros(6, 1);
                for i = 1:6
                    stdACC_fs_Multi(i, :) = std(ACC_fs(i, :));
                    stdNMI_fs_Multi(i, :) = std(NMI_fs(i, :));                    
                end

                tmp_Final_result_acc = [MeanACC_fs_Multi', stdACC_fs_Multi', struct2array(params)];
                tmp_Final_result_nmi = [MeanNMI_fs_Multi', stdNMI_fs_Multi', struct2array(params)];
                Final_result_acc = [Final_result_acc; tmp_Final_result_acc];
                Final_result_nmi = [Final_result_nmi; tmp_Final_result_nmi];
            end
        end
        save("result_MSRC_v1.mat", "Final_result_acc", "Final_result_nmi");
    end
end
% end

save("result_MSRC_v1.mat", "Final_result_acc", "Final_result_nmi");


