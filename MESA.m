% MESA: 
% Multi-view fEature selection with Soft label learning and tensor low-rank Approximation
% Chengrui Zhang <jczhang@live.it>

function [P] = MESA(X, YYl, option, param)

%% hyperparameters.
NITER = option.NITER;
fcm_options = option.fcm_options;
k = option.k;
nl = size(YYl, 1);
eps = 1e-10;

%% tensor and matrix definition.
L = YYl;

num_view = size(X, 2);
N = size(X{1, 1}, 2);
D = zeros(1, num_view);

W = cell(1, num_view);
G = cell(1, num_view);

A = cell(1, num_view);
P = cell(1, num_view);
M = cell(1, num_view);
U = cell(1, num_view);


for v = 1:num_view
    W{v} = zeros(N, k);
    G{v} = zeros(N, k);
    A{v} = zeros(N, k);
    D(v) = size(X{1, v}, 1);
end

for v = 1:num_view
    d = D(v);
    P{v} = zeros(d, k);
    M{v} = zeros(k, k);
end

%% FCM for U{v} init.

for v = 1:num_view
    [~, tmp_aff_matrix] = fcm(X{v}', k, fcm_options);
    U{v} = tmp_aff_matrix';
end

%% Iterations start.

for ITER = 1:NITER

    for v = 1:num_view

        Xv = X{v};
        Av = A{v};
        Uv = U{v};

        %% Update Pv.
        Pv = P{v};
        Pi = sqrt(sum(Pv .* Pv, 2) + eps);
        Hh = 0.5 ./ Pi;
        H = diag(Hh);


        Pv = inv(Xv * Xv' + param.lambda * H) * Xv * Av;
        P{v} = Pv;

        %% Update Mv.
        Mv = M{v};
        I = eye(k);
        Mv = inv(Av(1:nl, :)' * Av(1:nl, :) + param.gamma .* I) * Av(1:nl, :)' * L;
        M{v} = Mv;

        %% Update Av.
        Av = A{v};
        Mv = M{v};
        Pv = P{v};
        Wv = W{v};
        Gv = G{v};

        I = eye(k);
        %         Av1 = (Uv(1:nl, :) + param.alpha * L * Mv' + param.beta * Xv(:, 1:nl)' * Pv - Wv(1:nl, :) + param.rho / 2 .* Gv(1:nl, :)) ...
        %             * inv(I + param.alpha .* Mv' * Mv + param.beta + param.rho / 2);
        Av1 = (Uv(1:nl, :) + param.beta * Xv(:, 1:nl)' * Pv - Wv(1:nl, :) + param.rho / 2 .* Gv(1:nl, :)) ...
            * inv(I + param.beta + param.rho / 2);
        Av2 = (Uv(nl+1:N, :) + param.beta * Xv(:, nl+1:N)' * Pv - Wv(nl+1:N, :) + param.rho / 2 .* Gv(nl+1:N, :)) ...
            * inv(I + param.beta + param.rho / 2);

        for d = 1:size(Av1, 1)
            Av1_tmp(d, :) = EProjSimplex_new(Av1(d, :));
        end

        for d = 1:size(Av2, 1)
            Av2_tmp(d, :) = EProjSimplex_new(Av2(d, :));
        end

        A{v} = [Av1_tmp; Av2_tmp];

        %% Update G.


        A_tensor = cat(3, A{:, :});
        W_tensor = cat(3, W{:, :});
        tmp_A = A_tensor(:);
        tmp_W = W_tensor(:);

        size_A_tensor = [N, k, num_view];
        [g, ~] = Gshrink(tmp_A + 1 / param.rho * tmp_W(:), N / param.rho, size_A_tensor, 0, 3);

        G_tensor = reshape(g, size_A_tensor);

        for vv_in_update_G = 1:num_view
            G{vv_in_update_G} = G_tensor(:, :, vv_in_update_G);
        end

        %% update W.
        tmp_W = tmp_W + param.rho * (A_tensor(:) - g);
    end
end


end
