function [x,objV] = Gshrink(x,rho,sX, isWeight,mode)

if isWeight == 1
    %     C = 2*sqrt(2)*sqrt(sX(3)*sX(2));
    C = sqrt(sX(3)*sX(2));
end

if ~exist('mode','var')

    mode = 1;
end

X=reshape(x,sX);
if mode == 1
    Y=X2Yi(X,3);
elseif mode == 3
    Y=shiftdim(X, 1);
else
    Y = X;
end

Yhat = fft(Y,[],3);
% weight = C./sTrueV+eps;
% weight = 1;
% tau = rho*weight;
objV = 0;
if mode == 1
    n3 = sX(2);
elseif mode == 3
    n3 = sX(1);
else
    n3 = sX(3);
end

for i = 1:n3
    [uhat,shat,vhat] = svd(full(Yhat(:,:,i)),'econ');
    tau = rho;
    shat = max(shat - tau,0);
    objV = objV + sum(shat(:));
    Yhat(:,:,i) = uhat*shat*vhat';
end

Y = ifft(Yhat,[],3);

if mode == 1
    X = Yi2X(Y,3);
elseif mode == 3
    X = shiftdim(Y, 2);
else
    X = Y;
end

x = X(:);

end
