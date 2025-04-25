function value = instant_loss(p, pCdes, dijs, K_tr, K_fo)
    d = length(pCdes);
    n = size(dijs,1);
    p = reshape(p,d,n);
    value = 0;
    for i = 1:n
        pi = p(:,i);
        parfor j = 1:n
            if dijs(i,j) > 0
                value = value + ...
                    K_fo/4*(norm(pi-p(:,j))^2 - dijs(i,j)^2)^2;
            end
        end
    end
    value = value + n*K_tr/2*norm(reshape(pCdes,d,1)-sum(p,2)/n)^2;
end