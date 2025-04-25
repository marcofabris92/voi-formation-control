function [p_ideal, p_actual, sched] = episode(p0, dijs, T, dt, pCdes, K_tr, K_fo, T_tx, sigma_drift, tx_algo)

n = size(dijs,1); % number of agents
d = size(pCdes,2); % dimension of the ambient space
dn = d*n;
D = sum((dijs ~= 0),2);

% Centrality metrics
G = graph(dijs);
C = centrality(G, 'closeness', 'Cost', G.Edges.Weight);
C = C / sum(C);


%% embedding parameters for solving the ODE
par.dijs = dijs;
par.pCdes = pCdes;
par.K_tr = K_tr;
par.K_fo = K_fo;
par.K_est = 10*par.K_tr; 
par.n = n;
par.d = d;
par.T = T;
par.dt = dt;

%% Call to solver
% p0 = 10*ones(d*n,1)+0.1*randn(d*n,1); % initial positions of the agents
% LS = 1+T_tx; % Steps for each free evolution step
T0 = (0:T_tx) * dt;
K = ceil(T / (T_tx * dt));
sched = zeros(1, K);
p_ideal = zeros(dn, K * T_tx);
p_actual = zeros(dn, K * T_tx);
age = zeros(1, n);
p_ideal(:, 1) = p0;
p_actual(:, 1) = p0;
for k = 1 : K
    age = age + 1;
    p0 = p_ideal(:, 1 + (k - 1) * T_tx);
    par.next_pCdes = pCdes((k-1)*T_tx + 1 : k * T_tx + 1, :);
    p0estimator = zeros(dn,n); 
    for i = 1:n 
        ii = 1+d*(i-1) : d*i; 
        p0estimator(ii,i) = p0(ii); 
        for j = 1:n 
            if dijs(i,j) > 0 
                jj = 1+d*(j-1) : d*j; 
                p0estimator(jj,i) = p0(jj); 
            end 
        end 
    end 
    for i = 1:n 
        avg_i = sum(reshape(p0estimator(:,i),d,n),2)/(1+D(i));
        for j = 1:n 
            if dijs(i,j) == 0 && i ~= j 
                jj = 1+d*(j-1) : d*j; 
                p0estimator(jj,i) = avg_i; 
            end 
        end 
    end 
    p0estimator = reshape(p0estimator,n*dn,1); 
    [~,p_next] = ode45(@(t,p)dyn(t,p,par),T0,[p0; p0estimator]); 
    p_next = p_next(:,1:dn); 
    p_ideal(:,2+(k-1)*T_tx:k*T_tx + 1) = p_next(2:end,:)';
    p_cont = p_next - p0';
    % Add drift
    for in = 1 : n
        for id = 1 : d
            noise = sigma_drift(in)*randn(1, T_tx);
            p0_act = p_actual((in - 1) * d + id, 1 + (k-1)*T_tx);
            p_actual((in - 1) * d + id, 2+(k-1)*T_tx:k*T_tx+1) = ...
                p_cont(2:end, (in - 1) * d + id)' + p0_act + cumsum(noise);
        end
    end
    % Communication
    switch tx_algo
        case 0 % Round robin scheduling
            [~, tx] = max(age);
        case 1 % Minimum weighted age scheduling
            [~, tx] = max(age .* sigma_drift .^ 2);
        case 2 % Value-based scheduling
            [~, tx] = max(age .* sigma_drift .^ 2 .* C');
        otherwise
            error('Invalid scheduling algorithm')
    end
    % Successful tx: reset ideal state for last step to real state
    age(tx) = 0;
    sched(k) = tx;
    for i = 1 : d
        p_ideal((tx - 1) * d + i, 1+k*T_tx) = ...
            p_actual((tx - 1) * d + i, 1+k*T_tx);
    end
end

p_actual = p_actual';
p_ideal = p_ideal';


end



%% NETWORKED CONTROL
% dynamics of a first-order multi-agent system controlled through 
% - a proportional controller for tracking (gain K_tr)
% - a potential-based controller for attaining formation (gain K_fo)
function pdot = dyn(t,p,par)

    n = par.n;
    d = par.d;
    dn = d*n;

    pe = reshape(p(dn+1:end),dn,n); 
    p = reshape(p(1:dn),d,n); 
    pedot = zeros(dn,n); 
    for i = 1:n 
        ii = 1+d*(i-1) : d*i;
        pedot(ii,i) = -par.K_est*(pe(ii,i)-p(:,i)); 
        for j = 1:n 
            if par.dijs(i,j) > 0 
                pedot(:,i) = pedot(:,i) - par.K_est*(pe(:,i)-pe(:,j)); 
            end 
        end 
    end 

    pC = zeros(d,n); 
    parfor i = 1:n
        pC(:,i) = sum(reshape(pe(:,i),d,n),2); 
    end
    pC = pC/n;

    pCdes = par.next_pCdes(1+floor(t/par.dt),:)';
    
    pdot = zeros(d,n);
    for i = 1:n
       pi = p(:,i);
       for j = 1:n
           if par.dijs(i,j) > 0
                pj = p(:,j);
                pdot(:,i) = pdot(:,i) + ...
                    (norm(pi - pj, 2)^2 - par.dijs(i,j)^2) * (pj - pi);
           end
       end
       pdot(:,i) = par.K_tr*(pCdes-pC(:,i)) + par.K_fo*pdot(:,i); 
    end

    
    for i = 1:n 
        ii = 1+d*(i-1) : d*i; 
        pedot(ii,i) = pedot(ii,i) + pdot(:,i); 
        for j = 1:n 
            if par.dijs(i,j) > 0 
                jj = 1+d*(j-1) : d*j; 
                pedot(jj,i) = pedot(jj,i) + pdot(:,j); 
            end 
        end 
    end 

    pdot = reshape(pdot,dn,1);
    pdot = reshape([pdot pedot],(n+1)*dn,1); 

end