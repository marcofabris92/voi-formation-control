
clear all
close all
clc

%% MAIN PARAMETERS FOR SIMULATION
duration = 10;
step = 0.01;
tx_time = 10;
sigma = 0.01;
formation_shape = 0;
cube_side = 5;
episodes = 100;


% APPLICATION: FORMATION TRACKING (for dummies)
% agent space: the 3D world
% desired formation: a cube
% centroid tracking: a tanh-like curve

%% Desired distances
% (we don't need to specify them all, a complete graph isn't necessary!)

sq2 = sqrt(2);
sq3 = sqrt(3);
dfc = sq3/2; % distance from a vertex to the cube centroid
dd = cube_side;

if (formation_shape == 0)
    % Standard cube
    % dijs = dd*[...
    %     0 1 sq2 1 1 sq2 sq3 sq2;
    %     1 0 1 sq2 sq2 1 sq2 sq3;
    %     sq2 1 0 1 sq3 sq2 1 sq2;
    %     1 sq2 1 0 sq2 sq3 sq2 1;
    %     1 sq2 sq3 sq2 0 1 sq2 1;
    %     sq2 1 sq2 sq3 1 0 1 sq2;
    %     sq3 sq2 1 sq2 sq2 1 0 1;
    %     sq2 sq3 sq2 1 1 sq2 1 0];
    % Standard cube minus 8 diagonals
    dijs = dd*[...
        0 1 0 1 1 sq2 0 sq2;
        1 0 1 sq2 0 1 0 sq3;
        0 1 0 1 sq3 sq2 1 0;
        1 sq2 1 0 0 0 sq2 1;
        1 0 sq3 0 0 1 sq2 1;
        sq2 1 sq2 0 1 0 1 0;
        0 0 1 sq2 sq2 1 0 1;
        sq2 sq3 0 1 1 0 1 0];
end
if (formation_shape == 1)
    % % Asymmetrical cube with node 8 in the center
    % dijs = dd*[0 1 sq2 1 1 sq2 sq3 dfc;
    %     1 0 1 sq2 sq2 1 sq2 dfc;
    %     sq2 1 0 1 sq3 sq2 1 dfc;
    %     1 sq2 1 0 sq2 sq3 sq2 dfc;
    %     1 sq2 sq3 sq2 0 1 sq2 dfc;
    %     sq2 1 sq2 sq3 1 0 1 dfc;
    %     sq3 sq2 1 sq2 sq2 1 0 dfc;
    %     dfc dfc dfc dfc dfc dfc dfc 0];
    % Asymmetrical cube with node 8 in the center minus 6 diagonals
    dijs = dd*[...
        0 1 0 1 1 sq2 0 dfc;
        1 0 1 sq2 0 1 sq2 dfc;
        0 1 0 1 sq3 0 1 dfc;
        1 sq2 1 0 0 0 sq2 dfc;
        1 0 sq3 0 0 1 sq2 dfc;
        sq2 1 0 0 1 0 1 dfc;
        0 sq2 1 sq2 sq2 1 0 dfc;
        dfc dfc dfc dfc dfc dfc dfc 0];
end


n = size(dijs,1); % number of agents


%% Path planning for the centroid trajectory
T = duration; % max simulation time
dt = step;
d = 3; % dimension of the space
dn = d*n;
p0 = 10*ones(dn,1)+0.1*randn(dn,1); % initial positions of the agents
tspan = 0:dt:T;
LT = length(tspan);
pCdes = zeros(LT,d); % desired centroid position to be tracked
for l = 1:LT
    t = tspan(l);
    pCdes(l,:) = [0 t 5-5*tanh(1*(t-T/2))];
end
K_tr = 1; % tracking gain
K_fo = 5; % formation gain

%% Communication parameters
T_tx = tx_time; % Number of steps between scheduled transmissions
sigma_drift = sigma * (0.5 + randperm(n, n) * 0.5); %zeros(1, n);%

%% Run episodes
algo_MAX = 3;

p_actual_matrix = zeros(algo_MAX, episodes, length(tspan), dn);
p_ideal_matrix = zeros(algo_MAX, episodes, length(tspan), dn);
scheduling_matrix = zeros(algo_MAX, episodes, floor((length(tspan) - 1) / T_tx));
actual_formation_loss_matrix = zeros(algo_MAX, episodes, length(tspan));
ideal_formation_loss_matrix = zeros(algo_MAX, episodes, length(tspan));

parfor algo_TX = 0 : algo_MAX-1
    for ep = 1 : episodes
        fprintf(['algo_TX = ' num2str(algo_TX) ', ep = ' num2str(ep) '\n'])
        [p_ideal, p_actual, sched] = ...
            episode(p0, dijs, T, dt, pCdes, K_tr, K_fo, T_tx, sigma_drift, algo_TX);
        
        form_loss_act = zeros(1, length(tspan));
        for i = 1 : length(tspan)
            form_loss_act(i) = ...
                instant_loss(p_actual(i, :), pCdes(i, :), dijs, K_tr, K_fo);
        end
        form_loss_ide = zeros(1, length(tspan));
        for i = 1 : length(tspan)
            form_loss_ide(i) = ...
                instant_loss(p_ideal(i, :), pCdes(i, :), dijs, K_tr, K_fo);
        end

        p_actual_matrix(algo_TX + 1, ep, :, :) = p_actual;
        p_ideal_matrix(algo_TX + 1, ep, :, :) = p_ideal;
        scheduling_matrix(algo_TX + 1, ep, :) = sched;
        actual_formation_loss_matrix(algo_TX + 1, ep, :) = form_loss_act;
        ideal_formation_loss_matrix(algo_TX + 1, ep, :) = form_loss_ide;

    end
end

save('results.mat', 'p_actual_matrix', 'p_ideal_matrix', 'scheduling_matrix', 'dijs', 'T_tx', 'duration', 'step', 'sigma', 'actual_formation_loss_matrix', 'ideal_formation_loss_matrix', 'formation_shape', 'cube_side','pCdes');


