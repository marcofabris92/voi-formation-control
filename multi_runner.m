%% VoI-aware Scheduling Schemes for Multi-Agent Formation Control
% Authors: Federico Chiariotti and Marco Fabris
% e-mails: federico.chiariotti.@unipd.it   marco.fabris.1@unipd.it
% Department of Information Engineering, University of Padova

% This is the main script. It is used to run the Monte Carlo simulations.
% Invokes:
% - episode.m
% - instant_loss

clear all
close all
clc

%% Main parameters for the M.C. experiment
duration = 10;
step = 0.01;
tx_time = 10;
sigma = 0.01;
formation_shape = 1;
cube_side = 5;
episodes = 2; % >= 2


%% Application: first-order formation tracking
% agent space: the 3D world
% desired formation: a cube
% centroid tracking: a tanh-like curve

% Desired distances
sq2 = sqrt(2);
sq3 = sqrt(3);
dfc = sq3/2; % distance from a vertex to the cube centroid

switch formation_shape
    case 0
        % Standard cube (complete graph)
        % dijs = cube_side*[...
        %     0 1 sq2 1 1 sq2 sq3 sq2;
        %     1 0 1 sq2 sq2 1 sq2 sq3;
        %     sq2 1 0 1 sq3 sq2 1 sq2;
        %     1 sq2 1 0 sq2 sq3 sq2 1;
        %     1 sq2 sq3 sq2 0 1 sq2 1;
        %     sq2 1 sq2 sq3 1 0 1 sq2;
        %     sq3 sq2 1 sq2 sq2 1 0 1;
        %     sq2 sq3 sq2 1 1 sq2 1 0];
        % Standard cube minus 8 diagonals
        dijs = cube_side*[...
            0 1 0 1 1 sq2 0 sq2;
            1 0 1 sq2 0 1 0 sq3;
            0 1 0 1 sq3 sq2 1 0;
            1 sq2 1 0 0 0 sq2 1;
            1 0 sq3 0 0 1 sq2 1;
            sq2 1 sq2 0 1 0 1 0;
            0 0 1 sq2 sq2 1 0 1;
            sq2 sq3 0 1 1 0 1 0];
    case 1
        % Asymmetric 'cube' with node 8 in the center (complete graph)
        % dijs = cube_side*[...
        %     0 1 sq2 1 1 sq2 sq3 dfc;
        %     1 0 1 sq2 sq2 1 sq2 dfc;
        %     sq2 1 0 1 sq3 sq2 1 dfc;
        %     1 sq2 1 0 sq2 sq3 sq2 dfc;
        %     1 sq2 sq3 sq2 0 1 sq2 dfc;
        %     sq2 1 sq2 sq3 1 0 1 dfc;
        %     sq3 sq2 1 sq2 sq2 1 0 dfc;
        %     dfc dfc dfc dfc dfc dfc dfc 0];
        % Asymmetric 'cube' with node 8 in the center minus 6 diagonals
        dijs = cube_side*[...
            0 1 0 1 1 sq2 0 dfc;
            1 0 1 sq2 0 1 sq2 dfc;
            0 1 0 1 sq3 0 1 dfc;
            1 sq2 1 0 0 0 sq2 dfc;
            1 0 sq3 0 0 1 sq2 dfc;
            sq2 1 0 0 1 0 1 dfc;
            0 sq2 1 sq2 sq2 1 0 dfc;
            dfc dfc dfc dfc dfc dfc dfc 0];
end


%% Topological parameters
D = sum((dijs ~= 0),2); % node degrees
n = length(D); % number of agents

% Centrality metrics
G = graph(dijs); % undirected graph weighted by dijs
C = centrality(G, 'closeness', 'Cost', G.Edges.Weight);
C = C / sum(C);

% weighted adjacency matrix 
WA = zeros(n,n);
for i = 1:n
    for j = 1:n
        if dijs(i,j) > 0 || i == j
            WA(i,j) = sqrt(C(i)*C(j));
        end
    end
end


%% Path planning for the centroid trajectory
T = duration; % max simulation time
dt = step;
d = 3; % dimension of the ambient space
dn = d*n;
p0 = 10*ones(dn,1)+0.1*randn(dn,1); % initial positions of the agents
tspan = 0:dt:T;
LT = length(tspan);
pCdes = zeros(LT,d); % desired centroid position to be tracked
for l = 1:LT
    t = tspan(l);
    pCdes(l,:) = [t 0 5-5*tanh(t-T/2)];
end
K_tr = 10; % tracking gain
K_fo = 50; % formation gain

%% Communication parameters
T_tx = tx_time; % Number of steps between scheduled transmissions
sigma_drift = sigma*(0.5+randperm(n,n)*0.5); 

%% Run episodes
algo_MAX = 4;
% case algo_TX = 0 % Round robin scheduling
% case algo_TX = 1 % Minimum weighted age scheduling
% case algo_TX = 2 % Value-based scheduling
% case algo_TX = 3 % Ideal scheduling (Oracle)

p_actual_matrix = zeros(algo_MAX,episodes,LT,dn);
p_ideal_matrix = zeros(algo_MAX,episodes,LT,dn);
scheduling_matrix = zeros(algo_MAX,episodes,floor((LT-1)/T_tx));
actual_formation_loss_matrix = zeros(algo_MAX,episodes,LT);
ideal_formation_loss_matrix = zeros(algo_MAX,episodes,LT);

parfor algo_TX = 0:algo_MAX-1
    for ep = 1:episodes
        rng(ep*13);
        fprintf(['algo_TX = ' num2str(algo_TX) ', ep = ' num2str(ep) '\n'])
        [p_ideal, p_actual, sched] = ...
            episode(p0,dijs,T,dt,pCdes,K_tr,K_fo,T_tx,sigma_drift,...
            algo_TX,C,D,WA);
        
        form_loss_act = zeros(1,LT);
        for i = 1:LT
            form_loss_act(i) = instant_loss(p_actual(i, :), pCdes(i, :),...
                dijs,K_tr,K_fo,WA);
        end
        form_loss_ide = zeros(1,LT);
        for i = 1:LT
            form_loss_ide(i) = instant_loss(p_ideal(i, :), pCdes(i, :),...
                dijs,K_tr,K_fo,WA);
        end

        p_actual_matrix(algo_TX+1,ep,:,:) = p_actual;
        p_ideal_matrix(algo_TX+1,ep,:,:) = p_ideal;
        scheduling_matrix(algo_TX+1,ep,:) = sched;
        actual_formation_loss_matrix(algo_TX+1,ep,:) = form_loss_act;
        ideal_formation_loss_matrix(algo_TX+1,ep,:) = form_loss_ide;

    end
end

filename = ['results' num2str(formation_shape) '.mat'];
save(filename, 'p_actual_matrix', 'p_ideal_matrix',...
    'scheduling_matrix', 'dijs', 'T_tx', 'duration', 'step', 'sigma',...
    'actual_formation_loss_matrix', 'ideal_formation_loss_matrix',...
    'formation_shape', 'cube_side', 'pCdes', 'algo_MAX');


%% Loss computation
function value = instant_loss(p,pCdes,dijs,K_tr,K_fo,WA)
    d = length(pCdes);
    n = size(dijs,1);
    p = reshape(p,d,n);
    value = 0;
    loss_C = K_tr/2*norm(reshape(pCdes,d,1)-sum(p,2)/n)^2;
    for i = 1:n
        for j = i+1:n
            if dijs(i,j) > 0
                value = value + ...
                    K_fo/2*WA(i,j)*(norm(p(:,i)-p(:,j))^2 - dijs(i,j)^2)^2;
            end
        end
        value = value + WA(i,i)*loss_C;
    end
end