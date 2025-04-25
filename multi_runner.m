
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
        1 0 1 sq2 0 1 0 dfc;
        0 1 0 1 sq3 sq2 1 dfc;
        1 sq2 1 0 0 0 sq2 dfc;
        1 0 sq3 0 0 1 sq2 dfc;
        sq2 1 sq2 0 1 0 1 dfc;
        0 0 1 sq2 sq2 1 0 dfc;
        dfc dfc dfc dfc dfc dfc dfc 0];
end


n = size(dijs,1); % number of agents


%% Path planning for the centroid trajectory
T = duration-eps; % max simulation time
dt = step;
d = 3; % dimension of the space
p0 = 10*ones(d*n,1)+0.1*randn(d*n,1); % initial positions of the agents
tspan = 0:dt:T;
LT = length(tspan);
pCdes = zeros(LT,d); % desired centroid position to be tracked
for l = 1:LT
    t = tspan(l);
    pCdes(l,:) = [0*sin(10*t) t -10*tanh(0.1*(t-T/2))];
end
K_tr = 1; % tracking gain
K_fo = 5; % formation gain

%% Communication parameters
T_tx = tx_time; % Number of steps between scheduled transmissions
sigma_drift = sigma * (0.5 + randperm(n, n) * 0.5); %zeros(1, n);%

%% Run episodes
p_actual_matrix = zeros(3, episodes, length(tspan), d*n);
p_ideal_matrix = zeros(3, episodes, length(tspan), d*n);
scheduling_matrix = zeros(3, episodes, floor((length(tspan) - 1) / T_tx));
formation_loss_matrix = zeros(3, episodes, length(tspan));


for algo_TX = 0 : 2
    for ep = 1 : episodes
        algo_TX, ep
        [p_ideal, p_actual, sched] = ...
            episode(p0, dijs, T, dt, pCdes, K_tr, K_fo, T_tx, sigma_drift, algo_TX);
        
        form_loss = zeros(1, length(tspan));
        for i = 1 : length(tspan)
            form_loss(i) = ...
                compute_loss(p_actual(i, :), pCdes(i, :), d, n, dijs, K_tr, K_fo);
        end

        p_actual_matrix(algo_TX + 1, ep, :, :) = p_actual;
        p_ideal_matrix(algo_TX + 1, ep, :, :) = p_ideal;
        scheduling_matrix(algo_TX + 1, ep, :) = sched;
        formation_loss_matrix(algo_TX + 1, ep, :) = form_loss;

        save('results.mat', 'p_actual_matrix', 'p_ideal_matrix', 'scheduling_matrix', 'dijs', 'T_tx', 'duration', 'step', 'sigma', 'formation_loss_matrix', 'formation_shape', 'cube_side');
    end
end



% %% Graphics
% 
% % parameters
% ftsz = 30;
% lw = 2;
% grp.XBdes = pCdes;
% grp.Xi = p_actual;     % <----
% grp.kT = 5; % number of snapshots along the trajectory
% grp.T0 = tspan; %1:dt:K;
% 
% graphics_main(grp,n,d,ftsz,lw,A)





function value = compute_loss(p,pCdes,d, n, dijs, K_tr, K_fo)
    p = reshape(p,d,n);
    pC = zeros(d,1);
    for i = 1:n
        pC = pC + p(:,i);
    end
    pC = pC/n;
    value = 0;
    for i = 1:n
        pi = p(:,i);
        for j = 1:n
            if dijs(i,j) > 0
                pj = p(:,j);
                value = value + ...
                    K_fo/4*(norm(pi - pj, 2)^2 - dijs(i,j)^2)^2;
            end
        end
    end
    value = value + n*K_tr/2*norm(pCdes-pC, 2)^2;
end



function [] = graphics_main(grp,nAg,DIM,ftsz,lw,A)

T0 = grp.T0;
T = length(T0);


XBdes = grp.XBdes;
Xi = grp.Xi;
kT = grp.kT;

XB = zeros(length(Xi(:,1)), DIM);
for kkk = 1:DIM
    for kk = 1:nAg
        XB(:,kkk) = XB(:,kkk) + Xi(:,(kk-1)*DIM+kkk)/nAg;
    end
end



% printing info about distances
p = Xi(end,1:DIM*nAg)';
pp = zeros(DIM,nAg);
for i = 1:nAg
    pp(:,i) = p((i-1)*DIM+1:i*DIM);
end
dists = zeros((nAg*(nAg-1))/2,1);
ij = 1;
for i = 1:nAg
    for j = 1:i-1
        dists(ij) = norm(pp(:,i)-pp(:,j));
        ij = ij+1;
    end
end
dists;


%% phase portraits
kTgif = 2; % to make animated GIF > 2... if not desired, set it to 2
for ii = 1:kTgif-1
    traj_fig = figure;

    % displying trajectory tracking
    Trajdes = XBdes;
    Ltraj = length(Xi(:,1));
    color_traj = [0 153 76]/255;
    color_bary = [128 255 0]/255;

    if DIM == 2
        plot(Trajdes(:,1),Trajdes(:,2),'k');
        hold on


        for k = 0:nAg-1
            plot(Xi(:,DIM*k+1),Xi(:,DIM*k+2), 'color', color_traj, 'linewidth', 2)
            for kt = 1:kT
                if kt == 1
                    tt = 1; %1+floor(ii/kTgif*Ltraj);
                else
                    tt = round((kt-1)/(kT-1)*Ltraj);
                end
                plot(Xi(tt,DIM*k+1),Xi(tt,DIM*k+2), '*', 'color',...
                    [1 0 0]/(kT-kt+1)+([0 0 255]/255)*(1-1/(kT-kt+1)),...
                    'linewidth', 3)
                txt = '';
                if kt == 1 || kt == kT
                    txt = strcat('$',num2str(k+1),'$');
                end
                text(Xi(tt,DIM*k+1),Xi(tt,DIM*k+2),txt,'HorizontalAlignment',...
                    'right','interpreter','latex','fontsize',ftsz)
            end
        end

        plot(XB(:,1),XB(:,2), 'color', color_bary, 'linewidth', 1);
    end

    if DIM == 3
        plot3(Trajdes(:,1),Trajdes(:,2),Trajdes(:,3),'k');
        hold on

        for k = 0:nAg-1
            plot3(Xi(:,DIM*k+1),Xi(:,DIM*k+2),Xi(:,DIM*k+3), 'color', color_traj, 'linewidth', 2)
            for kt = 1:kT
                if kt == 1
                    tt = 1; %1+floor(ii/kTgif*Ltraj);
                else
                    tt = round((kt-1)/(kT-1)*Ltraj);
                end
                plot3(Xi(tt,DIM*k+1),Xi(tt,DIM*k+2),Xi(tt,DIM*k+3), '*', 'color',...
                    [1 0 0]/(kT-kt+1)+([0 0 255]/255)*(1-1/(kT-kt+1)),...
                    'linewidth', 3)
                txt = '';
                if kt == 1 || kt == kT
                    txt = strcat('$',num2str(k+1),'$');
                end
                text(Xi(tt,DIM*k+1),Xi(tt,DIM*k+2),Xi(tt,DIM*k+3),txt,...
                    'HorizontalAlignment', 'right','interpreter','latex',...
                    'fontsize',ftsz)
            end
        end

        plot3(XB(:,1),XB(:,2),XB(:,3), 'color', color_bary, 'linewidth', 1);
    end

    % showing polygons or polyhedra
    color_ini = [1 0 0];
    color_fin = [0 0 1];
    lambda = linspace(0,1);
    for kt = 1:kT
        if kt == 1
            tt = 1+floor(ii/kTgif*Ltraj); %tt = 1; % <------------------------------
        else
            tt = round((kt-1)/(kT-1)*Ltraj);
        end
        if DIM == 2
            plot(XB(tt,1),XB(tt,2),'+','color',...
                color_ini/(kT-kt+1)+color_fin*(1-1/(kT-kt+1)), 'linewidth', 1.5);
        elseif DIM == 3
            plot3(XB(tt,1),XB(tt,2),XB(tt,3),'+','color',...
                color_ini/(kT-kt+1)+color_fin*(1-1/(kT-kt+1)), 'linewidth', 1.5);
        end
        for i = 0:nAg-1
            for j = 0:i-1
                segment = zeros(DIM,length(lambda));
                for k = 1:length(lambda)
                    segment(:,k) = (Xi(tt,i*DIM+1:i*DIM+DIM)*lambda(k)+...
                        Xi(tt,j*DIM+1:j*DIM+DIM)*(1-lambda(k)))';
                end
                if DIM == 2 && A(i+1,j+1) == 1
                    plot(segment(1,:), segment(2,:), 'color',...
                        color_ini/(kT-kt+1)+color_fin*(1-1/(kT-kt+1)))
                elseif DIM == 3 && A(i+1,j+1) == 1
                    plot3(segment(1,:), segment(2,:), segment(3,:), 'color',...
                        color_ini/(kT-kt+1)+color_fin*(1-1/(kT-kt+1)))
                end
            end
        end
    end

    ax = gca;
    ax.FontSize = 20;

    hold off
    grid on, zoom on
    %title('Trajectory tracking + formation achieved')
    axis equal
    set(gca,'fontsize', ftsz)
    xlabel('$x$ [m]','Interpreter',...
        'latex', 'FontSize', ftsz, 'linewidth', lw);
    ylabel('$y$ [m]','Interpreter',...
        'latex', 'FontSize', ftsz, 'linewidth', lw);
    if DIM == 3
        zlabel('$z$ [m]','Interpreter',...
            'latex', 'FontSize', ftsz, 'linewidth', lw);
    end
    set(gca,'TickLabelInterpreter','latex')

    figure(traj_fig);
    axis equal
    filename = strcat('traj-',num2str(ii));
    path_office = 'C:\Users\Marco\Desktop\traj_2D_formation';
    formattype = 'pdf';
    %saveas(traj_fig,fullfile(path_office,filename),formattype)

    grid on, zoom on
    axis equal
    set(gca,'fontsize', ftsz)
    xlabel('$x$ [m]','Interpreter',...
        'latex', 'FontSize', ftsz, 'linewidth', lw);
    ylabel('$y$ [m]','Interpreter',...
        'latex', 'FontSize', ftsz, 'linewidth', lw);
    if DIM == 3
        zlabel('$z$ [m]','Interpreter',...
            'latex', 'FontSize', ftsz, 'linewidth', lw);
    end
    set(gca,'TickLabelInterpreter','latex')

end


ax = gca;
ax.FontSize = 20;

hold off
grid on, zoom on
%title('Trajectory tracking + formation achieved')
axis equal
set(gca,'fontsize', ftsz)
xlabel('$x$ [m]','Interpreter',...
    'latex', 'FontSize', ftsz, 'linewidth', lw);
ylabel('$y$ [m]','Interpreter',...
    'latex', 'FontSize', ftsz, 'linewidth', lw);
if DIM == 3
    zlabel('$z$ [m]','Interpreter',...
        'latex', 'FontSize', ftsz, 'linewidth', lw);
end
set(gca,'TickLabelInterpreter','latex')




end