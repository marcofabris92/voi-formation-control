%% VoI-aware Scheduling Schemes for Multi-Agent Formation Control
% Authors: Federico Chiariotti and Marco Fabris
% Emails: federico.chiariotti.@unipd.it   marco.fabris.1@unipd.it
% Department of Information Engineering, University of Padova

% This script shows the results obtained through Monte Carlo simulations.
% Invokes: 
% - graphics_main

clear all
close all
clc


results = 'results_shape0.mat'; % 0 or 1


load(results)
fprintf('-----------------------\n')
fprintf('Standard cube')
display(formation_shape)
display(cube_side)
display(dijs)
display(duration)
display(step)
display(T_tx)
display(sigma)

n = size(dijs,1);
d = size(pCdes,2);


%% Plotting means of loss values
tt = 0:step:duration;
T = length(tt);
mean_1a = mean(squeeze(actual_formation_loss_matrix(1,:,:)),1);
mean_1i = mean(squeeze(ideal_formation_loss_matrix(1,:,:)),1);
mean_2a = mean(squeeze(actual_formation_loss_matrix(2,:,:)),1);
mean_2i = mean(squeeze(ideal_formation_loss_matrix(2,:,:)),1);
mean_3a = mean(squeeze(actual_formation_loss_matrix(3,:,:)),1);
mean_3i = mean(squeeze(ideal_formation_loss_matrix(3,:,:)),1);

figure('position',[100 100 1300 600])
grid on
hold on
m1i = plot(tt,mean_1i,'color',[255 128 0]/255,'LineWidth',1.5);
m2i = plot(tt,mean_2i,'color',[0 204 0]/255,'LineWidth',1.5);
m3i = plot(tt,mean_3i,'color',[153 153 255]/255,'LineWidth',1.5);
m1a = plot(tt,mean_1a,'r','LineWidth',2);
m2a = plot(tt,mean_2a,'color',[51 255 51]/255,'LineWidth',2);
m3a = plot(tt,mean_3a,'b','LineWidth',2);
set(gca, 'YScale', 'log')
set(gca,"TickLabelInterpreter",'latex','FontSize',20)
xlabel('$t$','interpreter', 'latex','FontSize',20)
ylabel('$\mathcal{L}(t)$','interpreter', 'latex','FontSize',20)
legend([m1a m1i m2a m2i m3a m3i],...
    {'actual1','ideal1','actual2','ideal2','actual3','ideal3'},...
    'Interpreter','latex','FontSize',10)
hold off







%% Plotting trajectories
ftsz = 30;
lw = 2;
grp.XBdes = pCdes;
grp.Xi = squeeze(p_ideal_matrix(1,1,:,:));     
grp.kT = 3; % number of snapshots along the trajectory
grp.T0 = 0:step:duration;

graphics_main(grp,n,d,ftsz,lw,dijs~=0);




%% 3-D plot of the agents' trajectories
function [] = graphics_main(grp,nAg,DIM,ftsz,lw,A)


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
% dists


%% phase portraits
kTgif = 2; % to make animated GIF > 2... if not desired, set it to 2
for ii = 1:kTgif-1
    traj_fig = figure('position',[500 100 600 600]);

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