clear all;
close all;
clc;

%% Load Data 
load('Data')
load('Primary_data')
load('CBNN_GFA')

%% Calculation
D(1,:)=strcmp(GFA,'BMG');
D(2,:)=strcmp(GFA,'Ribbon');
D(3,:)=strcmp(GFA,'None');
X=Data;

    for k = 1:length(D)
        %% Data Correlating
        x    = X(:, :, k);
        yC1  = Conv(x, WC_GFA);
        yC2  = ReLU(yC1);
        yC   = Pool(yC2);
        %% Data Flattening
        yC_f         = reshape(yC, [], 1);
        x_f          = reshape(x, [], 1);
        x_flattened(:,k)  = [yC_f;x_f];
    end
    
t=D;
y=net_GFA(x_flattened);    
performance = perform(net_GFA,t,y);
% plotconfusion(D,y)


%%
figure (1)
axes('fontsize',24,'fontweight','Bold')
hold on
[tpr,fpr] = roc(t,y);
plot([0 fpr{1,1} 1],[0 tpr{1,1} 1],'color',[0.75 0 0 ],'linewidth',2)
plot([0 fpr{1,2} 1],[0 tpr{1,2} 1],'color',[0 0 0.75],'linewidth',2)
plot([0 fpr{1,3} 1],[0 tpr{1,3} 1],'color',[0 0.75 0],'linewidth',2)
plot=plot([0 1],[0 1],'--k','linewidth',1.5);
plot.Color(4) = 0.5; % Transparancy
title ''
xlabel('False Positive Rate','fontsize',24)
ylabel('True Positive Rate','fontsize',24)
xlim([0 1]);
ylim([0 1]);
axis square

ax = gca;
ax.LineWidth=0.5;
ax.XTick = 0:0.2:1;
ay = gca;
ay.LineWidth=0.5;
ay.YTick = 0:0.2:1;
legend('BMG','Ribbone','CRA','Random Guess','Location','southeast','Orientation','vertical')
set(gcf,'color','w')
% grid on
% grid minor
box on
axis
