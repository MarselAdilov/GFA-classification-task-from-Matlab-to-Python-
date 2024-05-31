clear all;
close all;
clc;

%% Load Data 
load('Data')
load('Primary_data')
load('CBNN_Dmax')

%% Calculation
D=D_Max(~strcmp(D_Max,'None'));
X=Data(:,:,~strcmp(D_Max,'None'));

D=reshape(str2double(D),1,[]);

    for k = 1:length(D)
        %% Data Correlating
        x    = X(:, :, k);
        yC1  = Conv(x, WC_Dmax);
        yC2  = ReLU(yC1);
        yC   = Pool(yC2);
        %% Data Flattening
        yC_f         = reshape(yC, [], 1);
        x_f          = reshape(x, [], 1);
        x_flattened(:,k)  = [yC_f;x_f];
    end
    
t=D;
y=net_Dmax(x_flattened);    
performance = perform(net_Dmax,t,y);
[r,m,b] = regression(t,y);
m = m(1); b = b(1); r = r(1);
rmse=immse(y(2:end),t(2:end));

%%
figure (1)
axes('fontsize',24,'fontweight','Bold')
hold on
scatter(D(1,:),y(1,:),50,'filled','LineWidth',1.5)
plot=plot([-5 75],[-5 75],'--k','linewidth',1.5);
plot.Color(4) = 0.5; % Transparancy
text(25,72,['R=' num2str(r)],'fontsize',20,'fontweight','Bold');
text(22,68,['RMSE=' num2str(rmse)],'fontsize',20,'fontweight','Bold');
xlabel('Measured D_{Max}','fontsize',24)
ylabel('Predicted D_{Max}','fontsize',24)
xlim([-5 75]);
ylim([-5 75]);
axis square

ax = gca;
ax.LineWidth=0.5;
ax.XTick = 0:15:75;
ay = gca;
ay.LineWidth=0.5;
ay.YTick = 0:15:75;
% legend('1^s^t subpeak ratio','2^n^d subpeak ratio','Location','southwest','Orientation','vertical')
set(gcf,'color','w')
% grid on
% grid minor
box on
axis
