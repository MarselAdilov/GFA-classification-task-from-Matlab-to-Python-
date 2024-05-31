clear all;
close all;
clc;

%% Load Data 
load('Data')
load('Primary_data')
load('CBNN_Tl')
load('CBNN_Tg')


%% Calculation


DL=Tl(~strcmp(Tl,'None')&~strcmp(Tg,'None'));
DG=Tg(~strcmp(Tl,'None')&~strcmp(Tg,'None'));

X=Data(:,:,~strcmp(Tl,'None')&~strcmp(Tg,'None'));

DL=reshape(str2double(DL),1,[]);
DG=reshape(str2double(DG),1,[]);

%% TL calculation
    for k = 1:length(DL)
        %% Data Correlating
        x    = X(:, :, k);
        yC1  = Conv(x, WC_Tl);
        yC2  = ReLU(yC1);
        yC   = Pool(yC2);
        %% Data Flattening
        yC_f         = reshape(yC, [], 1);
        x_f          = reshape(x, [], 1);
        x_flattened(:,k)  = [yC_f;x_f];
    end
    
tL=DL;
yL=net_Tl(x_flattened);    
performanceL = perform(net_Tl,tL,yL);

%% TG calculation
    for k = 1:length(DG)
        %% Data Correlating
        x    = X(:, :, k);
        yC1  = Conv(x, WC_Tg);
        yC2  = ReLU(yC1);
        yC   = Pool(yC2);
        %% Data Flattening
        yC_f         = reshape(yC, [], 1);
        x_f          = reshape(x, [], 1);
        x_flattened(:,k)  = [yC_f;x_f];
    end
    
tG=DG;
yG=net_Tg(x_flattened);    
performanceL = perform(net_Tg,tG,yG);

%%
t=tG./tL;
y=yG./yL;
[r,m,b] = regression(t,y);
m = m(1); b = b(1); r = r(1);
rmse=immse(y(2:end),t(2:end));

%%
figure (1)
axes('fontsize',24,'fontweight','Bold')
hold on
scatter(t(1,:),y(1,:),50,'filled','LineWidth',1.5)
plot=plot([0.35 0.75],[0.35 0.75],'--k','linewidth',1.5);
plot.Color(4) = 0.5; % Transparancy
text(0.5,0.73,['R=' num2str(r)],'fontsize',20,'fontweight','Bold');
text(0.48,0.71,['RMSE=' num2str(rmse)],'fontsize',20,'fontweight','Bold');
xlabel('Measured T_{g}/T_{l}','fontsize',24)
ylabel('Predicted T_{g}/T_{l}','fontsize',24)
xlim([0.35 0.75]);
ylim([0.35 0.75]);
axis square

ax = gca;
ax.LineWidth=0.5;
ax.XTick = 0.35:0.05:0.75;
ay = gca;
ay.LineWidth=0.5;
ay.YTick = 0.35:0.05:0.75;
% legend('1^s^t subpeak ratio','2^n^d subpeak ratio','Location','southwest','Orientation','vertical')
set(gcf,'color','w')
% grid on
% grid minor
box on
axis
