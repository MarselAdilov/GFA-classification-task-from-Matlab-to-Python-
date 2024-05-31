clear all;
close all;
clc;

%% Load Data 
load('Data')
load('Primary_data')

D=D_Max(~strcmp(D_Max,'None'));
X=Data(:,:,~strcmp(D_Max,'None'));

D=reshape(str2double(D),1,[]);


%% Initialize parameters
CC=[2 2 20];                    % Cross correlation architecture 4*3*20
hidden_layers=[10 10 10];        % 3 layers 
epoch=100;

tic
[WC_Dmax, net_Dmax, tr_Dmax]=trainConv(X(:,:,2:end),D(1,2:end),hidden_layers, CC, epoch);
toc


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
plotregression(t,y)
performance = perform(net_Dmax,t,y)

load gong.mat;
sound(y);


save('CBNN_Dmax.mat','WC_Dmax','net_Dmax','tr_Dmax');


