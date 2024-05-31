clear all;
close all;
clc;

%% Load Data 
load('Data')
load('Primary_data')

D(1,:)=strcmp(GFA,'BMG');
D(2,:)=strcmp(GFA,'Ribbon');
D(3,:)=strcmp(GFA,'None');
X=Data;

%% Initialize parameters
CC=[2 2 20];                    % Cross correlation architecture 4*3*20
hidden_layers=[10 10 10];        % 3 layers 
epoch=20;

tic
[WC_GFA, net_GFA]=trainConv(X(:,:,2:end),D(:,2:end),hidden_layers, CC, epoch);
toc


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
performance = perform(net_GFA,t,y)
view(net_GFA)

load gong.mat;
sound(y);


save('CBNN_GFA.mat','WC_GFA','net_GFA');


