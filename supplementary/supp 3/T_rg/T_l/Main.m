clear all;
close all;
clc;

%% Load Data 
load('Data')
load('Primary_data')

D=Tl(~strcmp(Tl,'None'));
X=Data(:,:,~strcmp(Tl,'None'));

D=reshape(str2double(D),1,[]);


%% Initialize parameters
CC=[2 2 20];                    % Cross correlation architecture 4*3*20
hidden_layers=[10 10 10];        % 3 layers 
epoch=100;

tic
[WC_Tl, net_Tl, tr_Tl]=trainConv(X(:,:,2:end),D(1,2:end),hidden_layers, CC, epoch);
toc


    for k = 1:length(D)
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
    
t=D;
y=net_Tl(x_flattened);    
plotregression(t,y)
performance = perform(net_Tl,t,y)

load gong.mat;
sound(y);


save('CBNN_Tl.mat','WC_Tl','net_Tl','tr_Tl');


