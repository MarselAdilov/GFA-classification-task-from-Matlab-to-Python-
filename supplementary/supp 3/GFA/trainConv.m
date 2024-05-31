function [WC_new, NET] = trainConv(X,D,hidden_layers, CC, epoch)


% WC = 0.1*randn(CC);
a=10;
WC=-a*ones(CC);

i=1;
OldPerformance=10;
h = waitbar(0,'... CBNN Training process  ...');

while (i <= epoch)
    
    waitbar(i/epoch);   % wait bar function
    
    step=2*a/epoch;
    WC=WC+step;
    
    for k = 1:length(D)
        %% Data Correlating
        x    = X(:, :, k);
        yC1  = Conv(x, WC);
        yC2  = ReLU(yC1);
        yC   = Pool(yC2);
        %% Data Flattening
        yC_f         = reshape(yC, [], 1);
        x_f          = reshape(x, [], 1);
        x_flattened(:,k)  = [yC_f;x_f];
    end
    
    x = x_flattened;
    t = D;
    
    % Choose a Training Function
    % For a list of all training functions type: help nntrain
    % 'trainlm' is usually fastest.
    % 'trainbr' takes longer but may be better for challenging problems.
    % 'trainscg' uses less memory. Suitable in low memory situations.
    trainFcn = 'trainlm'; % Levenberg-Marquardt backpropagation.
    
    net = patternnet(hidden_layers,trainFcn,'mse');
    net = train(net,x,t);
    net.divideParam.trainRatio = 96/100;
    net.divideParam.valRatio = 2/100;
    net.divideParam.testRatio = 2/100;
    y = net(x);
    performance = perform(net,t,y);
    classes = vec2ind(y);
      
    
    
    i=i+1;
    
    if performance<OldPerformance
        WC_new=WC;
        NET=net;
        OldPerformance=performance;
    end
    
OldPerformance
end
close (h)
end





