function [WC_new, NET, TR] = trainConv(X,D,hidden_layers, CC, epoch)


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
    net = fitnet(hidden_layers,trainFcn);
    net.trainParam.epochs=2000; %more epochs
    % Choose Input and Output Pre/Post-Processing Functions
    % For a list of all processing functions type: help nnprocess
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.output.processFcns = {'removeconstantrows','mapminmax'};
    % Setup Division of Data for Training, Validation, Testing
    % For a list of all data division functions type: help nndivide
    net.divideFcn = 'dividerand'; % Divide data randomly
    net.divideMode = 'sample'; % Divide up every sample
    net.divideParam.trainRatio = 96/100;
    net.divideParam.valRatio = 2/100;
    net.divideParam.testRatio = 2/100;
    % Choose a Performance Function
    % For a list of all performance functions type: help nnperformance
    net.performFcn = 'mse'; % Mean Squared Error
    % Choose Plot Functions
    % For a list of all plot functions type: help nnplot
    net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
        'plotregression', 'plotfit'};
    % Train the Network
    [net,tr] = train(net,x,t);
    % Test the Network
    y = net(x);
    e = gsubtract(t,y);
    performance = perform(net,t,y);
    % Recalculate Training, Validation and Test Performance
    trainTargets = t .* tr.trainMask{1};
    valTargets = t .* tr.valMask{1};
    testTargets = t .* tr.testMask{1};
    trainPerformance = perform(net,trainTargets,y);
    valPerformance = perform(net,valTargets,y);
    testPerformance = perform(net,testTargets,y);
    % View the Network
    %     view(net)
    % Plots
    % Uncomment these lines to enable various plots.
    %figure, plotperform(tr)
    %figure, plottrainstate(tr)
    %figure, ploterrhist(e)
    %figure, plotregression(t,y)
    %figure, plotfit(net,x,t)
    i=i+1;
    
    if performance<OldPerformance
        WC_new=WC;
        NET=net;
        TR=tr;
        
        OldPerformance=performance;
    end
    
OldPerformance
end
close (h)
end





