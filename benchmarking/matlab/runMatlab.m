function runMatlab()

% Set me!
gurlsDir = '~/GURLS/gurls';

for i = 1
    switch i
        case 1
            dir = '../../data/toy/'
        case 2
            dir = '../../data/titanic/'
        case 3
            dir = '../../data/big/'
        case 4
            dir = '../../data/reallyBig/'
    end
    xTrain = csvread([dir 'xTrain.csv']);
    xTest = csvread([dir 'xTest.csv']);
    yTrain = csvread([dir 'yTrain.csv']);
    yTest = csvread([dir 'yTest.csv']);
    primal(xTrain,xTest,yTrain,yTest,gurlsDir);
    dual(xTrain,xTest,yTrain,yTest,gurlsDir);
    gaussian(xTrain,xTest,yTrain,yTest,gurlsDir);

end
            


end

function primal(xTrain,xTest,yTrain,yTest,gurlsDir)
addpath(genpath(gurlsDir));
opt = gurls_defopt('Part A');

opt.seq = {'paramsel:loocvprimal','rls:primal','pred:primal','perf:macroavg'};
opt.process{1} = [2 2 0 0];
opt.process{2} = [3 3 2 2];
fprintf('Primal:\n')
tic();
gurls(xTrain,yTrain,opt,1);
gurls(xTest,yTest,opt,2);
toc();
fprintf('Accuracy: %.3f\n',opt.perf.acc)

end

function dual(xTrain,xTest,yTrain,yTest,gurlsDir)
addpath(genpath(gurlsDir));
opt = gurls_defopt('Part B');

opt.seq = {'kernel:linear','paramsel:loocvdual','rls:dual','pred:dual','perf:macroavg'};
opt.process{1} = [2 2 2 0 0];
opt.process{2} = [3 3 3 2 2];

fprintf('Dual');
tic();
gurls(xTrain,yTrain,opt,1);
gurls(xTest,yTest,opt,2);
toc();

fprintf('Accuracy: %.3f\n',opt.perf.acc)

end

function gaussian(xTrain,xTest,yTrain,yTest,gurlsDir)
addpath(genpath(gurlsDir));
opt = gurls_defopt('Part C');

opt.seq = {'paramsel:siglam',...
           'kernel:rbf',...
           'rls:dual',...
           'predkernel:traintest',...
           'pred:dual',...
           'perf:macroavg'};
opt.process{1} = [2 2 2 0 0 0];
opt.process{2} = [3 3 3 2 2 2];

fprintf('Gaussian');
tic();
gurls(xTrain,yTrain,opt,1);
gurls(xTest,yTest,opt,2);
toc();
fprintf('Accuracy: %.3f\n',opt.perf.acc)
end