# A bare bones test to quickly route out syntax errors
using GURLS

xTrain = readcsv("../data/xTrain.csv")
yTrain = readcsv("../data/yTrain.csv")
xTest  = readcsv("../data/xTest.csv")
yTest  = readcsv("../data/yTest.csv")

xMeans = mean(xTrain,1)
for i in 1:size(xTrain,1)
	xTrain[i,:] -= xMeans
	xTest[i,:]  -= xMeans
end

primal = Training(xTrain, yTrain, kernel = Linear(), rls = Primal())
dual   = Training(xTrain, yTrain, kernel = Linear(), rls = Dual())
primalpred = Prediction(primal, xTest)
dualpred   = Prediction(dual,   xTest)
primalperf = Performance(primalpred, yTest, MacroAvg())
dualperf   = Performance(dualpred,   yTest, MacroAvg())

ex = Experiment(primal, dual, primalpred, dualpred, primalperf, dualperf)

res = process(ex)
