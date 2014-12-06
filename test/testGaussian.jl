# A bare bones test to quickly route out syntax errors
using GURLS

xTrain = readcsv(Pkg.dir("GURLS") * "/data/xTrain.csv")
yTrain = readcsv(Pkg.dir("GURLS") * "/data/yTrain.csv")
xTest  = readcsv(Pkg.dir("GURLS") * "/data/xTest.csv")
yTest  = readcsv(Pkg.dir("GURLS") * "/data/yTest.csv")

xMeans = mean(xTrain,1)
for i in 1:size(xTrain,1)
	xTrain[i,:] -= xMeans
	xTest[i,:] -= xMeans
end

dual = Training(xTrain, yTrain, kernel = Gaussian(), rls = Dual())
pred = Prediction(dual, xTest)
perf = Performance(pred, yTest, MacroAvg())

ex = Experiment(dual, pred, perf)

res = process(ex)
m = res[dual].model

pred = sign(predict(m,xTest))
nCorrect = sum(pred .== yTest)
println("Gaussian: $(100*nCorrect/size(xTest,1))%")
