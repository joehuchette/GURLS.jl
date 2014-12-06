# A bare bones test to quickly route out syntax errors
using GURLS

xTrain = readcsv(Pkg.dir("GURLS") * "/data/xTrain.csv")
yTrain = readcsv(Pkg.dir("GURLS") * "/data/yTrain.csv")
xTest  = readcsv(Pkg.dir("GURLS") * "/data/xTest.csv")
yTest  = readcsv(Pkg.dir("GURLS") * "/data/yTest.csv")

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

println("Primal Performance: $(res[primalperf])")
println("Dual   Performance: $(res[dualperf])")
