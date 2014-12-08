# A bare bones test to quickly route out syntax errors
using GURLS.Legacy

xTrain = readcsv(Pkg.dir("GURLS") * "/data/xTrain.csv")
yTrain = readcsv(Pkg.dir("GURLS") * "/data/yTrain.csv")
xTest  = readcsv(Pkg.dir("GURLS") * "/data/xTest.csv")
yTest  = readcsv(Pkg.dir("GURLS") * "/data/yTest.csv")

xMeans = mean(xTrain,1)
for i in 1:size(xTrain,1)
	xTrain[i,:] -= xMeans
	xTest[i,:]  -= xMeans
end

opt = defopt("simple legacy example")
opt.seq = ["paramsel:loocvdual", "rls:dual", "pred:dual", "perf:macroavg"]
opt.process[1] = [2,2,0,0]
opt.process[2] = [3,3,2,2]

gurls(xTrain, yTrain, opt, 1)
gurls(xTest,  yTest,  opt, 2)
