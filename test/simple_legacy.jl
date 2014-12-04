# A bare bones test to quickly route out syntax errors
using GURLS

xTrain = readcsv("../data/xTrain.csv")
yTrain = readcsv("../data/yTrain.csv")
xTest = readcsv("../data/xTest.csv")
yTest = readcsv("../data/yTest.csv")

xMeans = mean(xTrain,1)
for i in 1:size(xTrain,1)
	xTrain[i,:] -= xMeans
	xTest[i,:] -= xMeans
end

opt = defopt("simple legacy example")
opt.seq = ["paramsel:loocvdual", "rls:dual", "pred:dual", "perf:macroavg"]
opt.process[1] = [2,2,0,0]
opt.process[2] = [3,3,2,2]

gurls(xTrain, yTrain, opt, 1)
gurls(xTest,  yTest,  opt, 2)

m = res[1].model

pred = sign(predict(m,xTest))
nCorrect = sum(pred .== yTest)
println("Primal: $(100*nCorrect/size(xTest,1))%")

m = res[2].model
pred = sign(predict(m,xTest))
nCorrect = sum(pred .== yTest)
println("Dual: $(100* nCorrect/size(xTest,1))%")
