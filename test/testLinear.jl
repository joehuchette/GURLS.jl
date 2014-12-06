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

# ex = Experiment(primal, dual, primalpred, dualpred)
# ex = Experiment(primal, dual)
ex = Experiment()
push!(ex,primal)
push!(ex,dual)
res = process(ex)

m = res[1].model

pred = sign(predict(m,xTest))
nCorrect = sum(pred .== yTest)
println("Primal: $(100*nCorrect/size(xTest,1))%")

m = res[2].model
pred = sign(predict(m,xTest))
nCorrect = sum(pred .== yTest)
println("Dual: $(100* nCorrect/size(xTest,1))%")
