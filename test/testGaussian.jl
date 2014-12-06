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

#primal = Training(xTrain, yTrain, kernel = Linear(), rls = Primal())
#push!(expr, primal)

dual = Training(xTrain, yTrain, kernel = Gaussian(), rls = Dual())

expr = Experiment(dual)

res = process(expr)

m = res[dual].model

pred = sign(predict(m,xTest))
nCorrect = sum(pred .== yTest)
println("Gaussian: $(100*nCorrect/size(xTest,1))%")
