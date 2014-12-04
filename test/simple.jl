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

expr = Experiment()

proc1 = TrainingProcess(xTrain,yTrain,kernel = Linear, rls = Primal)
push!(expr, proc1)

proc2 = TrainingProcess(xTrain,yTrain,kernel = Linear, rls = Dual)
push!(expr, proc2)

res = process(expr)

m = res[1].model

pred = sign(predict(m,xTest))
nCorrect = sum(pred .== yTest)
println("Primal: $(100*nCorrect/size(xTest,1))%")

m = res[2].model
pred = sign(predict(m,xTest))
nCorrect = sum(pred .== yTest)
println("Dual: $(100* nCorrect/size(xTest,1))%")