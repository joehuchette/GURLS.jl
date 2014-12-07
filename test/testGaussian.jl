# A bare bones test to quickly route out syntax errors
using GURLS

#Load the data from problem set 1 of the course
xTrain = readcsv(Pkg.dir("GURLS") * "/data/xTrain.csv")
yTrain = readcsv(Pkg.dir("GURLS") * "/data/yTrain.csv")
xTest  = readcsv(Pkg.dir("GURLS") * "/data/xTest.csv")
yTest  = readcsv(Pkg.dir("GURLS") * "/data/yTest.csv")

#Center data
xMeans = mean(xTrain,1)
for i in 1:size(xTrain,1)
	xTrain[i,:] -= xMeans
end
for i in 1:size(xTest,1)
	xTest[i,:] -= xMeans
end

#Specify a training task which gives all of the specification of the model.
#Form:
#Training(xTrainData,yTrainData,kernel = KernelType(), rls = AlgorithmType())
dual = Training(xTrain, yTrain, kernel = Gaussian(), rls = Dual())

#Specify a prediction task which takes a training tast and an independent variable data set.
#Form:
#Prediction(NameTrainingTask,xTestData)
pred = Prediction(dual, xTest)

#Specify a performance task which takes a prediction task, a dependent variable data set, and a
#performance metric to be used in the comparison.
#Form:
#Prediction(NameTrainingTask,xTestData)
perf = Performance(pred, yTest, MacroAvg())

#Form experiment, a series of tasks to be performed.
ex = Experiment(dual, pred, perf)

#Processes all of the tasks defined in the given Experiment in order.
res = process(ex)
m = res[dual].model

println(typeof(m))
println(typeof(xTest))

pred = sign(predict(m,xTest))
nCorrect = sum(pred .== yTest)
println("Gaussian: $(100*nCorrect/size(xTest,1))%")
