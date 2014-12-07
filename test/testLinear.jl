# A bare bones test to quickly route out syntax errors
using GURLS

#Load the data from problem set 1 of the course
function test_linear()
	xTrain = readcsv(Pkg.dir("GURLS") * "/data/xTrain.csv")
	yTrain = readcsv(Pkg.dir("GURLS") * "/data/yTrain.csv")
	xTest  = readcsv(Pkg.dir("GURLS") * "/data/xTest.csv")
	yTest  = readcsv(Pkg.dir("GURLS") * "/data/yTest.csv")

	#Center data
	xMeans = mean(xTrain,1)
	for i in 1:size(xTrain,1)
		xTrain[i,:] -= xMeans
		xTest[i,:]  -= xMeans
	end

	#Specify a training task which gives all of the specification of the model.
	#Form:
	#Training(xTrainData,yTrainData,kernel = KernelType(), rls = AlgorithmType())
	primal = Training(xTrain, yTrain, kernel = Linear(), rls = Primal())
	dual   = Training(xTrain, yTrain, kernel = Linear(), rls = Dual())

	#Specify a prediction task which takes a training tast and an independent variable data set.
	#Form:
	#Prediction(NameTrainingTask,xTestData)
	primalpred = Prediction(primal, xTest)
	dualpred   = Prediction(dual,   xTest)

	#Specify a performance task which takes a prediction task, a dependent variable data set, and a
	#performance metric to be used in the comparison.
	#Form:
	#Prediction(NameTrainingTask,xTestData)
	primalperf = Performance(primalpred, yTest, MacroAvg())
	dualperf   = Performance(dualpred,   yTest, MacroAvg())

	#Form experiment, a series of tasks to be performed.
	exprimal = Experiment(primal, primalpred, primalperf)
	exdual   = Experiment(dual, dualpred, dualperf)

	#Processes all of the tasks defined in the given Experiment in order.
	@time resp = process(exprimal)
	@time resd = process(exdual)

	println("Primal Performance: $(resp[primalperf])")
	println("Dual   Performance: $(resd[dualperf])")
end

test_linear()
test_linear()