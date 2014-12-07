# A bare bones test to quickly route out syntax errors
using GURLS

function test_gaussian()
	xTrain = readcsv(Pkg.dir("GURLS") * "/data/xTrain.csv")
	yTrain = readcsv(Pkg.dir("GURLS") * "/data/yTrain.csv")
	xTest  = readcsv(Pkg.dir("GURLS") * "/data/xTest.csv")
	yTest  = readcsv(Pkg.dir("GURLS") * "/data/yTest.csv")
	# x = readcsv("x.csv")
	# y = readcsv("y.csv")

	# p = randperm(150)
	# x = x[p,:]
	# y = y[p]

	# xTrain = x[1:100,:]
	# yTrain = y[1:100]

	# xTest = x[101:150,:]
	# yTest = y[101:150]

	xMeans = mean(xTrain,1)
	for i in 1:size(xTrain,1)
		xTrain[i,:] -= xMeans
	end
	for i in 1:size(xTest,1)
		xTest[i,:] -= xMeans
	end

	# xTrain = xTrain[1:4,:]
	# yTrain = yTrain[1:4,:]
	# xTest = xTest[1:4,:]
	# yTest = yTest[1:4,:]

	dual = Training(xTrain, yTrain, kernel = Gaussian(), rls = Dual())
	pred = Prediction(dual, xTest)
	perf = Performance(pred, yTest, MacroAvg())

	ex = Experiment(dual, pred, perf)

	@time res = process(ex)
	m = res[dual].model

	println("Gaussian: $(100*res[perf])%")
end

test_gaussian()
test_gaussian()