using GURLS

function benchmark_test()
	xTrain = readcsv(ARGS[1])
	xTest = readcsv(ARGS[2])
	yTrain = readcsv(ARGS[3])
	yTest = readcsv(ARGS[4])

	primal = Training(xTrain, yTrain, kernel = Linear(), rls = Primal())
	primalpred = Prediction(primal, xTest)
	primalperf = Performance(primalpred, yTest, MacroAvg())
	ex = Experiment(primal,primalpred,primalperf)

	tic()
	res = process(ex)
	print("Primal: ")
	toc()
	println(res[primalperf])

	primal = Training(xTrain, yTrain, kernel = Linear(), rls = Dual())
	primalpred = Prediction(primal, xTest)
	primalperf = Performance(primalpred, yTest, MacroAvg())
	ex = Experiment(primal,primalpred,primalperf)

	tic()
	res = process(ex)
	print("Dual: ")
	toc()
	println(res[primalperf])

	primal = Training(xTrain, yTrain, kernel = Gaussian(), rls = Dual())
	primalpred = Prediction(primal, xTest)
	primalperf = Performance(primalpred, yTest, MacroAvg())
	ex = Experiment(primal,primalpred,primalperf)

	tic()
	res = process(ex)
	print("Gaussian: ")
	toc()
	println(res[primalperf])

end

benchmark_test()
benchmark_test()