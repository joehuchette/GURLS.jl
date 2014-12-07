# A bare bones test to quickly route out syntax errors
using GURLS

eyedata = readcsv(Pkg.dir("GURLS") * "/data/EyeState.csv")

function test_big_gaussian()

	n = size(eyedata,1)
	randp = randperm(n)

	#14980 possible points in the data set
	ntrain = 3000
	ntest = 1000

	xTrain = convert(Matrix{Float64}, eyedata[randp[1:ntrain],1:end-1])
	yTrain = convert(Vector{Float64}, eyedata[randp[1:ntrain],end])
	xTest  = convert(Matrix{Float64}, eyedata[randp[ntrain+1:ntrain+ntest],1:end-1])
	yTest  = convert(Vector{Float64}, eyedata[randp[ntrain+1:ntrain+ntest],end])

	xMeans = mean(xTrain,1)
	for i in 1:size(xTrain,1)
		xTrain[i,:] -= xMeans
		if yTrain[i]==0
			yTrain[i] = -1
		end
	end
	for i in 1:size(xTest,1)
		xTest[i,:] -= xMeans
		if yTest[i]==0
			yTest[i] = -1
		end
	end

	dual = Training(xTrain, yTrain, kernel = Gaussian(), rls = Dual())
	pred = Prediction(dual, xTest)
	perf = Performance(pred, yTest, MacroAvg())

	ex = Experiment(dual, pred, perf)

	@time res = process(ex)

	println("Gaussian: $(100*res[perf])%")
end

run_big_gaussian()
