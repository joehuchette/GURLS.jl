# A bare bones test to quickly route out syntax errors
using GURLS

eyedata = readcsv(Pkg.dir("GURLS") * "/data/EyeState.csv")

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
end
for i in 1:size(xTest,1)
	xTest[i,:] -= xMeans
end

primal = Training(xTrain, yTrain, kernel = Linear(), rls = Primal())
dual   = Training(xTrain, yTrain, kernel = Linear(), rls = Dual())
primalpred = Prediction(primal, xTest)
dualpred   = Prediction(dual,   xTest)
primalperf = Performance(primalpred, yTest, MacroAvg())
dualperf   = Performance(dualpred,   yTest, MacroAvg())

ex = Experiment(primal, dual, primalpred, dualpred, primalperf, dualperf)

res = process(ex)

println("Primal Performance: $(res[primalperf])")
println("Dual   Performance: $(res[dualperf])")
