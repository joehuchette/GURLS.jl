#A demo of our GURLS implementation on standard datasets
using GURLS
using DataFrames
using RDatasets

titanicDF = dataset("count","titanic")
n = size(titanicDF,1)
ntrain = round(n*3/4)

randp = randperm(n)

xTrain = convert(Matrix{Float64}, array(titanicDF[randp[1:ntrain],2:end]))
yTrain = convert(Vector{Float64}, array(titanicDF[randp[1:ntrain],1]))
xTest  = convert(Matrix{Float64}, array(titanicDF[randp[ntrain+1:end],2:end]))
yTest  = convert(Vector{Float64}, array(titanicDF[randp[ntrain+1:end],1]))

xMeans = mean(xTrain,1)
for i in 1:size(xTrain,1)
	xTrain[i,:] -= xMeans
end
for i in 1:size(xTest,1)
	xTest[i,:] -= xMeans
end

dual = Training(xTrain, yTrain, kernel = Gaussian(), rls = Dual())
pred = Prediction(dual, xTest)
perf = Performance(pred, yTest, MacroAvg())

ex = Experiment(dual, pred, perf)

@time res = process(ex)

println("Gaussian: $(100*res[perf])%")