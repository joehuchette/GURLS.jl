importall DataFrames

type Model{T} <: AbstractResults
	innerModel::AbstractResults
	xMean::Vector{T}
	yMean::T
	formula::Formula
end

function train(model::Formula; data = [], kernel = Linear(), validation = LOOCV(), method = Primal())
	@assert !isempty(data) "No data specified-- call with data=[your dataframe]"

	modelMatrix = ModelMatrix(ModelFrame(model,data))

	x = modelMatrix.m[:,2:end] # get the data out of this, but ignore offset term. 

	y = (data[model.lhs]).data''

	xMean = mean(x,1)
	yMean = mean(y)

	for i in 1:size(x,1)
		x[i,:] -= xMean'' #double transpose to make row vector
		y[i] -= yMean
	end

	tr = Training(x,y,kernel = kernel, paramsel = validation, rls = method)

	ex = Experiment(tr)

	res = process(ex)

	return Model(res[tr].model,vec(xMean),yMean,model)
end

function predict(model::Model,data::DataFrames.DataFrame)
	modelMatrix = ModelMatrix(ModelFrame(model.formula,data))

	x = modelMatrix.m[:,2:end] # get the data out of this, but ignore offset term.

	for i in 1:size(x,1)
		x[i,:] -= model.xMean' #double transpose to make row vector
	end

	y = predict(model.innerModel,x)

	return y + model.yMean
end
