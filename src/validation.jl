function LOOCV{T<:TrainingProcess}(train::T,lambda,args...)
	# Zach, you fill in here! Look at the model.jl file for model building/testing functions. 
	# args is an array of arguments we'll need to build the models. For linear, this will
	# contain one argument, the value for lambda. 
	
	n=size(train.X,1)
	sse = 0.0

	for i in 1:n
		if i==1
			Xmod = train.X[2:end,:]
			ymod = train.y[2:end]
		elseif i==n
			Xmod = train.X[1:end-1,:]
			ymod = train.y[1:end-1]
		else
			Xmod = train.X[[1:i-1,i+1:end],:]
			ymod = train.y[[1:i-1,i+1:end]]
		end

		tp=TrainingProcess(Xmod, ymod; kernel=Linear, paramsel=LOOCV, rls=Primal)
		submodel = buildModel(tp,lambda)
		sse += (predict(submodel,train.X[i,:])[1] - train.y[i])^2
	end

	return sse/n
end