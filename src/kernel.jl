function buildKernel(train::Training{Linear,LOOCV,Dual})
	k = train.X * train.X'
	return k
end

function buildKernel(train::Training{Gaussian,LOOCV,Dual},sigma)
	(n,d) = size(train.X)

	k = zeros(n,n) # malloc
	# coef = 1/(sqrt(2 * pi) * sigma) ^ d
	denom = 2 * sigma ^ 2

	# Only go over top half 
	for i in 1:n
		for j in 1:i
			k[i,j] = exp(-norm(train.X[i,:] - train.X[j,:])/denom)
		end
	end

	k += k' - diagm(diag(k))

	return k
end

function buildCrossXKernel{R<:Real}(model::GaussianModel,X::Array{R,2})
	(nTrain,d) = size(model.X)
	nTest = size(X,1)

	out = zeros(nTest,nTrain)

	# coef = 1/(sqrt(2 * pi) * model.sigma) ^ d
	denom = 2 * model.sigma ^ 2

	for i in 1:nTrain
		for j in 1:nTest
			out[i,j] =  exp(-norm(model.X[i,:] - X[j,:])/denom)
		end
	end

	return out
end



getKernelSpace{P<:Paramsel}(train::Training{Linear,P,Dual}) = [()]

function getKernelSpace{P<:Paramsel}(train::Training{Gaussian,P,Dual})
	kerneldistance = square_distance(train.X',train.X')
	n = size(kerneldistance,1)

	dists = sort(vec(tril(kerneldistance,-1)))[(n^2+n+2)/2:end]
	sigmamin = dists[round(0.5 + length(dists) * 0.1)]
	sigmamax = maximum(dists)

	if sigmamin <= 0
		sigmamin = eps();
	end
	if sigmamax <= 0
		sigmamax = eps();
	end

	q = (sigmamax/sigmamin)^(1/(num_sigma(train.kernel)-1));
	return sigmamin*(q.^(num_sigma(train.kernel)-1:-1:0));

end

getKernelSpace(x) = error("Unknown kernel of type $(typeof(x))")

