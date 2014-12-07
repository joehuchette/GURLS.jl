buildKernel(train::Training{Linear,LOOCV,Dual}) = train.X * train.X'

function buildKernel(train::Training{Gaussian,LOOCV,Dual},sigma)
	X::Matrix{Float64} = train.X
	n, d = size(X)

	k = zeros(Float64,n,n) # malloc
	# coef = 1/(sqrt(2 * pi) * sigma) ^ d
	denom = 2 * sigma ^ 2

	# Only go over top half 
	for i in 1:n
		for j in 1:i
			acc = 0.0
			for ℓ in 1:d
				@inbounds acc += (X[i,ℓ] - X[j,ℓ])^2
			end
			tmp = exp(-acc / denom)
			k[i,j] = tmp
			k[j,i] = tmp
		end
	end
	return k
end

function buildCrossXKernel{R<:Real}(model::GaussianModel,X::Array{R,2})
	nTrain, d = size(model.X)
	nTest = size(X,1)
	out = zeros(R, nTest, nTrain)
	
	denom = 2 * model.sigma ^ 2

	for i in 1:nTest
		for j in i:nTrain
			out[i,j] = exp(-norm(model.X[j,:] - X[i,:])^2/denom)
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

