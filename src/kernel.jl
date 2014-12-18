importall Distances

buildKernel(train::Training{Linear,LOOCV,Dual}) = train.X * train.X'

function buildKernel{G<:Gaussian}(train::Training{G,LOOCV,Dual},sigma)
	denom = 2 * sigma ^ 2

	train.kernel.k = exp(-train.kernel.dists ./ denom)

	return train.kernel.k
end

function buildCrossXKernel{R<:Real}(model::GaussianModel,X::Array{R,2})
	denom = 2 * model.sigma ^ 2

	k = pairwise(SqEuclidean(),X',model.X')
	return exp(-k./denom)
end



getKernelSpace{P<:Paramsel}(train::Training{Linear,P,Dual}) = [()]

function getKernelSpace{G<:Gaussian,P<:Paramsel}(train::Training{G,P,Dual})
	# kerneldistance = square_distance(train.X',train.X')
	train.kernel.dists = pairwise(SqEuclidean(),train.X')
	n = size(train.kernel.dists,1)

	# dists = sort(vec(tril(train.kernel.dists,-1)))[(n^2+n+2)/2:end]
	dists = sort(vec(train.kernel.dists))
	sigmamin = sqrt(dists[round(0.5 + length(dists) * 0.01)])
	sigmamax = sqrt(maximum(dists))

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

