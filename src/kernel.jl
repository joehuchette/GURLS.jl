function buildKernel(train::Training{Linear,LOOCV,Dual})
	k = train.X * train.X'
	return k
end

function buildKernel(train::Training{Gaussian,LOOCV,Dual},sigma)
	(n,d) = size(train.X)

	k = zeros(n,n) # malloc
	coef = 1/(sqrt(2 * pi) * sigma) ^ d
	denom = 2 * sigma ^ 2

	# Only go over top half 
	for i in 1:n
		for j in 1:i
			k[i,j] = coef * exp(-norm(train.X[i,:] - train.X[j,:])/denom)
		end
	end

	k += k' - diagm(diag(k))

	return k
end

<<<<<<< HEAD
function getKernelSpace(k,train::Training)
	if k == Linear
		return [()]
	elseif k == Gaussian
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

		q = (sigmamax/sigmamin)^(1/(train.options.nSigma-1));
    	return sigmamin*(q.^(train.options.nSigma-1:-1:0));
	else
		error("Unknown Kernel")
	end
end
=======
getKernelSpace{P<:Paramsel}(train::Training{Linear,P,Dual}) = [()]
getKernelSpace{P<:Paramsel}(train::Training{Gaussian,P,Dual}) = error("Not yet implemented")
getKernelSpace(x) = error("Unknown kernel of type $(typeof(x))")
>>>>>>> 507cb556d4d0487002eb6b794b97987f2b2b2bb3
