function buildKernel(data::Training{Linear,LOOCV,Dual})
	k = data.X * data.X'
	return k
end

function buildKernel(data::Training{Gaussian,LOOCV,Dual},sigma)
	(n,d) = size(data.X)

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

getKernelSpace(::Linear) = [()]
getKernelSpace(::Gaussian) = error("Not yet implemented")
getKernelSpace(x) = error("Unknown kernel of type $(typeof(x))")

