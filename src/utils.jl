function GInverseDiagonal(Q::Matrix{Float64}, L, lambda::Real)
	nx, ny = size(Q, 1), size(Q, 2)
	Z = zeros(Float64, nx)

	# D = Q .^ 2

	# d = 1 ./ (L + n * lambda)
	# Z = (Q .^ 2) * d

	for i in 1:nx
		for j in 1:ny
			@inbounds Z[i] += Q[i,j]^2 / (L[j] + nx * lambda)
		end
	end
	return Z
end

function GInverseDiagonal(Q::Matrix{Float64}, L, lambda)
	n = size(Q,1)
	t = size(lambda,2)
	Z = zeros(Float64, n, t)

	D = Q .^ 2
	for i = 1:t
	    d = L + (n*lambda[i])
	    d = 1 ./ d
	    Z[:,i] = D*d
	end
	
	return Z	
end

function rls_eigen(Q, L, Qy, lambda, n)
	# Computes RLS estimator given the singular value decomposition of the
	# kernel matrix
	# 
	# INPUTS
	# -Q: eigenvectors of the kernel matrix
	# -L: eigenvalues of the kernel matrix
	# -Qy: result of the matrix multiplication of the transpose of Q times the
	#       labels vector Y (Q*Y)
	# -lambda: regularization parameter
	# -n: number of training samples
	# 
	# OUTPUT:
	# -C: rls coefficient vector

	# L += n*lambda
	# diagL = diagm(1 ./ L)

	ret = Array(Float64,n)
	for i in 1:n
		acc = 0.0
		for j in 1:n
			@inbounds acc += Q[i,j] * Qy[j]
		end
		@inbounds ret[i] = acc / (n * lambda * L[i])
	end
	return ret

	# return Q * diagL * Qy
end
