# function GInverseDiagonal(Q::Matrix{Float64}, L, lambda::Real, Z::Vector{Float64})
function GInverseDiagonal(Q::Matrix{Float64}, L, lambda::Real)
	nx, ny = size(Q, 1), size(Q, 2)
	Z = zeros(Float64, nx)
	# fill!(Z, 0.0)

	D = Q .^ 2

	d = 1 ./ (L + nx * lambda)
	Z = (Q .^ 2) * d

	# for i in 1:nx
	# 	for j in 1:ny
	# 		@inbounds Z[i] += Q[i,j]^2 / (L[j] + nx * lambda)
	# 	end
	# end
	return Z
end

# function rls_eigen(Q, L, Qy, lambda, n, C::Vector{Float64})
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

	L2 = L + n*lambda 
	diagL = diagm(1 ./ L2)

	# # ret = Array(Float64,n)
	# for i in 1:n
	# 	acc = 0.0
	# 	for j in 1:n
	# 		@inbounds acc += Q[i,j] * Qy[j]
	# 	end
	# 	@inbounds C[i] = acc / (n * lambda * L[i])
	# 	# @inbounds ret[i] = acc / (n * lambda * L[i])
	# end
	# # return ret
	# return C

	C = Q * diagL * Qy

	return C
end
