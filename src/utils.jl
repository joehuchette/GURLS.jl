function GInverseDiagonal(Q::Matrix{Float64}, L::Vector{Float64}, lambda::Real, Z::Vector{Float64})
# function GInverseDiagonal(Q::Matrix{Float64}, L, lambda::Real)
	(nx, ny) = size(Q)
	# Z = zeros(Float64,nx)
	# fill!(Z, 0.0)

	# D = Q .^ 2

	# d = 1 ./ (L + nx * lambda)
	# Z = (Q .^ 2) * d

	for i in 1:nx
		@simd for j in 1:ny
			@inbounds Z[i] += Q[i,j] * Q[i,j] / (L[j] + nx * lambda)
		end
	end
	return
end

function rls_eigen(Q, L, Qy, lambda, n, C::Vector{Float64})
	# function rls_eigen(Q, L, Qy, lambda, n)
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

	# L2 = L + n*lambda 
	# diagL = diagm(1 ./ L2)

	for i in 1:n
		@simd for j in 1:n
			@inbounds C[i] += Q[i,j] * Qy[j] / (n * lambda + L[j])
		end
	end

	# C = Q * diagL * Qy

	return
end
