function validateDual{R<:Real}(Q::Matrix{R},L::Vector{R},Qy::Vector{R},lambda::R,y::Vector{R})
	# Computes sum of square LOOE given the singular value decomposition of the
	# kernel matrix
	# 
	# INPUTS
	# -Q: eigenvectors of the kernel matrix
	# -L: eigenvalues of the kernel matrix
	# -Qy: result of the matrix multiplication of the transpose of Q times the
	#       labels vector Y (Q*Y)
	# -lambda: regularization parameter
	# 
	# OUTPUT:
	# -perfT: 1xT 2d-array of sum of square LOOE per class

	n, T = size(y,1), size(y,2)
	C = zeros(Float64,n)
	rls_eigen(Q,L,Qy,lambda,n,C)
	Z = zeros(Float64,n)
	GInverseDiagonal(Q,L,lambda,Z)

	@assert (size(C,1),size(C,2)) == (n,T) "size(C) = $(size(C))"
	@assert size(Z) == (n,)

	perf = zeros(n,T)
	for j in 1:T
		@simd for i in 1:n
			@inbounds perf[i,j] = (C[i,j] / Z[i])^2
		end
	end

	return sum(perf,1)[1]
end


function validatePrimal(LEFT,RIGHT,L,lambda,y::Vector)
	# Computes sum of square LOOE given the singular value decomposition of the
	# kernel matrix
	# 
	# INPUTS
	# -LEFT: X*Q  (data*eigenvectors of the kernel matrix)
	# -RIGHT: Q^-1 * X' * y 
	# -L: eigenvalues of the kernel matrix
	# -lambda: regularization parameter
	# 
	# OUTPUT:
	# -perfT: 1xT 2d-array of sum of square LOOE per class

	n, T = size(y,1), size(y,2)
	d1, d2 = size(LEFT,1), size(LEFT,2)
	@assert d1 == n

	# LL = diagm((L + (n*lambda)).^(-1))
	# num = y - LEFT*LL*RIGHT
	num = copy(y)
	@simd for i in 1:d1
		for j in 1:d2
			@inbounds num[i] -= LEFT[i,j] * RIGHT[j] / (L[j] + (n*lambda))
		end
	end

	den = fill(1.0, n)

	@simd for j in 1:n
		for k in 1:d2
			@inbounds den[j] -= L[k] * LEFT[j,k]^2 
		end
	end

	perfT = 0.0
	@simd for i in 1:n
		@inbounds perfT += (num[i] / den[i])^2
	end

	return perfT
end



