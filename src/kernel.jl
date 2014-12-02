function buildKernel(data::TrainingProcess{Linear,LOOCV,Dual})
	(n,d) = size(data.X)
	k = zeros(n,n)

	println(k[1,1])
	# Only calculuate bottom half of matrix
	for i in 1:n
		for j in 1:i
			for n in 1:d
				k[i,j] += data.X[i,n] * data.X[j,n]
			end
		end
	end

	# Use symmetry to 
	k += k' - diagm(diag(k))

	return k
end