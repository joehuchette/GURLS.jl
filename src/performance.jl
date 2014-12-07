process(p::Performance{MacroAvg}, results) = macroavg(p.ytrue, results[p.pred])
process(p::Performance{RMSE}, results) = rmse(p.ytrue, results[p.pred])
process(p::Performance{AbsErr}, results) = abserr(p.ytrue, results[p.pred])

macroavg(ytrue, ypred) = mean(sign(ytrue) .== sign(ypred))

function rmse(ytrue, ypred)
	@assert size(ytrue,2) == size(ypred,2) == 1
	@assert (n = length(ytrue)) == length(ypred)
	val = 0.0
	for i in 1:n
		val += (ytrue[i] - ypred[i]) ^ 2
	end
	return √(val / n)
end

function abserr(ytrue, ypred)
	@assert size(ytrue,2) == size(ypred,2) == 1
	@assert (n = length(ytrue)) == length(ypred)
	val = 0.0
	for i in 1:n
		val += abs(ytrue[i] - ypred[i])
	end
	return val
end