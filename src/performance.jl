process(p::Performance{MacroAvg}, results) = macroavg(p.ytrue, results[p.pred])

macroavg(ytrue, ypred) = mean(sign(ytrue) .== sign(ypred))
