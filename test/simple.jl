# A bare bones test to quickly route out syntax errors
using GURLS

expr = Experiment()

proc1 = TrainingProcess(rand(5,2), rand(5,1))
push!(expr, proc1)

res = process(expr)