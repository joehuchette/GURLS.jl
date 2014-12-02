# A bare bones test to quickly route out syntax errors
using GURLS

expr = Experiment()

proc1 = TrainingProcess(rand(5,2), rand(5,1))
push!(expr, proc1)

proc2 = TrainingProcess(rand(5,2),rand(5,1),rls=Dual)
push!(expr, proc2)

res = process(expr)