using GURLS

expr = Experiment()

proc1 = TrainingProcess(rand(5,2), rand(5))
push!(expr, proc1)

