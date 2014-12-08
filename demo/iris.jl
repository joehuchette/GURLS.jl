using RDatasets
using GURLS

# Load the iris dataset
iris = dataset("datasets","iris")

# Regress SepalWidth on PetalLength, PetalWidth, and the Interaction between the two
m = train(SepalWidth ~ PetalLength * PetalWidth,data = iris, kernel = Linear(),
													 		 validation = LOOCV(),
													 	     method = Dual()))

res = predict(m,iris)
