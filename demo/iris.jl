using Resampling
using RDatasets
using GURLS

# Load the iris dataset
iris = dataset("datasets","iris")

iris_train, iris_test = splitrandom(iris, 0.7)

# Regress SepalWidth on PetalLength, PetalWidth, and the Interaction between the two
m = train(SepalWidth ~ PetalLength * PetalWidth,data = iris_train, kernel = Linear(),
													 		 validation = LOOCV(),
													 	     method = Dual())

res = predict(m,iris_test)
