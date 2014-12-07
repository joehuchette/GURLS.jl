using RDatasets
using GURLS

iris = dataset("datasets","iris")

m = train(SepalWidth ~ PetalLength * PetalWidth,data = iris)

res = predict(m,iris)
