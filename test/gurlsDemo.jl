#A demo of our GURLS implementation on standard datasets
using RDatasets
using GURLS
using DataFrames

df = dataset("COUNT","titanic")

# Switch to -1,1 classification data
df[:Survived] = (df[:Survived] - 0.5) * 2

# Build the model. Uses R-like sytax for model building.
model = train(Survived ~ Age + Sex + Class, data = df, kernel = Linear(),
													 validation = LOOCV(),
													 method = Primal())

# Use our model to predict titanic deaths.
preds = predict(model,df)

# Evaluate our accuracy
accuracy = macroavg(preds,df[:Survived])
println("$accuracy % Correct!")