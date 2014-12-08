#A demo of our GURLS implementation on standard datasets
using RDatasets
using GURLS
using Resampling

df = dataset("COUNT","titanic")

# Switch to -1,1 classification data
df[:Survived] = (df[:Survived] - 0.5) * 2

df_train, df_test = splitrandom(df,0.7)

# Build the model. Uses R-like syntax for model building.
model = train(Survived ~ Age + Sex + Class, data = df_train, kernel = Linear(),
													 validation = LOOCV(),
													 method = Primal())

# Use our model to predict titanic deaths.
preds = predict(model,df_test)

# Evaluate our accuracy
accuracy = macroavg(preds,df_test[:Survived]) * 100
@printf("Prediction Accuracy: %.2f%%\n",accuracy)