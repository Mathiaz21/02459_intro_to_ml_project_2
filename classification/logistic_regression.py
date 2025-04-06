import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

X = pd.read_csv("../breast_cancer_wisconsin_features.csv") 
y = np.ravel(pd.read_csv("../breast_cancer_wisconsin_targets.csv"))
model = lm.LogisticRegression(max_iter=10000)

model.fit(X, y)

y_est = model.predict(X)
y_est_benign = model.predict_proba(X)[:, 0]

misclass_rate = np.sum(y_est != y) / float(len(y_est))

# BASELINE: returns benign
# Method 2: tree