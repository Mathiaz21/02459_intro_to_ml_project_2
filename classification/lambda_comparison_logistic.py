import pandas as pd
import numpy as np
import sklearn.linear_model as lm

from utils.basic_operations import standardize_data

X = pd.read_csv("breast_cancer_wisconsin_features.csv")
y = np.ravel(pd.read_csv("breast_cancer_wisconsin_targets.csv"))
X = standardize_data(X)

lambdas: list[float] = [.1, .2, .5, 1., 2., 5., 10.]
Cs: list[float] = list(map(lambda x:  1/x, lambdas)) # Mapping to the paramenter taken by the sklearn model

for l in lambdas:
  model = lm.LogisticRegression(max_iter=10000, penalty="l2", C=l)
  model.fit(X, y)
