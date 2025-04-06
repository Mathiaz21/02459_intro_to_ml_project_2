import pandas as pd
import numpy as np
import classification.models.logistic_regression as lr

from utils.basic_operations import standardize_data

X = pd.read_csv("breast_cancer_wisconsin_features.csv")
y = np.ravel(pd.read_csv("breast_cancer_wisconsin_targets.csv"))
X = standardize_data(X)

lambdas: list[float] = [.1, .2, .5, 1., 2., 5., 10.]

for lam in lambdas:
  model = lr.generate_logistic_regression(regularization_parameter=lam)
  model.fit(X, y)