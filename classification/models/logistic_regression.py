import numpy as np
from sklearn.linear_model import LogisticRegression

LAMBDA_START: float = .2
LAMBDA_STOP: float = 10.
LAMBDA_STEP: float = .2
LAMBDAS: list[float] = np.arange(stop=LAMBDA_STOP, start=LAMBDA_START, step=LAMBDA_STEP)


def generate_logistic_regression(regularization_parameter: float = 1.) -> LogisticRegression: 
  adapted_parameter: float = 1/regularization_parameter
  return LogisticRegression(max_iter=10000, penalty="l2", C=adapted_parameter)