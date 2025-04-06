from sklearn.linear_model import LogisticRegression



def generate_logistic_regression(regularization_parameter: float = 1.): 
  adapted_parameter: float = 1/regularization_parameter
  return LogisticRegression(max_iter=10000, penalty="l2", C=adapted_parameter)