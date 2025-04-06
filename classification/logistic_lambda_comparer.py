import numpy as np
import classification.models.logistic_regression as lr
import matplotlib.pyplot as plt
import utils.serve_data as sd

from sklearn import model_selection
from utils.basic_operations import standardize_data

# Parameters
K: int = 57
LAMBDA_START: float = .2
LAMBDA_STOP: float = 10.
LAMBDA_STEP: float = .2


X = sd.get_features_dataset()
y = sd.get_targets()
X = standardize_data(X)
N, M = X.shape

lambdas: list[float] = np.arange(stop=LAMBDA_STOP, start=LAMBDA_START, step=LAMBDA_STEP)
nb_lam: int = len(lambdas)
CV = model_selection.KFold(n_splits=K, shuffle=True)

# Initialize variables
outer_training_errors = np.empty((nb_lam,1))
outer_test_errors = np.empty((nb_lam,1))




for id, lam in enumerate(lambdas):

  inner_training_errors = np.empty((K, 1))
  inner_test_errors = np.empty((K, 1))
  k: int = 0
  for train_index, test_index in CV.split(X):
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    # Compute squared error with all features selected (no feature selection)
    m = lr.generate_logistic_regression(regularization_parameter=lam).fit(X_train, y_train)
    inner_training_errors[k] = np.sum(y_train != m.predict(X_train)) / y_train.shape[0] * 100
    inner_test_errors[k] = np.sum(y_test != m.predict(X_test)) / y_test.shape[0] * 100
    k += 1

  outer_training_errors[id] = np.mean(inner_training_errors)
  outer_test_errors[id] = np.mean(inner_test_errors)


# Display results
f = plt.figure()
plt.plot(lambdas, outer_training_errors)
plt.plot(lambdas, outer_test_errors)
plt.xlabel("Regularization parameter lambda")
plt.ylabel("Error (misclassification rate)")
plt.legend(["Training Error", "Testing Error"])

plt.show()
