import numpy as np
import utils.serve_data as sd

from sklearn import model_selection
from utils.basic_operations import standardize_data

# Parameters
K_outer: int = 10
K_inner: int = 10
LAMBDA_START: float = .2
LAMBDA_STOP: float = 10.
LAMBDA_STEP: float = .2

# Models

def get_baseline_error(y_train, y_test):
  largest_class: str = "B" if np.sum(y_train == "B") >= np.sum(y_test == "M") else "M"
  error: float = np.sum(y_test != largest_class) / len(y_test)
  return error


# Setup
X = sd.get_features_dataset()
y = sd.get_targets()
X = standardize_data(X)
N, M = X.shape

lambdas: list[float] = np.arange(stop=LAMBDA_STOP, start=LAMBDA_START, step=LAMBDA_STEP)
nb_lam: int = len(lambdas)
outer_fold = model_selection.KFold(n_splits=K_outer, shuffle=True)

for outer_train_index, outer_test_index in outer_fold.split(X):

  X_inner = X[outer_train_index]
  inner_fold = model_selection.KFold(n_splits=K_inner, shuffle=True)
  min_baseline_error = None
  for inner_train_index, inner_test_index in inner_fold.split(X_inner):

    X_train = X[inner_train_index,:]
    y_train = y[inner_train_index]
    X_test = X[inner_test_index,:]
    y_test = y[inner_test_index]

    baseline_error: float = get_baseline_error(y_train=y_train, y_test=y_test)
    min_baseline_error = baseline_error if min_baseline_error is None or min_baseline_error > baseline_error else min_baseline_error

  print(f"Baseline error: {min_baseline_error}")


