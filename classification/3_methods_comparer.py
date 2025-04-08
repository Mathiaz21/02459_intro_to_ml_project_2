import numpy as np
import utils.serve_data as sd
import classification.models.logistic_regression as lr

from sklearn import model_selection
from utils.basic_operations import standardize_data


# Parameters
K_outer: int = 10
K_inner: int = 10
LAMBDA_START: float = .2
LAMBDA_STOP: float = 10.
LAMBDA_STEP: float = .2

# Setup
X = sd.get_features_dataset()
y = sd.get_targets()
X = standardize_data(X)
N, M = X.shape

LAMBDAS: list[float] = np.arange(stop=LAMBDA_STOP, start=LAMBDA_START, step=LAMBDA_STEP)
nb_lam: int = len(LAMBDAS)
outer_fold = model_selection.KFold(n_splits=K_outer, shuffle=True)


# Models
def get_baseline_error(y_train, y_test) -> float:
  largest_class: str = "B" if np.sum(y_train == "B") >= np.sum(y_test == "M") else "M"
  error: float = np.sum(y_test != largest_class) / len(y_test)
  return error


def get_regression_error(m, X_test, y_test) -> float: 
  y_est = m.predict(X_test)
  return np.sum(y_est != y_test) / len(y_test)

def compute_regression_errors(cursor: int, error_df: np.ndarray, X_train, y_train, X_test, y_test) -> None:
  for id, lam in enumerate(LAMBDAS):
    m = lr.generate_logistic_regression(regularization_parameter=lam).fit(X_train, y_train)
    error_df[cursor, id] = get_regression_error(m, X_test, y_test)

cursor_outer: int = 0
baseline_outer_error = np.empty((K_outer, 1))
regression_outer_error = np.empty((K_outer, 1))
regression_lambdas = np.empty((K_outer, 1))
for outer_train_index, outer_test_index in outer_fold.split(X):

  X_inner = X[outer_train_index]
  inner_fold = model_selection.KFold(n_splits=K_inner, shuffle=True)
  cursor_inner: int = 0
  regression_inner_error = np.empty((K_inner, nb_lam))
  baseline_inner_error = np.empty((K_inner, 1))
  for inner_train_index, inner_test_index in inner_fold.split(X_inner):

    X_train = X[inner_train_index,:]
    y_train = y[inner_train_index]
    X_test = X[inner_test_index,:]
    y_test = y[inner_test_index]

    baseline_inner_error[cursor_inner] = get_baseline_error(y_train=y_train, y_test=y_test)
    compute_regression_errors(cursor=cursor_inner, error_df=regression_inner_error, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    cursor_inner += 1
  
  baseline_outer_error[cursor_outer] = np.min(baseline_inner_error)
  regression_generalization_errors = np.mean(regression_inner_error, axis=1)
  regression_outer_error[cursor_outer] = np.min(regression_generalization_errors) # get the mean error for each inner fold, get the best lambda, save that
  regression_lambdas[cursor_outer] = LAMBDAS[np.argmin(regression_generalization_errors)]
  cursor_outer += 1

print(f"Baseline errors: {baseline_outer_error}")
print(f"Regression errors; {regression_outer_error}")
print(f"Best lambdas: {regression_lambdas}")

