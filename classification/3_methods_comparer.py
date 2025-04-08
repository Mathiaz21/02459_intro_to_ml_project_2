import numpy as np
import pandas as pd
import utils.serve_data as sd
import classification.models.logistic_regression as lr
import classification.models.tree as tree

from sklearn import model_selection
from utils.basic_operations import standardize_data
from utils.save_results import save_results



# Parameters
K_outer: int = 10
K_inner: int = 10
LAMBDA_START: float = .2
LAMBDA_STOP: float = 10.
LAMBDA_STEP: float = .2

ALPHA_START: float = 0.
ALPHA_STOP: float = .05
ALPHA_STEP: float = .005

OUTPUT_NAME: str = "classification_comparison"

# Setup
X = sd.get_features_dataset()
y = sd.get_targets()
X = standardize_data(X)
N, M = X.shape

LAMBDAS: list[float] = np.arange(stop=LAMBDA_STOP, start=LAMBDA_START, step=LAMBDA_STEP)
ALPHAS: list[int] = np.arange(stop=ALPHA_STOP, start=ALPHA_START, step=ALPHA_STEP)
nb_lam: int = len(LAMBDAS)
nb_alphas: int = len(ALPHAS)
outer_fold = model_selection.KFold(n_splits=K_outer, shuffle=True)



# Models
def get_baseline_error(y_train, y_test) -> float:
  largest_class: str = "B" if np.sum(y_train == "B") >= np.sum(y_test == "M") else "M"
  error: float = np.sum(y_test != largest_class) / len(y_test)
  return error


def get_model_error(m, X_test, y_test) -> float: 
  y_est = m.predict(X_test)
  return np.sum(y_est != y_test) / len(y_test)


def compute_regression_errors(cursor: int, error_df: np.ndarray, X_train, y_train, X_test, y_test) -> None:
  for id, lam in enumerate(LAMBDAS):
    m = lr.generate_logistic_regression(regularization_parameter=lam).fit(X_train, y_train)
    error_df[cursor, id] = get_model_error(m, X_test, y_test)


def compute_tree_errors(cursor: int, error_df: np.ndarray, X_train, y_train, X_test, y_test) -> None:
  for id, mss in enumerate(ALPHAS):
    m = tree.generate_decision_tree(ccp_alpha=mss).fit(X_train, y_train)
    error_df[cursor, id] = get_model_error(m, X_test, y_test)


# Main script
cursor_outer: int = 0
baseline_outer_errors = np.empty(K_outer)
regression_outer_errors = np.empty(K_outer)
best_regression_lambdas = np.empty(K_outer)
tree_outer_errors = np.empty(K_outer)
best_tree_alphas = np.empty(K_outer)
for outer_train_index, outer_test_index in outer_fold.split(X):

  X_inner = X[outer_train_index]
  inner_fold = model_selection.KFold(n_splits=K_inner, shuffle=True)
  cursor_inner: int = 0
  baseline_inner_errors = np.empty((K_inner, 1))
  regression_inner_errors = np.empty((K_inner, nb_lam))
  tree_inner_errors = np.empty((K_inner, nb_alphas))
  for inner_train_index, inner_test_index in inner_fold.split(X_inner):

    X_train = X[inner_train_index,:]
    y_train = y[inner_train_index]
    X_test = X[inner_test_index,:]
    y_test = y[inner_test_index]

    baseline_inner_errors[cursor_inner] = get_baseline_error(y_train=y_train, y_test=y_test)
    compute_regression_errors(cursor=cursor_inner, error_df=regression_inner_errors, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    compute_tree_errors(cursor=cursor_inner, error_df=tree_inner_errors, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    cursor_inner += 1
  
  baseline_outer_errors[cursor_outer] = np.min(baseline_inner_errors)

  regression_generalization_errors = np.mean(regression_inner_errors, axis=0)
  regression_outer_errors[cursor_outer] = np.min(regression_generalization_errors)
  best_regression_lambdas[cursor_outer] = LAMBDAS[np.argmin(regression_generalization_errors)]

  tree_generalization_errors = np.mean(tree_inner_errors, axis=0)
  tree_outer_errors[cursor_outer] = np.min(tree_generalization_errors)
  best_tree_alphas[cursor_outer] = ALPHAS[np.argmin(tree_generalization_errors)]

  cursor_outer += 1

print(f"Baseline errors: {baseline_outer_errors}")
print(f"Regression errors; {regression_outer_errors}")
print(f"Best lambdas: {best_regression_lambdas}")
print(f"Tree errors: {tree_outer_errors}")
print(f"Best parameters: {best_tree_alphas}")

results_df = pd.DataFrame({
  "Baseline errors": baseline_outer_errors,
  "Regression errors": regression_outer_errors,
  "Best lambdas": best_regression_lambdas,
  "Tree errors": tree_outer_errors,
  "Best alphas": best_tree_alphas,
})

save_results(results_df=results_df, output_name=OUTPUT_NAME)