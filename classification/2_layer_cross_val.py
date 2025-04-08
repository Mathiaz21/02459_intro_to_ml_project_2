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

OUTPUT_NAME: str = "classification_2lcv"

# Setup
X = sd.get_features_dataset()
y = sd.get_targets()
X = standardize_data(X)
N, M = X.shape

nb_lam: int = len(lr.LAMBDAS)
nb_alphas: int = len(tree.ALPHAS)
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
  for id, lam in enumerate(lr.LAMBDAS):
    m = lr.generate_logistic_regression(regularization_parameter=lam).fit(X_train, y_train)
    error_df[cursor, id] = get_model_error(m, X_test, y_test)


def compute_tree_errors(cursor: int, error_df: np.ndarray, X_train, y_train, X_test, y_test) -> None:
  for id, mss in enumerate(tree.ALPHAS):
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

  X_inner = X[outer_train_index, :]
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

  y_inner = y[outer_train_index]
  X_outer = X[outer_test_index, :]
  y_outer = y[outer_test_index]

  baseline_outer_errors[cursor_outer] = np.min(baseline_inner_errors)

  regression_avg_errors = np.mean(regression_inner_errors, axis=0)
  best_lambda = lr.LAMBDAS[np.argmin(regression_avg_errors)]
  best_regression_lambdas[cursor_outer] = best_lambda
  outer_logreg_model = lr.generate_logistic_regression(regularization_parameter=best_lambda).fit(X_inner, y_inner)
  regression_outer_errors[cursor_outer] = get_model_error(outer_logreg_model, X_outer, y_outer)

  tree_generalization_errors = np.mean(tree_inner_errors, axis=0)
  best_alpha: float = tree.ALPHAS[np.argmin(tree_generalization_errors)]
  best_tree_alphas[cursor_outer] = best_alpha
  outer_tree_model = tree.generate_decision_tree(ccp_alpha=best_alpha).fit(X_inner, y_inner)
  tree_outer_errors[cursor_outer] = get_model_error(outer_tree_model, X_outer, y_outer)

  cursor_outer += 1

print(f"Baseline errors: {baseline_outer_errors}")
print(f"Regression errors; {regression_outer_errors}")
print(f"Best lambdas: {best_regression_lambdas}")
print(f"Tree errors: {tree_outer_errors}")
print(f"Best parameters: {best_tree_alphas}")

results_df = pd.DataFrame({
  "outer_fold": np.arange(1, len(baseline_outer_errors)+1),
  "best_alphas": best_tree_alphas,
  "tree_errors": tree_outer_errors,
  "best_lambdas": best_regression_lambdas,
  "regression_errors": regression_outer_errors,
  "baseline_errors": baseline_outer_errors,
})

save_results(results_df=results_df, output_name=OUTPUT_NAME)