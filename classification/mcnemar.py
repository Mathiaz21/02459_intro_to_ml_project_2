import numpy as np
import utils.serve_data as sd
import classification.models.logistic_regression as lr
import classification.models.tree as tree

from sklearn import model_selection
from utils.basic_operations import standardize_data
from dtuimldmtools import jeffrey_interval
from dtuimldmtools import mcnemar as dtu_mcnemar


# Parameters
ALPHA_CONFIDENCE: float = .05
K: int = 100

# Setup
X = sd.get_features_dataset()
# Mapping "B" to 0 and "M" to 1
y = np.vectorize(lambda s : 0 if s == "B" else 1)(sd.get_targets())
X = standardize_data(X)
N, M = X.shape

# Main Script
outer_cursor: int = 0
baseline_observations = np.empty(N)
tree_observations = np.empty(N)
logreg_observations = np.empty(N)
K_Fold = model_selection.KFold(n_splits=K, shuffle=True)
for train_index, test_index in K_Fold.split(X):

  X_train = X[train_index, :]
  y_train = y[train_index]
  X_test = X[test_index, :]
  y_test = y[test_index]

  baseline_class: str = 0 if np.sum(y_train == 0) >= np.sum(y_train == 1) else 1
  dectree = tree.generate_decision_tree().fit(X_train, y_train)
  logreg = lr.generate_logistic_regression().fit(X_train, y_train)

  baseline_observations[test_index] = baseline_class
  tree_observations[test_index] = dectree.predict(X_test)
  logreg_observations[test_index] = logreg.predict(X_test)

# Solo evaluation
[thetahat_bl, CIA_bl] = jeffrey_interval(y, baseline_observations, alpha=ALPHA_CONFIDENCE)
[thetahat_tree, CIA_tree] = jeffrey_interval(y, tree_observations, alpha=ALPHA_CONFIDENCE)
[thetahat_logreg, CIA_logreg] = jeffrey_interval(y, logreg_observations, alpha=ALPHA_CONFIDENCE)
print(f"Basline, Thetahat: {thetahat_bl}, CI: {CIA_bl}")
print(f"Tree, Thetahat: {thetahat_tree}, CI: {CIA_tree}")
print(f"Logreg, Thetahat: {thetahat_logreg}, CI: {CIA_logreg}")

# Pair evaluation
observation_pairs = [
  (baseline_observations, tree_observations),
  (baseline_observations, logreg_observations),
  (tree_observations, logreg_observations)
]
pairlist: list[str] = ["bl-tree", "bl-logreg", "tree-logreg"]
for index, pair in enumerate(observation_pairs):
  obs1, obs2 = pair
  print("")
  print(pairlist[index])
  [thetahat, CI, p] = dtu_mcnemar(y, obs1, obs2, alpha=ALPHA_CONFIDENCE)
