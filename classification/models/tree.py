import numpy as np
from sklearn.tree import DecisionTreeClassifier


ALPHA_START: float = 0.
ALPHA_STOP: float = .05
ALPHA_STEP: float = .005
ALPHAS: list[int] = np.arange(stop=ALPHA_STOP, start=ALPHA_START, step=ALPHA_STEP)


def generate_decision_tree(
    criterion: str = "gini", 
    min_sample_spit: int = 10,
    ccp_alpha: int = 0
) -> DecisionTreeClassifier:
  return DecisionTreeClassifier(
    criterion=criterion, 
    min_samples_split=min_sample_spit,
    ccp_alpha=ccp_alpha
  )