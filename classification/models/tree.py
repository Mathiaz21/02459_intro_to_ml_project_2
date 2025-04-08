from sklearn.tree import DecisionTreeClassifier


def generate_decision_tree(
    criterion: str = "gini", 
    min_sample_spit: int = 100,
    ccp_alpha: int = 0
) -> DecisionTreeClassifier:
  return DecisionTreeClassifier(
    criterion=criterion, 
    min_samples_split=min_sample_spit,
    ccp_alpha=ccp_alpha
  )