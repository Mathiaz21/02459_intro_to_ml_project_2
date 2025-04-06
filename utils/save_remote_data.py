import os

from ucimlrepo import fetch_ucirepo
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
features_df = breast_cancer_wisconsin_diagnostic.data.features 
targets_df = breast_cancer_wisconsin_diagnostic.data.targets

def save_data() -> None:
  if not os.path.exists("local_data"):
    os.makedirs("local_data")
  features_df.to_csv('./local_data/breast_cancer_wisconsin_features.csv', index=False)
  targets_df.to_csv('./local_data/breast_cancer_wisconsin_targets.csv', index=False)


if __name__ == "__main__":
  save_data()